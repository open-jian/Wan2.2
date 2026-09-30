"""Numerical and cache contract tests; use CPU unless CUDA is explicitly enabled."""

import os
import functools
import importlib
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from vae_loader import load_vae_module

VAE = load_vae_module()
winograd = sys.modules[VAE.__package__ + '.winograd_conv'].spatial_winograd_conv3d
full_winograd = sys.modules[VAE.__package__ + '.winograd_3d_conv'].full_winograd_conv3d


class SpatialWinogradTest(unittest.TestCase):
    backend = 'winograd_2d'
    compute = staticmethod(winograd)

    def test_fp64_fp32_edges_batch_channels_bias_and_chunking(self):
        torch.manual_seed(13)
        for dtype in (torch.float64, torch.float32):
            for shape, k, bias in [((1, 1, 3, 3, 3), 1, False),
                                   ((2, 3, 6, 7, 9), 5, True),
                                   ((1, 7, 4, 8, 7), 3, False),
                                   ((2, 3, 5, 5, 7), 5, True)]:
                with self.subTest(dtype=dtype, shape=shape):
                    x = torch.randn(shape, dtype=dtype)
                    w = torch.randn(k, shape[1], 3, 3, 3, dtype=dtype)
                    b = torch.randn(k, dtype=dtype) if bias else None
                    expected = F.conv3d(x, w, b)
                    y, _ = self.compute(x, w, b, workspace_bytes=4096)
                    tol = 1e-10 if dtype == torch.float64 else 5e-5
                    torch.testing.assert_close(y, expected, atol=tol, rtol=tol)

    def test_causal_stream_matches_full_sequence_and_has_no_future_leak(self):
        torch.manual_seed(14)
        conv = VAE.CausalConv3d(3, 5, 3, padding=1).double().eval().requires_grad_(False)
        conv._conv_backend = self.backend
        x = torch.randn(1, 3, 7, 5, 7, dtype=torch.float64)
        with torch.no_grad():
            expected = F.conv3d(F.pad(x, (1, 1, 1, 1, 2, 0)), conv.weight, conv.bias)
            full = conv(x)
            outputs = []
            for start, end in [(0, 1), (1, 3), (3, 7)]:
                history = x[:, :, max(0, start-2):start] if start else None
                outputs.append(conv(x[:, :, start:end], history))
            torch.testing.assert_close(torch.cat(outputs, 2), full, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(full, expected, rtol=1e-12, atol=1e-12)
            changed = x.clone()
            changed[:, :, 4:] += 5
            torch.testing.assert_close(conv(changed)[:, :, :4], full[:, :, :4], rtol=0, atol=0)

    def test_cache_reuse_mutation_reload_dtype_and_parameter_replacement(self):
        conv = VAE.CausalConv3d(2, 3, 3, padding=1).eval().requires_grad_(False)
        conv._conv_backend = self.backend
        x = torch.randn(1, 2, 2, 3, 5)
        keys = list(conv.state_dict())
        with torch.no_grad():
            original = conv(x)
            cache = conv._winograd_weight_cache
            conv(x)
            self.assertIs(cache, conv._winograd_weight_cache)
            conv.weight.add_(0.1)
            mutated = conv(x)
            self.assertIsNot(cache, conv._winograd_weight_cache)
            self.assertFalse(torch.equal(original, mutated))
            expected = F.conv3d(F.pad(x, (1, 1, 1, 1, 2, 0)), conv.weight, conv.bias)
            torch.testing.assert_close(mutated, expected, atol=2e-5, rtol=2e-5)
            conv.load_state_dict({n: p.clone() for n, p in conv.state_dict().items()})
            self.assertIsNone(conv._winograd_weight_cache)
            conv(x)
            conv.load_state_dict({n: p.clone() for n, p in conv.state_dict().items()}, assign=True)
            self.assertIsNone(conv._winograd_weight_cache)
            conv(x)
            conv.double()
            self.assertIsNone(conv._winograd_weight_cache)
            conv(x.double())
            cache = conv._winograd_weight_cache
            conv.weight = torch.nn.Parameter(conv.weight.clone()+0.1, requires_grad=False)
            conv(x.double())
            self.assertIsNot(cache, conv._winograd_weight_cache)
        self.assertEqual(list(conv.state_dict()), keys)

    def test_inference_weights_are_recomputed_without_version_counters(self):
        with torch.inference_mode():
            x = torch.randn(1, 2, 3, 5, 5)
            w = torch.randn(3, 2, 3, 3, 3)
            first, cache = self.compute(x, w)
            self.assertIsNone(cache)
            w.add_(0.5)
            second, cache = self.compute(x, w, cache=cache)
            self.assertIsNone(cache)
            self.assertFalse(torch.equal(first, second))
            torch.testing.assert_close(second, F.conv3d(x, w), atol=5e-5, rtol=5e-5)

    def test_training_and_autograd_are_explicitly_rejected(self):
        conv = VAE.CausalConv3d(2, 3, 3, padding=1)
        conv._conv_backend = self.backend
        x = torch.randn(1, 2, 1, 3, 3)
        with self.assertRaisesRegex(RuntimeError, 'eval'):
            conv(x)
        conv.eval()
        with self.assertRaisesRegex(RuntimeError, 'inference'):
            conv(x)

    @unittest.skipUnless(os.environ.get('WAN_TEST_CUDA') == '1', 'GPU tests require explicit opt-in')
    def test_cuda_weight_cache_does_not_cross_execution_streams(self):
        conv = VAE.CausalConv3d(7, 5, 3, padding=1).cuda().eval().requires_grad_(False)
        conv._conv_backend = self.backend
        x = torch.randn(1, 7, 2, 5, 7, device='cuda')
        torch.cuda.synchronize()
        first, second = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.no_grad(), torch.cuda.stream(first), torch.autocast('cuda', dtype=torch.bfloat16):
            a = conv(x)
            cache = conv._winograd_weight_cache
        with torch.no_grad(), torch.cuda.stream(second), torch.autocast('cuda', dtype=torch.bfloat16):
            b = conv(x)
            self.assertIsNot(cache, conv._winograd_weight_cache)
        first.synchronize()
        second.synchronize()
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    @unittest.skipUnless(os.environ.get('WAN_TEST_CUDA') == '1', 'GPU tests require explicit opt-in')
    def test_cuda_low_precision_strides_edges_and_autocast_cache(self):
        torch.manual_seed(15)
        for dtype in (torch.float16, torch.bfloat16):
            for layout in (torch.contiguous_format, torch.channels_last_3d):
                with self.subTest(dtype=dtype, layout=layout), torch.no_grad():
                    x = torch.randn(2, 7, 6, 7, 9, device='cuda', dtype=dtype).to(memory_format=layout)
                    w = torch.randn(5, 7, 3, 3, 3, device='cuda', dtype=dtype)*0.1
                    b = torch.randn(5, device='cuda', dtype=dtype)*0.1
                    y, _ = self.compute(x, w, b, workspace_bytes=16384)
                    expected = F.conv3d(x.float(), w.float(), b.float())
                    relative = (y.float()-expected).norm()/expected.norm()
                    self.assertLess(relative.item(), 0.02 if dtype == torch.bfloat16 else 0.003)
                    self.assertTrue(y.isfinite().all())
        conv = VAE.CausalConv3d(7, 5, 3, padding=1).cuda().eval().requires_grad_(False)
        conv._conv_backend = self.backend
        x = torch.randn(1, 7, 2, 5, 7, device='cuda')
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y = conv(x)
                cache = conv._winograd_weight_cache
            with torch.autocast('cuda', dtype=torch.float16):
                z = conv(x)
            self.assertEqual(y.dtype, torch.bfloat16)
            self.assertEqual(z.dtype, torch.float16)
            self.assertIsNot(cache, conv._winograd_weight_cache)


class FullWinogradTest(SpatialWinogradTest):
    backend = 'winograd_3d'
    compute = staticmethod(full_winograd)

    def test_switching_between_spatial_and_full_invalidates_transform(self):
        conv = VAE.CausalConv3d(3, 5, 3, padding=1).eval().requires_grad_(False)
        x = torch.randn(1, 3, 3, 5, 7)
        with torch.no_grad():
            expected = conv(x)
            for backend, shape in [('winograd_2d', (16, 5, 9)),
                                   ('winograd_3d', (64, 5, 3)),
                                   ('winograd_2d', (16, 5, 9))]:
                conv._conv_backend = backend
                actual = conv(x)
                self.assertEqual(tuple(conv._winograd_weight_cache[1].shape), shape)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    @unittest.skipUnless(os.environ.get('WAN_TEST_CUDA') == '1', 'GPU tests require explicit opt-in')
    def test_cuda_odd_time_tiles_causality_and_partition_roundoff(self):
        torch.manual_seed(33)
        conv = VAE.CausalConv3d(7, 5, 3, padding=1).cuda().eval().requires_grad_(False)
        conv._conv_backend = self.backend
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            x = torch.randn(2, 7, 7, 5, 7, device='cuda')
            full = conv(x)
            for cut in (1, 2, 3, 4, 5):
                changed = x.clone()
                changed[:, :, cut:] += 9
                torch.testing.assert_close(conv(changed)[:, :, :cut], full[:, :, :cut], rtol=0, atol=0)
            outputs = []
            for start, end in [(0, 1), (1, 3), (3, 7)]:
                history = x[:, :, max(0, start-2):start] if start else None
                outputs.append(conv(x[:, :, start:end], history))
            streamed = torch.cat(outputs, 2)
            # Different temporal tile alignment changes low-precision roundoff.
            self.assertLess(((streamed.float()-full.float()).norm()/full.float().norm()).item(), 0.02)


class SpatialFusedWinogradTest(SpatialWinogradTest):
    backend = 'winograd_2d_fused'
    compute = staticmethod(functools.partial(winograd, fused=True))


class FullFusedWinogradTest(FullWinogradTest):
    backend = 'winograd_3d_fused'
    compute = staticmethod(functools.partial(full_winograd, fused=True))

    @unittest.skipUnless(os.environ.get('WAN_TEST_CUDA') == '1', 'GPU tests require explicit opt-in')
    def test_partial_fusion_edges_chunking_and_optional_bias(self):
        fused = importlib.import_module(VAE.__package__+'.winograd_fused_triton')
        for full in (False, True):
            weights = importlib.import_module(VAE.__package__+('.winograd_3d_conv' if full else '.winograd_conv'))
            transform = weights._transform_weight_3d if full else weights._transform_weight
            for dtype in (torch.float16, torch.bfloat16):
                for bias in (False, True):
                    with self.subTest(full=full, dtype=dtype, bias=bias), torch.no_grad():
                        x = torch.randn(2,7,5,7,9,device='cuda',dtype=dtype).to(memory_format=torch.channels_last_3d)
                        w = torch.randn(5,7,3,3,3,device='cuda',dtype=dtype)*.1
                        b = torch.randn(5,device='cuda',dtype=dtype)*.1 if bias else None
                        u = transform(w,dtype,True)
                        out = torch.empty(2,5,3,5,7,device='cuda',dtype=dtype,
                                          memory_format=torch.channels_last_3d)
                        fused.run_fused_winograd(x,u,b,out,17,full_3d=full,fuse_input=False)
                        expected = F.conv3d(x.float(),w.float(),None if b is None else b.float())
                        self.assertLess(((out.float()-expected).norm()/expected.norm()).item(),
                                        .02 if dtype == torch.bfloat16 else .003)


if __name__ == '__main__':
    unittest.main()
