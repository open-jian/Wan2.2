"""Matched native/Winograd measurements on the execution host; no training."""

import argparse
import hashlib
import json
import os
import random
import socket
import statistics
import time
from pathlib import Path

import torch

from vae_loader import load_vae_module


def differences(value, reference):
    delta = value.float()-reference.float()
    return dict(max_abs=delta.abs().max().item(), mae=delta.abs().mean().item(),
                relative_l2=(delta.norm()/reference.float().norm().clamp_min(1e-12)).item())


def psnr(value, target):
    mse = (value.float()-target.float()).square().mean((0, 2, 3))
    return (10*torch.log10(4/mse.clamp_min(1e-12))).mean().item()


def paired_timings(functions, repeats):
    values = {name: [] for name in functions}
    peak_extra = {name: [] for name in functions}
    rng = random.Random(23)
    for _ in range(repeats):
        order = list(functions)
        rng.shuffle(order)
        for name in order:
            baseline_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = functions[name]()
            end.record()
            end.synchronize()
            values[name].append(start.elapsed_time(end))
            peak_extra[name].append(torch.cuda.max_memory_allocated()-baseline_bytes)
            del result
    medians = {k: statistics.median(v) for k, v in values.items()}
    return dict(samples_ms=values, median_ms=medians,
                peak_extra_allocated_bytes={k: max(v) for k, v in peak_extra.items()},
                speedup=medians['native']/medians['candidate'],
                latency_reduction_percent=100*(1-medians['candidate']/medians['native']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['layers', 'decode', 'all'], default='all')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--conv-config', type=Path, default=Path('configs/vae_conv/winograd_2d.json'))
    parser.add_argument('--weight-dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--channels-last', action='store_true')
    parser.add_argument('--layer-backend', choices=['winograd_2d', 'winograd_3d'], default='winograd_2d')
    parser.add_argument('--layer-names', nargs='+', default=[
        'decoder.middle.0.residual.2', 'decoder.upsamples.1.upsamples.0.residual.2',
        'decoder.upsamples.2.upsamples.0.residual.2',
        'decoder.upsamples.2.upsamples.0.residual.6',
        'decoder.upsamples.3.upsamples.1.residual.2'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    module = load_vae_module()
    bank = torch.load(args.bank, map_location='cpu', weights_only=True, mmap=True)
    config = module.load_conv_config(args.conv_config)
    report = dict(hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  torch=torch.__version__, cuda_build=torch.version.cuda,
                  cudnn=torch.backends.cudnn.version(), conv_config=config, layer_backend=args.layer_backend,
                  bank_sha256=hashlib.sha256(args.bank.read_bytes()).hexdigest(),
                  precision=f'{args.weight_dtype} weights, BF16 autocast, eager, no compile or pruning',
                  channels_last_3d_weights=args.channels_last,
                  source_sha256={n: hashlib.sha256((root/'wan/modules'/n).read_bytes()).hexdigest()
                                 for n in ['vae2_2.py', 'winograd_conv.py', 'winograd_triton.py',
                                           'winograd_3d_conv.py', 'winograd_3d_triton.py']},
                  layers=[], videos=[], checks={})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')

    native = module.Wan2_2_VAE(vae_pth=str(args.checkpoint), device='cuda', dtype=torch.bfloat16)
    # Create parameters outside inference_mode so their version counters can
    # invalidate/reuse transformed weights. Inference tensors are not cacheable.
    candidate = None
    if args.stage in ('decode', 'all'):
        candidate = module.Wan2_2_VAE(vae_pth=str(args.checkpoint), device='cuda',
                                      dtype=torch.bfloat16, conv_config=config)
        assert not any(p.is_inference() for p in candidate.model.parameters())
    for vae in [native, candidate]:
        if vae is None:
            continue
        if args.weight_dtype == 'bfloat16':
            vae.model.to(dtype=torch.bfloat16)
        if args.channels_last:
            for layer in vae.model.modules():
                if isinstance(layer, torch.nn.Conv3d):
                    layer.to(memory_format=torch.channels_last_3d)
    modules = dict(native.model.named_modules())
    with torch.inference_mode():
        if args.stage in ('layers', 'all'):
            names = args.layer_names
            records, counts, handles = {}, {}, []
            def capture(name):
                def hook(layer, inputs):
                    count = counts.get(name, 0)
                    if count in (0, 2):
                        records[(name, count)] = tuple(x.clone() if x is not None else None for x in inputs)
                    counts[name] = count+1
                return hook
            for name in names:
                handles.append(modules[name].register_forward_pre_hook(capture(name)))
            native.decode([bank[0]['z'][0, :, :3].cuda()])
            for handle in handles:
                handle.remove()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                for (name, chunk_index), inputs in records.items():
                    layer = modules[name]
                    def direct():
                        layer._conv_backend = 'native'
                        return layer(*inputs)
                    def accelerated():
                        layer._conv_backend = args.layer_backend
                        return layer(*inputs)
                    expected = direct()
                    layer._winograd_weight_cache = None
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    actual = accelerated()
                    torch.cuda.synchronize()
                    cold_ms = 1000*(time.perf_counter()-started)
                    error = differences(actual, expected)
                    assert actual.isfinite().all()
                    for _ in range(3):
                        direct()
                        accelerated()
                    timing = paired_timings({'native': direct, 'candidate': accelerated}, args.repeats)
                    row = dict(layer=name, chunk=chunk_index, input_shape=list(inputs[0].shape),
                               cache_frames=0 if inputs[1] is None else inputs[1].shape[2],
                               error=error, cold_ms_including_weight_transform_and_jit=cold_ms, **timing)
                    report['layers'].append(row)
                    print(json.dumps(row), flush=True)
                    layer._conv_backend = 'native'
                    layer._winograd_weight_cache = None
                    save()
            del records, expected, actual

        if args.stage in ('decode', 'all'):
            report['effective_backends'] = candidate.model.decoder_conv_plan()
            for i, item in enumerate(bank):
                z, x = item['z'][0].cuda(), item['x'][0].cuda()
                expected = native.decode([z])[0]
                torch.cuda.synchronize()
                started = time.perf_counter()
                actual = candidate.decode([z])[0]
                torch.cuda.synchronize()
                first_ms = 1000*(time.perf_counter()-started)
                assert actual.shape == expected.shape and actual.isfinite().all()
                row = dict(role=item['role'], dataset=item['dataset'], index=item['index'],
                           shape=list(actual.shape), native_psnr_db=psnr(expected, x),
                           winograd_psnr_db=psnr(actual, x), output_difference=differences(actual, expected),
                           initial_call_wall_ms=first_ms)
                row['psnr_delta_db'] = row['winograd_psnr_db']-row['native_psnr_db']
                if item['role'] == 'evaluation':
                    native.decode([z])
                    candidate.decode([z])
                    row.update(paired_timings({'native': lambda: native.decode([z]),
                                              'candidate': lambda: candidate.decode([z])}, args.repeats))
                report['videos'].append(row)
                print(json.dumps(row), flush=True)
                if i == 0:
                    repeated = candidate.decode([z])[0]
                    assert torch.equal(repeated, actual)
                    prefix = candidate.decode([z[:, :3]])[0]
                    torch.testing.assert_close(prefix, actual[:, :9], atol=0, rtol=0)
                    altered = z.clone()
                    altered[:, 3:] += 3
                    perturbed = candidate.decode([altered])[0]
                    torch.testing.assert_close(perturbed[:, :9], actual[:, :9], atol=0, rtol=0)
                    report['checks'].update(repeated_decode_equal=True, prefix_causal_equal=True,
                                            future_perturbation_causal_equal=True)
                    report['derived_weight_cache_bytes'] = sum(
                        layer._winograd_weight_cache[1].numel()*layer._winograd_weight_cache[1].element_size()
                        for layer in candidate.model.decoder.modules()
                        if isinstance(layer, module.CausalConv3d) and layer._winograd_weight_cache is not None)
                    assert report['derived_weight_cache_bytes'] > 0, 'Weight caching was not exercised'
                save()
            evaluated = [row for row in report['videos'] if row['role'] == 'evaluation']
            report['evaluation_summary'] = dict(
                mean_native_ms=statistics.mean(row['median_ms']['native'] for row in evaluated),
                mean_winograd_ms=statistics.mean(row['median_ms']['candidate'] for row in evaluated),
                mean_psnr_delta_db=statistics.mean(row['psnr_delta_db'] for row in evaluated))
            summary = report['evaluation_summary']
            summary['speedup'] = summary['mean_native_ms']/summary['mean_winograd_ms']
            summary['latency_reduction_percent'] = 100*(1-summary['mean_winograd_ms']/summary['mean_native_ms'])
        report['status'] = 'complete'
        save()


if __name__ == '__main__':
    main()
