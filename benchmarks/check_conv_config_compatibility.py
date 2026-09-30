"""Compare native configuration paths against the unmodified Git baseline on CUDA."""

import argparse
import hashlib
import json
import socket
import subprocess
import types
from pathlib import Path

import torch

from vae_loader import load_vae_module


def load_source(source, name, filename):
    module = types.ModuleType(name)
    exec(compile(source, filename, 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reference_commit = '1ea34ff48f87168174e12956e200b1d908b1c5ff'
    reference_source = subprocess.check_output(
        ['git', 'show', f'{reference_commit}:wan/modules/vae2_2.py'], cwd=root)
    current_source = (root / 'wan/modules/vae2_2.py').read_bytes()
    reference_module = load_source(reference_source, 'vae_original', '<git baseline>')
    current_module = load_vae_module()
    options = dict(vae_pth=str(args.checkpoint), device='cuda:0', dtype=torch.bfloat16)
    torch.manual_seed(0)
    reference = reference_module.Wan2_2_VAE(**options)
    current = current_module.Wan2_2_VAE(**options)
    ref_state, state = reference.model.state_dict(), current.model.state_dict()
    assert list(ref_state) == list(state), 'Checkpoint keys changed'
    assert all(torch.equal(ref_state[n], state[n]) for n in state), 'Checkpoint tensors changed'
    checks = {'checkpoint_keys_and_tensors_equal': True}
    latent = torch.randn(48, 3, 2, 3, device='cuda:0')
    video = torch.randn(3, 9, 32, 48, device='cuda:0').clamp(-1, 1)
    with torch.inference_mode():
        expected = reference.decode([latent])[0]
        assert torch.equal(current.decode([latent])[0], expected)
        checks['default_native_bitwise_equal'] = True
        current.configure_decoder_convolutions(root / 'configs/vae_conv/native.json')
        assert torch.equal(current.decode([latent])[0], expected)
        checks['native_json_bitwise_equal'] = True
        mixed = current_module.load_conv_config(root / 'configs/vae_conv/mixed.json')
        mixed['enabled'] = False
        current.configure_decoder_convolutions(mixed)
        assert torch.equal(current.decode([latent])[0], expected)
        checks['disabled_mixed_bitwise_equal'] = True

        # A configured but unavailable backend must fail before executing conv1.
        calls = []
        handle = current.model.decoder.conv1.register_forward_pre_hook(
            lambda layer, inputs: calls.append(True))
        errors = {}
        for backend in ['winograd_3d']:
            current.configure_decoder_convolutions(root / f'configs/vae_conv/{backend}.json')
            current.model.clear_cache()
            cache = current.model._feat_map
            try:
                current.decode([latent])
            except NotImplementedError as exc:
                assert backend in str(exc)
                errors[backend] = str(exc)
            else:
                raise AssertionError(f'{backend} silently executed another backend')
            assert not calls and current.model._feat_map is cache
        handle.remove()
        checks['unimplemented_fail_before_convolution_and_cache_mutation'] = True
        assert torch.equal(current.encode([video])[0], reference.encode([video])[0])
        checks['encoder_bitwise_equal_with_decoder_winograd_selected'] = True
        current.configure_decoder_convolutions(None)
        assert torch.equal(current.decode([latent])[0], expected)
        checks['restore_native_after_failure_bitwise_equal'] = True
    torch.cuda.synchronize()
    report = dict(
        scope='Native compatibility/configuration test only; spatial Winograd tested separately.',
        hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__, cuda_build=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), reference_commit=reference_commit,
        reference_source_sha256=hashlib.sha256(reference_source).hexdigest(),
        current_source_sha256=hashlib.sha256(current_source).hexdigest(),
        latent_shape=list(latent.shape), rgb_shape=list(expected.shape),
        weight_dtype='float32', autocast_dtype='bfloat16', checks=checks,
        expected_errors=errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
