"""Check checkpoint loading and a small CUDA decode; not a quality/speed benchmark."""

import argparse
import hashlib
import json
import platform
import socket
import sys
from pathlib import Path

import torch

from vae_loader import load_vae_module


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--conv-config', type=Path, help='Decoder backend JSON config.')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != 'cuda':
        parser.error('This check is intended for a CUDA execution host.')

    source = Path(__file__).resolve().parents[1] / 'wan/modules/vae2_2.py'
    module = load_vae_module()
    conv_config = module.load_conv_config(args.conv_config)
    checkpoint_hash = sha256(args.checkpoint)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(0)
    vae = module.Wan2_2_VAE(vae_pth=str(args.checkpoint),
                          dtype=torch.bfloat16, device=device, conv_config=conv_config)
    latent = torch.randn(48, 3, 2, 2, device=device, dtype=torch.float32)
    with torch.inference_mode():
        output = vae.decode([latent])[0]
        repeated = vae.decode([latent])[0]
    torch.cuda.synchronize(device)
    expected = (3, 9, 32, 32)
    assert tuple(output.shape) == expected, (output.shape, expected)
    assert bool(output.isfinite().all()), 'Non-finite RGB output'
    assert torch.equal(output, repeated), 'Repeated decode differs after cache reset'
    props = torch.cuda.get_device_properties(device)
    report = {
        'scope': 'Environment/checkpoint/tiny-decode smoke check only; no quality or speed result.',
        'hostname': socket.gethostname(), 'platform': platform.platform(),
        'python': sys.version, 'executable': sys.executable,
        'torch': torch.__version__, 'cuda_build': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'gpu': props.name, 'gpu_total_bytes': props.total_memory,
        'compute_capability': list(torch.cuda.get_device_capability(device)),
        'conv_config': conv_config, 'effective_backends': vae.model.decoder_conv_plan(),
        'source': str(source), 'source_sha256': sha256(source),
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': checkpoint_hash,
        'weight_dtype': str(next(vae.model.parameters()).dtype),
        'autocast_dtype': str(vae.dtype), 'latent_shape': list(latent.shape),
        'output_shape': list(output.shape), 'output_dtype': str(output.dtype),
        'all_finite': True, 'repeated_decode_equal': True,
        'max_memory_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'max_memory_reserved_bytes': torch.cuda.max_memory_reserved(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
