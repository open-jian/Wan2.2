"""Inspect the TI2V-5B decoder using meta tensors, without weights or a GPU.

This is a shape/cache audit, not a numerical or performance benchmark.
The source is loaded directly to avoid importing the full generation pipeline.
"""

import argparse
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

import torch

from vae_loader import load_vae_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latent-height', type=int, default=15)
    parser.add_argument('--latent-width', type=int, default=20)
    parser.add_argument('--latent-frames', type=int, default=3)
    parser.add_argument('--conv-config', type=Path, help='Decoder backend JSON config.')
    parser.add_argument('--plan-only', action='store_true',
                        help='Resolve layer backends without executing the decoder.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.latent_height, args.latent_width, args.latent_frames) < 1:
        parser.error('Latent dimensions must be positive.')

    root = Path(__file__).resolve().parents[1]
    source = root / 'wan/modules/vae2_2.py'
    module = load_vae_module()
    signature = inspect.signature(module.Wan2_2_VAE.__init__)
    defaults = {name: p.default for name, p in signature.parameters.items()}
    config = dict(dim=defaults['c_dim'], z_dim=defaults['z_dim'],
                  dim_mult=defaults['dim_mult'],
                  temperal_downsample=defaults['temperal_downsample'])
    with torch.device('meta'):
        model = module.WanVAE_(**config).eval()
    conv_config = module.load_conv_config(args.conv_config)
    conv_plan = model.configure_decoder_convolutions(conv_config)

    layers = []
    for name, conv in model.decoder.named_modules():
        if not isinstance(conv, module.CausalConv3d):
            continue
        eligible = (conv.kernel_size == (3, 3, 3) and conv.stride == (1, 1, 1)
                    and conv.dilation == (1, 1, 1) and conv.groups == 1)
        layers.append(dict(path='decoder.'+name, in_channels=conv.in_channels,
                           out_channels=conv.out_channels, kernel=list(conv.kernel_size),
                           stride=list(conv.stride), dilation=list(conv.dilation),
                           groups=conv.groups, causal_padding=list(conv._padding),
                           bias=conv.bias is not None, spatial_winograd_candidate=eligible))

    if args.plan_only:
        report = dict(scope='Configuration resolution only; no decoder execution.',
                      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      conv_config=conv_config, effective_backends=conv_plan, layers=layers)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({name: backend for name, backend in conv_plan.items()
                          if backend != 'native'}, indent=2))
        return

    calls = []
    stage_shapes = []
    current_chunk = [0]
    handles = []

    def make_conv_hook(name):
        def hook(conv, inputs, output):
            x = inputs[0]
            cache = inputs[1] if len(inputs) > 1 else None
            padding = list(conv._padding)
            cached_frames = 0 if cache is None else cache.shape[2]
            if cache is not None and conv._padding[4] > 0:
                padding[4] -= cached_frames
            padded = list(x.shape)
            padded[2] += cached_frames + padding[4] + padding[5]
            padded[3] += padding[2] + padding[3]
            padded[4] += padding[0] + padding[1]
            calls.append(dict(chunk=current_chunk[0], path='decoder.'+name,
                              input_shape=list(x.shape), cached_frames=cached_frames,
                              convolution_input_shape=padded,
                              output_shape=list(output.shape)))
        return hook

    def make_stage_hook(name):
        def hook(stage, inputs, output):
            stage_shapes.append(dict(chunk=current_chunk[0], path='decoder.'+name,
                                     input_shape=list(inputs[0].shape),
                                     output_shape=list(output.shape)))
        return hook

    for name, submodule in model.decoder.named_modules():
        if isinstance(submodule, module.CausalConv3d):
            handles.append(submodule.register_forward_hook(make_conv_hook(name)))
        if name in ['conv1', 'middle', 'head'] or name in [f'upsamples.{i}' for i in range(4)]:
            handles.append(submodule.register_forward_hook(make_stage_hook(name)))

    chunks = []
    with torch.no_grad():
        model.clear_cache()
        allocated_slots = model._conv_num
        z = torch.empty(1, defaults['z_dim'], args.latent_frames,
                        args.latent_height, args.latent_width, device='meta')
        projected = model.conv2(z)
        for i in range(args.latent_frames):
            current_chunk[0] = i
            index = [0]
            output = model.decoder(projected[:, :, i:i+1], feat_cache=model._feat_map,
                                   feat_idx=index, first_chunk=(i == 0))
            rgb = module.unpatchify(output, patch_size=2)
            expected = (1, 3, 1 if i == 0 else 4,
                        args.latent_height * 16, args.latent_width * 16)
            if tuple(rgb.shape) != expected:
                raise AssertionError((tuple(rgb.shape), expected))
            chunks.append(dict(index=i, rgb_shape=list(rgb.shape),
                               used_cache_slots=index[0],
                               cache_types=['tensor' if torch.is_tensor(x)
                                            else ('none' if x is None else str(x))
                                            for x in model._feat_map]))
    for handle in handles:
        handle.remove()

    summary = dict(decoder_parameters=sum(p.numel() for p in model.decoder.parameters()),
                   decoder_and_projection_parameters=sum(p.numel() for p in model.decoder.parameters())
                   + sum(p.numel() for p in model.conv2.parameters()),
                   residual_blocks=sum(isinstance(x, module.ResidualBlock)
                                       for x in model.decoder.modules()),
                   causal_conv3d_count=len(layers),
                   kernel_counts=dict(Counter('x'.join(map(str, x['kernel'])) for x in layers)),
                   candidate_count=sum(x['spatial_winograd_candidate'] for x in layers),
                   allocated_cache_slots=allocated_slots,
                   temporal_upsample=model.temperal_upsample)
    report = dict(scope='Meta-tensor shape/cache trace only; no weights, GPU execution, quality or speed test.',
                  source='wan/modules/vae2_2.py',
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  torch=torch.__version__, model_config=config,
                  conv_config=conv_config, effective_backends=conv_plan,
                  latent_shape=list(z.shape), summary=summary,
                  layers=layers, chunks=chunks, stage_shapes=stage_shapes, calls=calls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'summary': summary,
                      'chunks': [{k: v for k, v in x.items() if k != 'cache_types'} for x in chunks]}, indent=2))


if __name__ == '__main__':
    main()
