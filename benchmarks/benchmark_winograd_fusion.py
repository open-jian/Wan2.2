"""Compare fusion on captured decoder inputs; excludes padding/weight preparation.

This is a kernel diagnostic, not an end-to-end decoder speedup measurement.
The bounded sweep records resource failures as well as successful candidates.
"""

import argparse
import hashlib
import importlib
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from vae_loader import load_vae_module
from benchmark_winograd import differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    module = load_vae_module()
    package = module.__package__
    fused = importlib.import_module(package+'.winograd_fused_triton')
    w2 = importlib.import_module(package+'.winograd_conv')
    w3 = importlib.import_module(package+'.winograd_3d_conv')
    k2 = importlib.import_module(package+'.winograd_triton')
    k3 = importlib.import_module(package+'.winograd_3d_triton')
    vae = module.Wan2_2_VAE(vae_pth=args.checkpoint, device='cuda', dtype=torch.bfloat16)
    bank = torch.load(args.bank, map_location='cpu', weights_only=True, mmap=True)
    names = ['decoder.middle.0.residual.2',
             'decoder.upsamples.1.upsamples.0.residual.2',
             'decoder.upsamples.2.upsamples.0.residual.2',
             'decoder.upsamples.2.upsamples.0.residual.6',
             'decoder.upsamples.3.upsamples.1.residual.2']
    modules = dict(vae.model.named_modules())
    records, counts = {}, {}
    def capture(name):
        def hook(layer, inputs):
            count = counts.get(name, 0)
            if count == 2:
                records[name] = tuple(x.clone() if x is not None else None for x in inputs)
            counts[name] = count+1
        return hook
    handles = [modules[name].register_forward_pre_hook(capture(name)) for name in names]
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(fused.__file__).parent.glob('winograd*.py')},
                  precision='BF16, FP32 transform/accumulation', layers=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    with torch.inference_mode():
        vae.decode([bank[0]['z'][0, :, :3].cuda()])
        for handle in handles:
            handle.remove()
        for name, inputs in records.items():
            layer = modules[name]
            x, history = inputs
            pad = list(layer._padding)
            if history is not None:
                x = torch.cat([history, x], 2)
                pad[4] -= history.shape[2]
            x = F.pad(x, pad).to(torch.bfloat16)
            w = layer.weight.to(x.dtype)
            b = layer.bias.to(x.dtype) if layer.bias is not None else None
            out = torch.empty((x.shape[0], w.shape[0], *(d-2 for d in x.shape[2:])),
                              device=x.device, dtype=x.dtype)
            expected = F.conv3d(x, w, b)
            row = dict(layer=name, padded_shape=list(x.shape), output_shape=list(out.shape), modes={})
            report['layers'].append(row)
            for full in (False, True):
                u = (w3._transform_weight_3d if full else w2._transform_weight)(w, x.dtype, True)
                chunk = max(1, (128*1024*1024)//((64 if full else 16)*
                            ((x.shape[1] if full else 3*x.shape[1])*2+w.shape[0]*4)))
                if chunk >= 128:
                    chunk = (chunk//128)*128
                run_old = k3.run_full_winograd if full else k2.run_spatial_winograd
                functions = {'native': lambda: F.conv3d(x, w, b),
                             'separate': lambda: run_old(x, u, b, out, chunk)}
                records_mode = row['modes']['3d' if full else '2d'] = {}
                # Small bounded resource sweep, not an exhaustive autotuner.
                tiles = [(16,16,16,8), (16,16,32,8), (16,32,16,8), (32,16,16,8)]
                if not full:
                    tiles += [(32,32,32,8), (32,32,32,4)]
                for fuse_input in (False, True):
                    for bm,bn,bk,warps in tiles:
                        key = f'{"all" if fuse_input else "output"}_{bm}_{bn}_{bk}_w{warps}'
                        def run(fi=fuse_input,m=bm,n=bn,k=bk,nw=warps):
                            return fused.run_fused_winograd(x,u,b,out,chunk,full_3d=full,
                                fuse_input=fi,block_m=m,block_n=n,block_k=k,num_warps=nw)
                        try:
                            kernels = run()
                            torch.cuda.synchronize()
                            error = differences(out, expected)
                            assert torch.isfinite(out).all() and error['relative_l2'] < .03, error
                            records_mode[key] = dict(error=error, registers=kernels[0].n_regs,
                                spills=kernels[0].n_spills, shared_bytes=kernels[0].metadata.shared)
                            functions[key] = run
                        except (RuntimeError, AssertionError, triton.OutOfResources) as exc:
                            records_mode[key] = dict(error_message=str(exc)[:1500])
                        print(name, '3d' if full else '2d', key, records_mode[key], flush=True)
                        save()
                for fn in functions.values():
                    for _ in range(2):
                        fn()
                samples = {key: [] for key in functions}
                rng = random.Random(19)
                for _ in range(args.repeats):
                    order = list(functions)
                    rng.shuffle(order)
                    for key in order:
                        start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        start.record()
                        functions[key]()
                        end.record(); end.synchronize()
                        samples[key].append(start.elapsed_time(end))
                for key, values in samples.items():
                    records_mode.setdefault(key, {}).update(samples_ms=values,
                                                           median_ms=statistics.median(values))
                print('TIMINGS', name, full, {k: round(statistics.median(v),4) for k,v in samples.items()}, flush=True)
                save()


if __name__ == '__main__':
    import triton
    main()
