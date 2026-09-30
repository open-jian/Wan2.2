# Wan2.2 VAE full 3D Winograd

`winograd_3d` implements F(2×2×2,3×3×3) along time, height and width with the original weights and causal history. It is inference-only and requires no training. CUDA FP16/BF16 uses Triton; FP32/FP64 uses the PyTorch reference. The default remains native.

Select `configs/vae_conv/winograd_3d_all_residuals.json` for all 28 residual convolutions, or `winograd_mixed_temporal.json` for 10 spatial and 18 full 3D convolutions. The existing two-layer and mixed examples also work. Use the VAE's `conv_config` argument or `configure_decoder_convolutions`; see the configuration README.

Measured on one RTX PRO 6000 Blackwell Max-Q, four 81-frame 240×320 evaluation clips, seven randomized native/candidate pairs per clip, warm eager decoding:

| Configuration | Paired native ms | Winograd ms | Speedup | Latency reduction | PSNR change dB | Extra cache MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Spatial, all 28 layers (remeasured) | 1167.661 | 773.104 | 1.5104× | 33.79% | +0.000868 | 1746 |
| Full 3D, all 28 layers | 1159.412 | 716.727 | 1.6176× | 38.18% | +0.001224 | 2328 |
| Mixed: 10 spatial + 18 full 3D | 1177.861 | 721.359 | 1.6328× | 38.76% | +0.001104 | 2008 |
| Full 3D, BF16 weights + channels-last baseline | 1085.845 | 728.981 | 1.4895× | 32.87% | +0.002377 | 2328 |

Each row uses its own paired native run. The mixed and full 3D latencies are close (721/717 ms), and their native runs also vary; these measurements do not establish a stable mixed-versus-3D winner. PSNR changes are near zero, not evidence of better quality. The BF16/channels-last result is 1.4895×, below a strict 1.5× target. No compile, pruning or training was added.

Full 3D reduces the core products for an entire 2×2×2 output tile from 216 to 64. This arithmetic ratio is not a runtime speedup. A one-frame call crops half the temporal output tile, which can be slower than spatial Winograd. The mixed preset keeps layers that normally produce one frame on the spatial backend. A layer configured as 3D always executes the full 3D transform, including one-frame calls; it never silently falls back.

`winograd_3d_conv.py` supplies weight transforms, the numerical reference and cache management; `winograd_3d_triton.py` supplies input transforms, 64 GEMMs and output transforms. GEMM reduction is over input channels. Transforms and accumulation use FP32 in the CUDA low-precision path. Output boundaries are masked, bias is added once, and the caller retains padding and history handling.

26 tests passed, including FP64/FP32 math, CUDA BF16/FP16, odd time/space tiles, layouts, causal prefixes, cache invalidation and backend switching. Native checkpoint and encoder compatibility were checked against the original Git source, and real-video repeat/prefix/future-perturbation checks passed. Changing temporal chunk alignment can change low-precision roundoff; it is not guaranteed bitwise equal across different chunk partitions.

The full 3D transformed-weight cache adds 2328 MiB (2.273 GiB), versus 1746 MiB for spatial and 2008 MiB for mixed. The 128 MiB workspace target covers only V/M tile buffers, not all model memory. Timing excludes cold JIT and weight transforms. Higher resolutions, longer videos, LPIPS/SSIM, other GPUs and full-model torch.compile remain unverified.

Reproduce with `benchmarks/benchmark_winograd.py --conv-config configs/vae_conv/winograd_3d_all_residuals.json --layer-backend winograd_3d`, supplying the checkpoint, RGB/latent bank and output path. CUDA tests require an idle selected GPU and `WAN_TEST_CUDA=1`. Tested with torch2.7.1+cu128 and Triton3.3.1. Raw evidence is archived in the parent research workspace at `results/20260930_winograd_3d/`.
