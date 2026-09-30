# Winograd 融合实验

本轮实现了可运行的二维/三维融合内核，但首轮真实层实验**比现有分步 Winograd 慢**。这是研究用可选后端，不是新的加速推荐。默认 `native` 和已有 `winograd_2d` / `winograd_3d` 保持原路径。

## 源码依据与实现范围

参考作者 Neon 的固定提交 [`8c3fb8a93b4a`](https://github.com/NervanaSystems/neon/tree/8c3fb8a93b4a89303467b25817c60536542d08bd)，重点阅读 `neon/backends/winograd_conv.py` 的 FX 调度和 `neon/backends/kernels/sass/xconv_winograd_2x2_3x3_32x32.sass`。这是 2019 年维护版本，不能等同于 2016 年论文测速提交。

借鉴其外置静态权重变换、同一 block 内输入变换/乘加/输出还原的设计。本实现独立用 Triton 编写，没有移植旧 Maxwell SASS 的指令调度，也没有实现与其等价的手工共享内存双缓冲。

`wan/modules/winograd_fused_triton.py` 使用 batched Tensor Core dot，沿输入通道累加，随后在同一个内核内逆变换并写最终卷积输出。权重预变换和原派生缓存继续保留。

| 路径 | 输入变换 V | 乘加结果 M | 对外入口 |
| --- | --- | --- | --- |
| 原分步版 | 全局显存 | 全局显存 | `winograd_2d` / `winograd_3d` |
| 本轮全融合 | block 内 | block 内 | `winograd_2d_fused` / `winograd_3d_fused` |
| 本轮部分融合 | 全局显存 | block 内 | 单层诊断脚本，`fuse_input=False` |

“block 内”表示不分配显式全局 V/M 张量；寄存器溢出仍可能产生 local-memory 访问。因此不能把融合直接等同于没有显存访问。原输入、输出和变换权重仍在显存。

## 配置与验证

沿用 schema_version=1 的逐层 JSON。`configs/vae_conv/winograd_2d_fused_all_residuals.json`、`winograd_3d_fused_all_residuals.json` 将 28 个残差卷积切到实验路径；可以只选个别层。训练限制、原权重、因果历史、奇数尾部裁切和缓存失效规则不变。

CUDA FP16/BF16 执行融合内核；CPU、CUDA FP32/FP64 仍执行相同数学参考路径，不能用参考路径证明融合性能。当前只验证 erebus 的 sm_120 / torch 2.7.1+cu128 / Triton 3.3.1；不是跨 GPU 的调优实现。

```bash
# 指定空闲 GPU 后运行；检查数学、缓存、布局、分块和因果性。
WAN_TEST_CUDA=1 python -m unittest discover -s tests -p 'test_*.py' -v

# 捕获同一批真实层输入，扫描少量分块并记录寄存器/溢出/共享内存。
python benchmarks/benchmark_winograd_fusion.py \
  --checkpoint /path/to/Wan2.2_VAE.pth --bank /path/to/eval_bank.pt \
  --output /path/to/layer_sweep.json

# 同一进程随机交错比较 native、全融合与已有三维版。
python benchmarks/benchmark_winograd.py \
  --checkpoint /path/to/Wan2.2_VAE.pth --bank /path/to/eval_bank.pt \
  --stage decode --repeats 5 \
  --conv-config configs/vae_conv/winograd_3d_fused_all_residuals.json \
  --comparison-config configs/vae_conv/winograd_3d_all_residuals.json \
  --output /path/to/decode_fused.json
```

## 首轮发现

43 项测试通过，包含融合路径的 CUDA FP16/BF16、channels-last、奇数边界、多 batch、缓存更新、跨 stream、未来扰动和时间分块。部分融合也检查了带/不带 bias 及跨 workspace chunk 的边界。

在 5 个真实层形状上，二维扫描 12 组融合设置，三维扫描 8 组（每种形状其中 2 组超过共享内存上限）。本轮成功运行的全融合与部分融合均未超过已有分步实现。诊断计时排除了因果 padding、权重转换和变换准备，不能与完整 decoder 的毫秒数混用。

以稳定 chunk 的 `decoder.upsamples.2.upsamples.0.residual.2` 为例，输出为 `[1,512,4,60,80]`：三维分步内核约 1.142 ms，所扫描最快全融合约 15.594 ms，最快部分融合约 4.398 ms。这不是融合的一般性能上限，只是当前实现和有限分块的实测结果。

资源证据与解释：

- 初始全融合 `BM=BN=BK=16, num_warps=8`；三维编译结果达到 255 个寄存器/线程、64 KiB 共享内存。扩大分块时出现寄存器溢出，部分设置需要 128 KiB 共享内存，超过该卡每 block 的 99 KiB 上限。
- 根据扫描结果，当前实验默认二维改为 `BM=BN=BK=32`，三维改为 `BM=32, BN=BK=16`，均为 8 warps。三维的 96 KiB 共享内存与溢出代价仍高，但在本轮五种形状中速度优于初始 16 通道块。不是运行时自动调优，也未证明跨硬件最优。
- 输入变换在每个输出通道分块中重新执行。K=512 时，BM=16 会重复 32 次，BM=32 仍重复 16 次；分步版先计算一次 V 再复用。省去了 V 的写回，但增加了输入读取和变换。
- 逆变换要求同一 block 持有 16/64 组累加结果，限制矩阵乘分块大小。原分步版可以对每一组使用更大的 GEMM tile。结合部分融合仍较慢的结果，资源占用和矩阵计算效率是后续重点；尚未用硬件计数器分离每项因素的耗时比例。

下一步需重新设计变换坐标的线程分配、输入复用和流水调度，而不是把“内核数量减少”当成性能结论。当前保留已有三维后端作为速度推荐；完整解码对照与原始结果归档在父项目 `results/20260930_winograd_fused/`。

## 完整解码结果

erebus GPU2，FP32 原权重＋BF16 autocast，eager，无训练/剪枝/compile。使用原有 2 段 validation 和 4 段 evaluation，均为 81 帧、240×320；同一 latent 输入，计时仅为 decoder，包含其包装/归一化/输出转换，排除 encoder、DiT、加载、传输、首次权重变换和 JIT。下面是 4 段 evaluation 各自 5 次随机交错计时的中位数再取均值。

| 后端 | 完整解码 ms | 相对原生速度 | 相对原生耗时变化 | 对原 RGB 的平均 PSNR |
| --- | ---: | ---: | ---: | ---: |
| native | 1175.761 | 1.000× | 0% | 37.840749 dB |
| 已有三维分步版，28 层 | 726.168 | 1.619× | 降低 38.24% | 37.841972 dB |
| 三维全融合，28 层，调整分块后 | 3574.391 | 0.329× | 增加 204.01% | 37.841573 dB |

融合版耗时仍是已有三维版的 4.922 倍。初始 16 通道块的独立配对轮次为 5365.230 ms（同轮 native 1170.893 ms、分步版 712.466 ms）；调整分块有改善，仍没有新增收益。两轮原始结果均保留。

融合版相对 native 的平均 PSNR 变化 +0.000824 dB，6 段变化范围 −0.001038 到 +0.002644 dB；这是近似不变，不是质量改进或逐位无损声明。最终分块的 43 项测试以及完整视频重复解码/因果前缀/未来扰动检查均通过。未验证 LPIPS/SSIM、更大尺寸或其他 GPU；不外推这些结果。
