# Wan2.2 VAE 的 Winograd 接入设计

日期：2026-09-30。分支：`winograd`。基线提交：`1ea34ff48f87168174e12956e200b1c5ff`。

2026-09-30 实现更新：`winograd_2d` 已接入空间F(2×2,3×3)，包括PyTorch参考路径和CUDA FP16/BF16 Triton输入变换/矩阵乘法/输出还原，保留完整时间切片。配置和数学/缓存/CUDA合计17项测试通过。erebus最终4段81帧240×320对照：原精度基线1179.043→783.823ms，1.5042倍、耗时降33.52%、平均PSNR变化+0.000868dB；相对BF16+channels-last基线另有1.3865倍、耗时降27.88%。28层变换权重缓存额外1746MiB。逐段1.495—1.513倍，尚不能称稳定全面超过1.5。详情归档到主项目 `results/20260930_winograd_impl/RESULTS.md`。`winograd_3d`在后续本轮已实现，见下段。使用和限制见 `configs/vae_conv/README.md`。

2026-09-30 三维实现更新：新增`winograd_3d_conv.py`和`winograd_3d_triton.py`，实现F(2×2×2,3×3×3)，64个变换域矩阵乘法。所有三轴都变换，没有二维/原生回退；时间边界按调用内补零裁切，保留原历史缓存。新增全部三维和按时间形状混合的配置；26项检查及真实视频因果/重建验证通过；原精度基线1.6176倍、耗时降38.18%，相对BF16＋channels-last基线另有1.4895倍。摘要见[WINOGRAD_3D.md](WINOGRAD_3D.md)，原始结果见主项目`results/20260930_winograd_3d/RESULTS.md`。

以下结构和行号分析以页首 Git 基线为准；当前源码已增加配置代码，行号有变化。`docs/winograd_decoder_inventory.json` 保留接入前结构审计，更新后的 native 形状检查与兼容性证据在主项目 `results/20260930_conv_config/`。

## 1. 我们准备改什么

在原解码器内，为指定的 `3×3×3` 卷积增加一种计算实现。原权重、通道数、层数和 latent 接口保持一致；不需要通过训练重新学习卷积。目标是用 Winograd 的代数变换减少乘法，再验证实际速度和浮点误差。

修改 Wan 自己的 `CausalConv3d`，不意味着整个项目所有 `nn.Conv3d` 都被替换。建议给每个卷积实例设置默认关闭的后端选项，只在 `model.decoder` 下的明确层名上开启。编码器和其他模块继续走原路径。

## 2. 真实调用路径

```text
WanTI2V：wan/textimage2video.py
  └─ Wan2_2_VAE.decode：wan/modules/vae2_2.py:1038
       └─ WanVAE_.decode：812
            ├─ 恢复 latent 的均值/尺度
            ├─ model.conv2：48→48，1×1×1
            ├─ 每次取一个 latent 时间片
            │    └─ Decoder3d.forward：672
            │         ├─ conv1：48→1024
            │         ├─ middle：残差块 + attention + 残差块
            │         ├─ upsamples.0：3 个残差块 + 时空上采样
            │         ├─ upsamples.1：3 个残差块 + 时空上采样
            │         ├─ upsamples.2：3 个残差块 + 空间上采样
            │         ├─ upsamples.3：3 个残差块，无上采样
            │         └─ head：256→12
            └─ unpatchify：12 通道重排为 RGB，空间再放大 2 倍
```

这是 TI2V-5B 配置使用的 `Wan2.2_VAE.pth` 路径。本轮先处理 `vae2_2.py`；仓库名称为 Wan2.2 不代表其所有生成模型都使用这套 VAE。

实际 wrapper 的 `z_dim=48`、encoder 基宽 `c_dim=160`；`WanVAE_` 的 decoder 基宽为 `dec_dim=256`，不能用 encoder 的 160 来推算 decoder。实际时间下采样配置为 `[False, True, True]`，在构造 decoder 时反转，因此时间上采样是 **`[True, True, False]`**。直接采用 `Decoder3d` 的独立默认值会测错配置。

## 3. 已核实的结构和形状

使用 `benchmarks/inspect_vae_decoder.py`，直接导入本 fork 的源码，在 meta device 上构建模型和跟踪连续三个 latent。没有加载 checkpoint 或占用 GPU。完整清单见 `docs/winograd_decoder_inventory.json`。

- 解码器参数：555,049,228；加上外部 latent 投影 `model.conv2` 为 555,051,580。
- 残差块：middle 2 个，四个 upsamples 分组各 3 个，共 14 个。
- 每个残差块有两个 `3×3×3` 卷积，位置是 `residual.2`、`residual.6`，共 28 个。
- 加上输入 `decoder.conv1` 和输出 `decoder.head.2`，共 **30 个 `3×3×3` 候选**，全部 stride=1、dilation=1、groups=1。
- 此外还有 2 个时间上采样 `3×1×1`、2 个残差旁路 `1×1×1`，所以 decoder 内共有 34 个 `CausalConv3d`。
- 空间上采样中的 `Conv2d`、attention 内的投影、外部 `model.conv2` 也不属于这 30 个候选。

输入 latent 为 `[1,48,3,15,20]` 时，各阶段如下。表中 T 为首个 chunk / 后续 chunk 的输出时间长度；H×W 为特征图大小，尚不是最终 RGB 大小。

| 部分 | 主通道变化 | 残差块数 | 输出 T | 输出 H×W |
| --- | --- | --- | --- | --- |
| conv1 + middle | 48→1024，随后保持 1024 | 2 | 1 / 1 | 15×20 |
| upsamples.0 | 1024→1024 | 3 | 1 / 2 | 30×40 |
| upsamples.1 | 1024→1024 | 3 | 1 / 4 | 60×80 |
| upsamples.2 | 1024→512 | 3 | 1 / 4 | 120×160 |
| upsamples.3 | 512→256 | 3 | 1 / 4 | 120×160 |
| head + unpatchify | 256→12→3 | 0 | 1 / 4 | 240×320 |

连续三个 latent 的输出分别为 1、4、4 帧，总计 9 帧。通常总帧数为 `1 + 4*(T_latent-1)`。

## 4. 最小接入位置

`wan/modules/vae2_2.py:34` 的 `CausalConv3d.forward` 已经完成历史帧拼接和因果 padding，最后在第 42 行调用 `super().forward(x)`。**在这最后一步增加分支**，两条计算路径共用前面的缓存与补边。

当前空间版和三维版均接在这个位置，调用各自的Winograd计算函数并更新非持久权重缓存。以下为简化示意：

```python
# x 已经过原来的历史帧拼接与 F.pad
if self._conv_backend in ('winograd_2d', 'winograd_3d'):
    return selected_winograd_backend(x, self.weight, self.bias)
return super().forward(x)
```

建议文件分工：

| 文件 | 工作 |
| --- | --- |
| `wan/modules/vae2_2.py` | 已增加严格 JSON 解析、完整配置校验后切换、默认 native 的实例选项及后端检查；保留原类、权重和缓存逻辑 |
| `configs/vae_conv/` | 已提供原生、空间版、三维版、混合版四份配置和使用说明 |
| `wan/configs/wan_ti2v_5B.py`、`wan/textimage2video.py` | 已增加默认 None 的 `vae_conv_config` 字段，并将其传给 VAE |
| `wan/modules/winograd_conv.py` | 已实现空间Winograd参考路径、权重变换/失效管理、分块与精度选择 |
| `wan/modules/winograd_triton.py` | 已实现CUDA半精度的输入变换、16组GEMM和逆变换，FP32累加 |
| `benchmarks/` 下新增验证/测速脚本 | 明确加载本 fork；分别验证单层和完整解码器，对比两条路径 |

启用函数检查完整层名，列出实际启用的层；不存在的层名或非目标 kernel/stride/dilation/groups 直接报错。两种后端仅做推理，设备和精度条件见配置README。空间版没有原卷积回退；不同精度分别走Triton或PyTorch Winograd参考路径，不混称为同一性能实现。

### 必须保留的缓存语义

`ResidualBlock.forward` 和 decoder head 通过 `isinstance(layer, CausalConv3d)` 分配历史帧。替换成普通 `nn.Module` 包装器会使这些分支失效；因此优先保留原类和原参数，只切换类内的计算后端。

时间上采样的首 chunk 使用字符串 `"Rep"` 占位并跳过时间卷积，之后才逐步建立张量缓存。meta trace 已观察到槽位 11、18 的这个转换。空间 Winograd 不改时间上采样代码，不自行决定历史帧索引。

上游 `count_conv3d` 分配 34 个槽位，但实际每个 chunk 消耗 32 个：两个 `1×1×1` shortcut 不走缓存流程，最后两个槽位始终空。保留这种分配及顺序，不在本项目中顺手重排。

## 5. 怎么将二维 Winograd 用到三维卷积

先选择空间 `F(2×2,3×3)`，时间方向仍保留全部三个权重切片。已补边的输入记为 X，原卷积权重记为 W：

```text
Y[n,k,t,y,x] = Σ(c,τ,i,j) W[k,c,τ,i,j] * X[n,c,t+τ,y+i,x+j]
τ、i、j 都取 0、1、2。
```

PyTorch 这里执行互相关，不能反转权重。因果性由现有时间左侧补边和历史缓存保证。

每个输出 `2×2` 空间块读取 `4×4` 输入块；对每个时间切片做同一套二维变换：`U=G W Gᵀ`、`V=Bᵀ X_tile B`。每个变换位置将三个时间切片和输入通道一起求和，然后 `Aᵀ M A` 还原输出。

可组织成 16 组矩阵乘法：

```text
U[q, K, 3*C] @ V[q, 3*C, P] -> M[q, K, P]
q = 0..15
P = N * T_out * ceil(H_out/2) * ceil(W_out/2)
```

最后加一次 bias，并裁掉奇数边界的多余输出。单个完整空间输出块的核心乘积由 36 降到 16；时间方向仍然计算三项。这个 2.25 倍算术比值不是 GPU 速度承诺，还没有计入变换、搬运和边界浪费。

### 内存决定它是否值得做

朴素地把 V 全部展开，会复制很多输入。当前 240×320 视频配置中，最高分辨率组的第一个卷积为 C=512、K=256、T_out=4、H_out=120、W_out=160；其 V 就有 `16*1536*19200` 个元素，BF16 约 **900 MiB**。M 另需约 150 MiB；FP32 缓冲占用再翻倍。这些只是按形状估计的缓冲大小，不是 GPU 峰值实测。

因此参考版可先在小输入上验证公式，性能版必须控制 P 的分块大小，考虑复用相邻时间位置的输入变换、融合变换和写回，避免一次展开整个视频。权重变换后本身也从每空间核 9 个系数变成 16 个，不是权重压缩。

先用 PyTorch 写可对照的实现；确认瓶颈后再使用 Triton 融合变换。仅将数学公式拆成很多 Python/PyTorch 小算子，未必比 cuDNN 的原 Conv3d 更快。

## 6. 数值和权重管理

- 保持原 `state_dict` 的键和权重；变换权重是可重建缓存，不是新训练参数，不写入原 checkpoint。
- 加载新权重、原地更新权重、改变设备/精度或有效 autocast 精度时，必须使缓存失效；也覆盖 `load_state_dict(assign=True)`。`torch.compile` 接入前确定缓存的预生成和失效方式，避免每帧重建或重复编译。
- 首先用 FP64/FP32 检验公式，然后才测 BF16。实数等价不意味着浮点逐位一致。
- 历史原版是 FP32 权重加 BF16 autocast。新路径必须明确输入/权重进入变换前的有效精度，避免把精度变化冒充算法收益。BF16 GEMM 的累加、变换及逆变换精度需要分别记录和验证。
- 首轮输入/输出变换优先用 FP32 做误差对照，再研究是否降精度。变换后的系数再次量化会产生另一类误差，要实际测。

## 7. 实施与验证顺序

1. **单层参考实现**：用原权重、小尺寸和真实层形状，对照 native Conv3d；覆盖 bias、奇数空间边界、首帧和带历史的 chunk。确认公式、输出形状、误差。
2. **单层 GPU 测速**：先测 1024→1024、1024→512、512→512、256→256 等代表层，分别记录首 chunk 和后续 chunk；包含变换和写回成本，权重预变换时间另报。检查临时显存。层名 `decoder.upsamples.2.upsamples.0.residual.2` 可与旧研究对应，但不预设它最适合 Winograd。
3. **逐层启用**：根据当前 GPU 的实测挑选有收益的层，不因 30 层都满足公式就全部开启。常见的 256→256 高分辨率层可作为另一组候选；输入/输出窄通道卷积未必划算。
4. **完整 VAE 验证**：同一 encoder/latent、同一权重和执行口径，检查 RGB 误差、PSNR/SSIM/LPIPS 和时序误差；同时核验因果性、reset、交错双流、缓存稳定、权重重载及 native 开关回退。
5. **公平性能对照**：同卡交错测原版与 Winograd；另在相同 BF16、channels-last、compile 条件下比较增量收益，报告完整解码延迟、峰值显存、首次准备成本、倍率及耗时降幅。暂不叠加低秩或训练，避免混淆来源。

例如 100ms→80ms，应同时写 1.25 倍和耗时下降 20%。以前 1.515 倍来自精度/布局/编译/局部低秩的组合，不属于本轮 Winograd 已验证收益。

## 8. 本轮检查如何复现

在本仓库根目录运行（服务器使用父目录已有环境）：

```bash
source ../activate.sh
python benchmarks/inspect_vae_decoder.py --output docs/winograd_decoder_inventory.json
```

此脚本直接导入本 fork 的 `vae2_2.py`，不经过旧 profiles 适配器。完整参数、卷积层名、每次调用形状、缓存槽位状态和源码 SHA256 写入 JSON。meta 检查只确认结构与形状，不证明数值正确、因果性或加速。

接入前基线源码 SHA256：`eab5ce4aa1ce03af2978f2a8e8364c419f5fbb8535d265ac86b0b02ddcf0c1f6`。只有配置开关时的源码 SHA256：`e09e277bd895223aa8449479222276dac8ea2f4ee7ec95cee4bd1fb62826f1d1`。空间内核实现后的三份源码哈希随本轮评测JSON保存；原始权重未改。
