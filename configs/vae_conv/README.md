# 按配置选择 VAE 卷积后端

状态：**`winograd_2d` 和 `winograd_3d` 均已实现并接入实际推理**。二维模式为空间 F(2×2,3×3)，三维模式为时间/空间 F(2×2×2,3×3×3)，均保留原三维卷积权重和因果历史，不调用原卷积来冒充 Winograd。

三维实现与实测摘要见 [WINOGRAD_3D.md](../../docs/WINOGRAD_3D.md)。

新增实验入口 `winograd_2d_fused` / `winograd_3d_fused`，对应 `_fused_all_residuals.json` 配置。**首轮实测更慢，暂不推荐用于加速。**实现、源码依据和限制见 [WINOGRAD_FUSION.md](../../docs/WINOGRAD_FUSION.md)；原有模式保持不变。

## 配置格式

```json
{
  "schema_version": 1,
  "enabled": true,
  "layers": {
    "decoder.upsamples.2.upsamples.0.residual.2": "winograd_2d",
    "decoder.upsamples.2.upsamples.0.residual.6": "winograd_3d"
  }
}
```

- `enabled` 是总开关。设为 `false` 后，所有层都使用 `native`，保留的 layers 内容仍会校验。
- `layers` 按**精确层名**逐个选择；不支持通配符。未列出的层自动使用 `native`，也可以显式写 `native`。
- `winograd_2d` 表示只对空间高/宽轴做 Winograd，原三维卷积的时间权重和计算仍保留。
- `winograd_3d` 表示时间/高/宽三轴一起做 Winograd。
- 所有 Winograd 入口只允许 decoder 的 `3×3×3`、stride=1、dilation=1、groups=1 卷积。编码器、latent 投影及非目标算子不可选。
- 错误字段、未知模式、重复 JSON key、拼错层名或不支持的卷积形状会报错。整份配置校验通过后才替换旧配置，失败不会只改一部分层。
- 每次加载配置都会完整替换上一次选择，不会累计；配置不写入模型权重，复现时同时保存 JSON 和生成的生效层清单。

## 现成配置

| 文件 | 作用 |
| --- | --- |
| `native.json` | 原版卷积，所有加速关闭 |
| `winograd_2d.json` | 为第三组第一个残差块的两个重卷积选择空间 Winograd |
| `winograd_3d.json` | 为同样两层选择三维 Winograd |
| `mixed.json` | 第一层选空间版，第二层选三维版，展示逐层混合配置 |
| `winograd_2d_last_two_stages.json` | 最后两组的12个残差卷积选择空间版，供范围消融 |
| `winograd_2d_all_residuals.json` | 全部14个残差块的28个卷积选择空间版，供范围消融 |
| `winograd_3d_all_residuals.json` | 同样28层全部使用完整三维版，首个/奇数帧裁切无效输出 |
| `winograd_mixed_temporal.json` | middle和upsamples.0用二维，其余18个残差卷积用三维，供单帧/多帧形状对照 |

这些层只是配置和后续同层消融的起点，不代表已经筛选出最优加速位置。

## 预览生效层，不运行 GPU

在 Wan2.2 仓库根目录运行：

```bash
python benchmarks/inspect_vae_decoder.py \
  --conv-config configs/vae_conv/mixed.json \
  --plan-only --output /tmp/vae_conv_plan.json
```

屏幕打印非 native 的层；JSON 保存输入配置、全部 34 个 CausalConv3d 的实际后端、层形状参数和源码哈希。不开 `--plan-only` 时进行 meta 形状跟踪，支持二维、三维及混合配置。

## VAE 调用入口

```python
vae = Wan2_2_VAE(
    vae_pth="/path/to/Wan2.2_VAE.pth",
    device="cuda",
    conv_config="/path/to/configs/vae_conv/native.json",
)

# 也可在已有模型上切换；返回完整生效层清单。
plan = vae.configure_decoder_convolutions("/path/to/another_config.json")

# 恢复全部原生卷积。
vae.configure_decoder_convolutions(None)
```

TI2V-5B 原生成管线读取 `wan/configs/wan_ti2v_5B.py` 中新增的 `vae_conv_config` 字段，默认 `None`。可以将其设为 JSON 文件的绝对路径；管线会传给 VAE。当前没有新增 generate.py 的命令行参数。

独立 GPU 环境检查入口也接受 `--conv-config`，并在结果中保存输入配置和实际生效清单：

```bash
python benchmarks/check_vae_environment.py \
  --checkpoint /home/jian/vae-speedup/assets/wan22/Wan2.2_VAE.pth \
  --conv-config configs/vae_conv/native.json \
  --output /home/jian/vae-speedup/results/native_environment.json
```

在当前项目中，GPU 检查应经 ada0 调度到 erebus，并在启动前确认空闲卡。原 checkpoint、通道和层数保留，不需要恢复训练；速度和数值变化必须同卡配对测量。

## 二维/三维实现与运行限制

- `wan/modules/winograd_conv.py`：权重变换、有效 autocast 精度、派生缓存、分块，以及 FP32/FP64 PyTorch 参考实现。
- `wan/modules/winograd_triton.py`：CUDA FP16/BF16 输入变换、16组矩阵乘法、输出逆变换。变换用FP32计算，变换系数存为有效低精度，乘法累加/中间结果/逆变换为FP32，输出转回有效精度。
- `wan/modules/winograd_3d_conv.py`、`winograd_3d_triton.py`：完整三维变换和64组GEMM，沿输入通道求和；时间输出成对计算，奇数尾部补零并裁切，不额外等待下一帧，单帧也使用完整三维算法。
- CPU和CUDA的FP32/FP64使用参考路径；CUDA FP16/BF16使用Triton。需要支持所选精度的GPU及Triton，不会静默回退原卷积。当前验证环境为erebus的sm_120、torch2.7.1+cu128、Triton3.3.1。
- 仅支持推理：模型须 `eval()`，需要梯度时明确报错。默认 `native` 的训练行为保持原状。整网 `torch.compile` 尚未验证，本轮按eager运行。
- 按输入tile分块，默认目标为V/M两类缓冲合计128MiB；原输入/输出、转换副本和权重缓存不包含在这个预算内。PyTorch参考路径另有patch/变换临时张量，不能把128MiB称为全模型峰值显存。
- 派生权重不进入state_dict。在原地更新、替换参数、load_state_dict（含assign=True）、切换精度/设备/布局、autocast精度或CUDA stream后重建，避免跨stream读取尚未生成的变换。避免使用绕过版本计数的 `.data` 写权重；这类操作之后应重新应用配置来清缓存。
- **在 `inference_mode()` 外创建和加载模型，再进入推理上下文执行。**在inference_mode内创建的权重没有版本计数，为保证修改后不会使用旧结果，此实现不缓存它们，每次重新变换，性能会不同。

三维版每个完整2×2×2输出块的核心乘积由216变成64，算术比值3.375倍；这不等于实际加速倍率。每次只输出1帧时，时间tile的另一半输出被裁切，可能比二维慢。三维派生权重为64/27倍原卷积元素数，比二维的48/27倍更多。跨调用改变时间分块边界时，浮点舍入可能不同，不能要求低精度逐位相同。

数学、缓存和CUDA检查：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
# 确认空闲GPU并设置CUDA_VISIBLE_DEVICES之后：
WAN_TEST_CUDA=1 python -m unittest discover -s tests -p 'test_*.py' -v
```

整网与单层评测入口为 `benchmarks/benchmark_winograd.py`；二维历史结果归档到父目录 `results/20260930_winograd_impl/`，本轮三维评测归档到 `results/20260930_winograd_3d/`。单层测试用 `--layer-backend winograd_3d` 指定后端；完整解码测试按 `--conv-config` 逐层生效，JSON的candidate计时对应所选配置。
