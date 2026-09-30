"""Spatial F(2x2, 3x3) for valid 3D cross-correlation, inference only.

The caller supplies the original causal history and padding. All three temporal
weight slices are retained. CUDA FP16/BF16 uses Triton transforms and GEMM with
FP32 accumulation; other supported dtypes use a PyTorch numerical reference.
"""

import torch
import torch.nn.functional as F


WORKSPACE_BYTES = 128 * 1024 * 1024


def _effective_dtype(x, weight, bias):
    enabled = torch.is_autocast_enabled(x.device.type)
    if enabled and x.dtype != torch.float64 and weight.dtype != torch.float64:
        return torch.get_autocast_dtype(x.device.type)
    if x.dtype != weight.dtype or (bias is not None and bias.dtype != x.dtype):
        raise ValueError('Without autocast, input, weight and bias must have the same dtype.')
    return x.dtype


def _transform_weight(weight, dtype, fast):
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    w = weight.to(dtype).to(work_dtype)
    r0, r1, r2 = w.unbind(-2)
    rows = torch.stack((r0, (r0+r1+r2)*0.5, (r0-r1+r2)*0.5, r2), -2)
    c0, c1, c2 = rows.unbind(-1)
    u = torch.stack((c0, (c0+c1+c2)*0.5, (c0-c1+c2)*0.5, c2), -1)
    # [K,C,tau,4,4] -> [16,K,3*C], reduction order tau then C.
    u = u.permute(3, 4, 0, 2, 1).reshape(16, weight.shape[0], -1).contiguous()
    return u.to(dtype) if fast else u


def _cache_key(weight, dtype, fast):
    # Inference tensors have no version counter, so do not cache those weights.
    if weight.is_inference():
        return None
    # Never reuse a transform still being produced on another CUDA stream.
    stream = torch.cuda.current_stream(weight.device).cuda_stream if weight.is_cuda else None
    return (id(weight), weight._version, weight.data_ptr(), tuple(weight.shape),
            tuple(weight.stride()), weight.device, weight.dtype, dtype, fast, stream)


def _reference(x, u, bias, out, chunk):
    n, c, time, height, width = x.shape
    _, k, ot, oh, ow = out.shape
    th, tw = (oh+1)//2, (ow+1)//2
    x = F.pad(x.to(u.dtype), (0, ow % 2, 0, oh % 2))
    windows = x.unfold(3, 4, 2).unfold(4, 4, 2)
    bt = x.new_tensor([[1, 0, -1, 0], [0, 1, 1, 0],
                       [0, -1, 1, 0], [0, 1, 0, -1]])
    at = x.new_tensor([[1, 1, 1, 0], [0, 1, -1, -1]])
    for start in range(0, n*ot*th*tw, chunk):
        p = torch.arange(start, min(start+chunk, n*ot*th*tw), device=x.device)
        nn = p // (ot*th*tw)
        tt = (p // (th*tw)) % ot
        yy, xx = (p // tw) % th, p % tw
        d = torch.stack([windows[nn, :, tt+tau, yy, xx] for tau in range(3)], 1)
        v = bt @ d @ bt.T  # [P,3,C,4,4]
        v = v.permute(3, 4, 1, 2, 0).reshape(16, 3*c, len(p)).contiguous()
        m = torch.bmm(u, v).reshape(4, 4, k, len(p)).permute(2, 3, 0, 1)
        y = at @ m @ at.T  # [K,P,2,2]
        if bias is not None:
            y = y + bias.to(u.dtype)[:, None, None, None]
        for i in range(2):
            for j in range(2):
                valid = (2*yy+i < oh) & (2*xx+j < ow)
                out[nn[valid], :, tt[valid], 2*yy[valid]+i, 2*xx[valid]+j] = (
                    y[:, valid, i, j].T.to(out.dtype))


def spatial_winograd_conv3d(x, weight, bias=None, cache=None,
                           workspace_bytes=WORKSPACE_BYTES, *, fused=False):
    """Return (output, derived_weight_cache); never call native convolution.

    x: already padded NCTHW; weight: [K,C,3,3,3]. stride/dilation/groups are
    validated by CausalConv3d. The cache is transient and excluded from state_dict.
    """
    if x.ndim != 5 or weight.ndim != 5 or tuple(weight.shape[2:]) != (3, 3, 3):
        raise ValueError('Spatial Winograd requires NCTHW input and K,C,3,3,3 weights.')
    if x.device != weight.device or (bias is not None and bias.device != x.device):
        raise ValueError('Input, weight and bias must be on the same device.')
    if x.shape[1] != weight.shape[1] or min(x.shape[2:]) < 3:
        raise ValueError('Invalid padded input shape for valid 3x3x3 convolution.')
    if bias is not None and tuple(bias.shape) != (weight.shape[0],):
        raise ValueError('Bias must contain one value per output channel.')
    if type(workspace_bytes) is not int or workspace_bytes <= 0:
        raise ValueError('workspace_bytes must be a positive integer.')
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, weight, bias) if t is not None):
        raise RuntimeError('winograd_2d currently supports inference only; use no_grad/inference_mode.')
    shape = (x.shape[0], weight.shape[0], x.shape[2]-2, x.shape[3]-2, x.shape[4]-2)
    if x.device.type == 'meta':
        return x.new_empty(shape), None
    if x.device.type not in ('cpu', 'cuda'):
        raise ValueError('winograd_2d supports CPU reference and CUDA execution only.')
    dtype = _effective_dtype(x, weight, bias)
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f'Unsupported Winograd dtype: {dtype}')
    fast = x.is_cuda and dtype in (torch.float16, torch.bfloat16)
    work_dtype = dtype if fast else (torch.float64 if dtype == torch.float64 else torch.float32)
    with torch.autocast(device_type=x.device.type, enabled=False):
        key = _cache_key(weight, dtype, fast)
        if key is not None and cache is not None and cache[0] == key:
            u = cache[1]
        else:
            u = _transform_weight(weight, dtype, fast)
            cache = (key, u) if key is not None else None
        x = x.to(dtype)
        bias = bias.to(dtype) if bias is not None else None
        layout = torch.channels_last_3d if x.is_contiguous(memory_format=torch.channels_last_3d) else torch.contiguous_format
        out = torch.empty(shape, dtype=dtype, device=x.device, memory_format=layout)
        per_tile = 16 * (3*x.shape[1]*torch.empty((), dtype=work_dtype).element_size()
                         + shape[1]*(4 if fast else torch.empty((), dtype=work_dtype).element_size()))
        chunk = max(1, workspace_bytes // per_tile)
        if fast:
            from .winograd_triton import run_spatial_winograd
            if chunk >= 128:
                chunk = (chunk // 128) * 128
            with torch.cuda.device(x.device):
                if fused:
                    from .winograd_fused_triton import run_fused_winograd
                    run_fused_winograd(x, u, bias, out, chunk, full_3d=False)
                else:
                    run_spatial_winograd(x, u, bias, out, chunk)
        else:
            _reference(x, u, bias, out, chunk)
    return out, cache
