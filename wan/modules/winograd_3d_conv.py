"""Full F(2x2x2, 3x3x3) valid cross-correlation, inference only.

All three axes use Winograd transforms. Temporal tiles are local to the padded
call; an odd final output is cropped, never delayed until a future call. The
caller remains responsible for the original causal padding/history.
"""

import torch
import torch.nn.functional as F

from .winograd_conv import WORKSPACE_BYTES, _cache_key, _effective_dtype


def _g_axis(x, axis):
    a, b, c = x.unbind(axis)
    return torch.stack((a, (a+b+c)*0.5, (a-b+c)*0.5, c), axis)


def _bt_axis(x, axis):
    a, b, c, d = x.unbind(axis)
    return torch.stack((a-c, b+c, c-b, b-d), axis)


def _at_axis(x, axis):
    a, b, c, d = x.unbind(axis)
    return torch.stack((a+b+c, b-c-d), axis)


def _transform_weight_3d(weight, dtype, fast):
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    u = weight.to(dtype).to(work_dtype)
    for axis in (4, 3, 2):
        u = _g_axis(u, axis)
    # [K,C,4,4,4] -> [64,K,C], q = temporal*16 + row*4 + column.
    u = u.permute(2, 3, 4, 0, 1).reshape(64, *weight.shape[:2]).contiguous()
    return u.to(dtype) if fast else u


def _reference_3d(x, u, bias, out, chunk):
    n, c = x.shape[:2]
    _, k, ot, oh, ow = out.shape
    nt, nh, nw = (ot+1)//2, (oh+1)//2, (ow+1)//2
    x = F.pad(x.to(u.dtype), (0, ow % 2, 0, oh % 2, 0, ot % 2))
    windows = x.unfold(2, 4, 2).unfold(3, 4, 2).unfold(4, 4, 2)
    for start in range(0, n*nt*nh*nw, chunk):
        p = torch.arange(start, min(start+chunk, n*nt*nh*nw), device=x.device)
        nn, tt = p // (nt*nh*nw), (p // (nh*nw)) % nt
        yy, xx = (p // nw) % nh, p % nw
        v = windows[nn, :, tt, yy, xx]  # [P,C,4,4,4]
        for axis in (4, 3, 2):
            v = _bt_axis(v, axis)
        v = v.permute(2, 3, 4, 1, 0).reshape(64, c, len(p)).contiguous()
        y = torch.bmm(u, v).reshape(4, 4, 4, k, len(p)).permute(3, 4, 0, 1, 2)
        for axis in (4, 3, 2):
            y = _at_axis(y, axis)
        if bias is not None:
            y = y + bias.to(u.dtype)[:, None, None, None, None]
        for t in range(2):
            for h in range(2):
                for w in range(2):
                    valid = (2*tt+t < ot) & (2*yy+h < oh) & (2*xx+w < ow)
                    out[nn[valid], :, 2*tt[valid]+t, 2*yy[valid]+h, 2*xx[valid]+w] = (
                        y[:, valid, t, h, w].T.to(out.dtype))


def full_winograd_conv3d(x, weight, bias=None, cache=None,
                         workspace_bytes=WORKSPACE_BYTES):
    """Return (output, transient cache); no spatial/native fallback is used."""
    if x.ndim != 5 or weight.ndim != 5 or tuple(weight.shape[2:]) != (3, 3, 3):
        raise ValueError('Full Winograd requires NCTHW input and K,C,3,3,3 weights.')
    if x.device != weight.device or (bias is not None and bias.device != x.device):
        raise ValueError('Input, weight and bias must be on the same device.')
    if x.shape[1] != weight.shape[1] or min(x.shape[2:]) < 3:
        raise ValueError('Invalid padded input shape for valid 3x3x3 convolution.')
    if bias is not None and tuple(bias.shape) != (weight.shape[0],):
        raise ValueError('Bias must contain one value per output channel.')
    if type(workspace_bytes) is not int or workspace_bytes <= 0:
        raise ValueError('workspace_bytes must be a positive integer.')
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, weight, bias) if t is not None):
        raise RuntimeError('winograd_3d currently supports inference only; use no_grad/inference_mode.')
    shape = (x.shape[0], weight.shape[0], x.shape[2]-2, x.shape[3]-2, x.shape[4]-2)
    if x.device.type == 'meta':
        return x.new_empty(shape), None
    if x.device.type not in ('cpu', 'cuda'):
        raise ValueError('winograd_3d supports CPU reference and CUDA execution only.')
    dtype = _effective_dtype(x, weight, bias)
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f'Unsupported Winograd dtype: {dtype}')
    fast = x.is_cuda and dtype in (torch.float16, torch.bfloat16)
    work_dtype = dtype if fast else (torch.float64 if dtype == torch.float64 else torch.float32)
    with torch.autocast(device_type=x.device.type, enabled=False):
        key = _cache_key(weight, dtype, fast)
        # The spatial and full transforms have different shapes and formulas.
        key = (*key, 'winograd_3d') if key is not None else None
        if key is not None and cache is not None and cache[0] == key:
            u = cache[1]
        else:
            u = _transform_weight_3d(weight, dtype, fast)
            cache = (key, u) if key is not None else None
        x = x.to(dtype)
        bias = bias.to(dtype) if bias is not None else None
        layout = torch.channels_last_3d if x.is_contiguous(memory_format=torch.channels_last_3d) else torch.contiguous_format
        out = torch.empty(shape, dtype=dtype, device=x.device, memory_format=layout)
        itemsize = torch.empty((), dtype=work_dtype).element_size()
        per_tile = 64 * (x.shape[1]*itemsize + shape[1]*(4 if fast else itemsize))
        chunk = max(1, workspace_bytes // per_tile)
        if fast:
            from .winograd_3d_triton import run_full_winograd
            if chunk >= 128:
                chunk = (chunk // 128) * 128
            with torch.cuda.device(x.device):
                run_full_winograd(x, u, bias, out, chunk)
        else:
            _reference_3d(x, u, bias, out, chunk)
    return out, cache
