"""Winograd fusion with pretransformed weights, inspired by Neon's FX design.

The fused kernel holds every transform coordinate of its output tile locally,
accumulates over channels, performs the inverse transform and writes final
convolution outputs only. The optional output-only fusion keeps the input transform in a
separate kernel, but never materializes the GEMM result M in global memory.

This is a new Triton implementation, not a translation of Maxwell SASS. See
NervanaSystems/neon@8c3fb8a93b4a, winograd_conv.py and the 2x2_3x3_32x32 kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bt_axis(values, STRIDE: tl.constexpr):
    q = tl.arange(0, values.shape[0])
    row = (q // STRIDE) % 4
    base = q-row*STRIDE
    ia = base+tl.where(row == 0, 0, tl.where(row == 2, 2, 1))*STRIDE
    ib = base+tl.where(row < 2, 2, tl.where(row == 2, 1, 3))*STRIDE
    a = tl.gather(values, tl.broadcast_to(ia[:, None, None], values.shape), 0)
    b = tl.gather(values, tl.broadcast_to(ib[:, None, None], values.shape), 0)
    return tl.where((row == 1)[:, None, None], a+b, a-b)


@triton.jit
def _at_axis(values, STRIDE: tl.constexpr):
    r = tl.arange(0, values.shape[0] // 2)
    row = (r // STRIDE) % 2
    base = (r // (2*STRIDE))*(4*STRIDE) + r % STRIDE + row*STRIDE
    shape: tl.constexpr = (values.shape[0] // 2, values.shape[1], values.shape[2])
    a = tl.gather(values, tl.broadcast_to(base[:, None, None], shape), 0)
    b = tl.gather(values, tl.broadcast_to((base+STRIDE)[:, None, None], shape), 0)
    c = tl.gather(values, tl.broadcast_to((base+2*STRIDE)[:, None, None], shape), 0)
    return tl.where((row == 0)[:, None, None], (a+b)+c, (a-b)-c)


@triton.jit
def _fused(X, U, V, BIAS, Y,
           C: tl.constexpr, K: tl.constexpr, T: tl.constexpr,
           H: tl.constexpr, W: tl.constexpr,
           OT: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
           NT: tl.constexpr, NH: tl.constexpr, NW: tl.constexpr,
           P: tl.constexpr, START,
           X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr,
           X3: tl.constexpr, X4: tl.constexpr,
           Y0: tl.constexpr, Y1: tl.constexpr, Y2: tl.constexpr,
           Y3: tl.constexpr, Y4: tl.constexpr,
           FULL_3D: tl.constexpr, FUSE_INPUT: tl.constexpr,
           HAS_BIAS: tl.constexpr, BM: tl.constexpr,
           BN: tl.constexpr, BK: tl.constexpr):
    Q: tl.constexpr = 64 if FULL_3D else 16
    R: tl.constexpr = C if FULL_3D else 3*C
    kk = tl.program_id(0)*BM + tl.arange(0, BM)
    pp = tl.program_id(1)*BN + tl.arange(0, BN)
    rr = tl.arange(0, BK)
    qq = tl.arange(0, Q)
    gp = START+pp
    nn = gp // (NT*NH*NW)
    tt = (gp // (NH*NW)) % NT
    yy, xx = 2*((gp // NW) % NH), 2*(gp % NW)
    if FULL_3D:
        tt = 2*tt
    acc = tl.full((Q, BM, BN), 0, tl.float32)
    for step in range(tl.cdiv(R, BK)):
        r = step*BK+rr
        a = tl.load(U+qq[:, None, None]*K*R+kk[None, :, None]*R+r[None, None, :],
                    (kk[None, :, None]<K) & (r[None, None, :]<R), 0)
        if FUSE_INPUT:
            if FULL_3D:
                it = tt[None, None, :]+(qq//16)[:, None, None]
            else:
                it = tt[None, None, :]+(r//C)[None, :, None]
            ih = yy[None, None, :]+((qq//4) % 4)[:, None, None]
            iw = xx[None, None, :]+(qq % 4)[:, None, None]
            offset = (nn[None, None, :]*X0+(r % C)[None, :, None]*X1
                      +it*X2+ih*X3+iw*X4)
            d = tl.load(X+offset, (pp[None, None, :]<P) & (r[None, :, None]<R)
                        & (it<T) & (ih<H) & (iw<W), 0).to(tl.float32)
            d = _bt_axis(d, 1)
            d = _bt_axis(d, 4)
            if FULL_3D:
                d = _bt_axis(d, 16)
            b = d.to(X.dtype.element_ty)
        else:
            b = tl.load(V+qq[:, None, None]*R*P+r[None, :, None]*P+pp[None, None, :],
                        (r[None, :, None]<R) & (pp[None, None, :]<P), 0)
        acc = tl.dot(a, b, acc, input_precision='ieee')

    y = _at_axis(acc, 1)
    y = _at_axis(y, 2)
    if FULL_3D:
        y = _at_axis(y, 4)
    if HAS_BIAS:
        bias = tl.load(BIAS+kk, kk<K, 0).to(tl.float32)
        y = y+bias[None, :, None]
    oo = tl.arange(0, 8 if FULL_3D else 4)
    out_t = tt[None, None, :] + (oo//4)[:, None, None]
    out_h = yy[None, None, :] + ((oo//2) % 2)[:, None, None]
    out_w = xx[None, None, :] + (oo % 2)[:, None, None]
    offset = (nn[None, None, :]*Y0+kk[None, :, None]*Y1
              +out_t*Y2+out_h*Y3+out_w*Y4)
    tl.store(Y+offset, y, (kk[None, :, None]<K) & (pp[None, None, :]<P)
             & (out_t<OT) & (out_h<OH) & (out_w<OW))


def run_fused_winograd(x, u, bias, out, chunk, *, full_3d,
                        fuse_input=True, block_m=32, block_n=None,
                        block_k=None, num_warps=8):
    """Return launch metadata for diagnostics; allocate no global product M."""
    # Selected from the bounded sm_120 layer sweep. Still slower than the
    # separate kernels: these experimental settings are not general tuning.
    if block_n is None:
        block_n = 16 if full_3d else 32
    if block_k is None:
        block_k = 16 if full_3d else 32
    n, c, t, h, w = x.shape
    _, k, ot, oh, ow = out.shape
    nt, nh, nw = ((ot+1)//2 if full_3d else ot), (oh+1)//2, (ow+1)//2
    total = n*nt*nh*nw
    launches = []
    # A fully fused launch needs no V/M workspace or spatial chunk loop.
    step = total if fuse_input else chunk
    for start in range(0, total, step):
        p = min(step, total-start)
        if fuse_input:
            v = x  # unused pointer in this specialization
        elif full_3d:
            from .winograd_3d_triton import _input_transform_3d
            v = torch.empty((64, c, p), device=x.device, dtype=x.dtype)
            _input_transform_3d[(triton.cdiv(c*p, 128),)](
                x, v, c, t, h, w, nt, nh, nw, p, start, *x.stride(), BLOCK=128)
        else:
            from .winograd_triton import _input_transform
            v = torch.empty((16, 3*c, p), device=x.device, dtype=x.dtype)
            _input_transform[(triton.cdiv(3*c*p, 256),)](
                x, v, c, t, h, w, ot, nh, nw, p, start, *x.stride(), BLOCK=256)
        kernel = _fused[(triton.cdiv(k, block_m), triton.cdiv(p, block_n))](
            x, u, v, bias if bias is not None else out, out,
            c, k, t, h, w, ot, oh, ow, nt, nh, nw, p, start,
            *x.stride(), *out.stride(), full_3d, fuse_input, bias is not None,
            block_m, block_n, block_k, num_warps=num_warps, num_stages=1)
        launches.append(kernel)
    return launches
