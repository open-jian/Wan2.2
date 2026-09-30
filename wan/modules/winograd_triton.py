"""CUDA kernels for spatial Winograd: FP32 transforms, low-precision GEMM inputs.

GEMM accumulation, stored products and inverse transform are FP32. There is no
native-convolution fallback. Input buffers are bounded by a spatial tile chunk.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bt(a, b, c, d):
    return a-c, b+c, c-b, b-d


@triton.jit
def _input_transform(X, V, C: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
                     W: tl.constexpr, OT: tl.constexpr, TH: tl.constexpr,
                     TW: tl.constexpr, P: tl.constexpr, START,
                     S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                     S3: tl.constexpr, S4: tl.constexpr, BLOCK: tl.constexpr):
    z = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    r, p = z//P, z % P
    global_p = START+p
    nn = global_p//(OT*TH*TW)
    tt = (global_p//(TH*TW)) % OT + r//C
    yy, xx = 2*((global_p//TW) % TH), 2*(global_p % TW)
    base = nn*S0 + (r % C)*S1 + tt*S2 + yy*S3 + xx*S4
    mask = z < 3*C*P
    d00 = tl.load(X+base, mask & (yy<H) & (xx<W), 0).to(tl.float32)
    d01 = tl.load(X+base+S4, mask & (yy<H) & (xx+1<W), 0).to(tl.float32)
    d02 = tl.load(X+base+2*S4, mask & (yy<H) & (xx+2<W), 0).to(tl.float32)
    d03 = tl.load(X+base+3*S4, mask & (yy<H) & (xx+3<W), 0).to(tl.float32)
    d10 = tl.load(X+base+S3, mask & (yy+1<H) & (xx<W), 0).to(tl.float32)
    d11 = tl.load(X+base+S3+S4, mask & (yy+1<H) & (xx+1<W), 0).to(tl.float32)
    d12 = tl.load(X+base+S3+2*S4, mask & (yy+1<H) & (xx+2<W), 0).to(tl.float32)
    d13 = tl.load(X+base+S3+3*S4, mask & (yy+1<H) & (xx+3<W), 0).to(tl.float32)
    d20 = tl.load(X+base+2*S3, mask & (yy+2<H) & (xx<W), 0).to(tl.float32)
    d21 = tl.load(X+base+2*S3+S4, mask & (yy+2<H) & (xx+1<W), 0).to(tl.float32)
    d22 = tl.load(X+base+2*S3+2*S4, mask & (yy+2<H) & (xx+2<W), 0).to(tl.float32)
    d23 = tl.load(X+base+2*S3+3*S4, mask & (yy+2<H) & (xx+3<W), 0).to(tl.float32)
    d30 = tl.load(X+base+3*S3, mask & (yy+3<H) & (xx<W), 0).to(tl.float32)
    d31 = tl.load(X+base+3*S3+S4, mask & (yy+3<H) & (xx+1<W), 0).to(tl.float32)
    d32 = tl.load(X+base+3*S3+2*S4, mask & (yy+3<H) & (xx+2<W), 0).to(tl.float32)
    d33 = tl.load(X+base+3*S3+3*S4, mask & (yy+3<H) & (xx+3<W), 0).to(tl.float32)
    a00, a01, a02, a03 = _bt(d00, d01, d02, d03)
    a10, a11, a12, a13 = _bt(d10, d11, d12, d13)
    a20, a21, a22, a23 = _bt(d20, d21, d22, d23)
    a30, a31, a32, a33 = _bt(d30, d31, d32, d33)
    v00, v10, v20, v30 = _bt(a00, a10, a20, a30)
    v01, v11, v21, v31 = _bt(a01, a11, a21, a31)
    v02, v12, v22, v32 = _bt(a02, a12, a22, a32)
    v03, v13, v23, v33 = _bt(a03, a13, a23, a33)
    span = 3*C*P
    tl.store(V+z+0*span, v00, mask)
    tl.store(V+z+1*span, v01, mask)
    tl.store(V+z+2*span, v02, mask)
    tl.store(V+z+3*span, v03, mask)
    tl.store(V+z+4*span, v10, mask)
    tl.store(V+z+5*span, v11, mask)
    tl.store(V+z+6*span, v12, mask)
    tl.store(V+z+7*span, v13, mask)
    tl.store(V+z+8*span, v20, mask)
    tl.store(V+z+9*span, v21, mask)
    tl.store(V+z+10*span, v22, mask)
    tl.store(V+z+11*span, v23, mask)
    tl.store(V+z+12*span, v30, mask)
    tl.store(V+z+13*span, v31, mask)
    tl.store(V+z+14*span, v32, mask)
    tl.store(V+z+15*span, v33, mask)


@triton.jit
def _gemm(U, V, M, K: tl.constexpr, R: tl.constexpr, P: tl.constexpr,
          BK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    kk = tl.program_id(0)*BM + tl.arange(0, BM)
    pp = tl.program_id(1)*BN + tl.arange(0, BN)
    rr = tl.arange(0, BK)
    q = tl.program_id(2)
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(tl.cdiv(R, BK)):
        r = step*BK+rr
        a = tl.load(U+q*K*R+kk[:, None]*R+r[None, :],
                    (kk[:, None]<K) & (r[None, :]<R), 0)
        b = tl.load(V+q*R*P+r[:, None]*P+pp[None, :],
                    (r[:, None]<R) & (pp[None, :]<P), 0)
        acc = tl.dot(a, b, acc, input_precision='ieee')
    tl.store(M+q*K*P+kk[:, None]*P+pp[None, :], acc,
             (kk[:, None]<K) & (pp[None, :]<P))


@triton.jit
def _output_transform(M, BIAS, Y, K: tl.constexpr, P: tl.constexpr, START,
                      OT: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
                      TH: tl.constexpr, TW: tl.constexpr, HAS_BIAS: tl.constexpr,
                      S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                      S3: tl.constexpr, S4: tl.constexpr, BLOCK: tl.constexpr):
    z = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    k, p = z//P, z % P
    valid = z<K*P
    span = K*P
    m00 = tl.load(M+z+0*span, valid, 0)
    m01 = tl.load(M+z+1*span, valid, 0)
    m02 = tl.load(M+z+2*span, valid, 0)
    m03 = tl.load(M+z+3*span, valid, 0)
    m10 = tl.load(M+z+4*span, valid, 0)
    m11 = tl.load(M+z+5*span, valid, 0)
    m12 = tl.load(M+z+6*span, valid, 0)
    m13 = tl.load(M+z+7*span, valid, 0)
    m20 = tl.load(M+z+8*span, valid, 0)
    m21 = tl.load(M+z+9*span, valid, 0)
    m22 = tl.load(M+z+10*span, valid, 0)
    m23 = tl.load(M+z+11*span, valid, 0)
    m30 = tl.load(M+z+12*span, valid, 0)
    m31 = tl.load(M+z+13*span, valid, 0)
    m32 = tl.load(M+z+14*span, valid, 0)
    m33 = tl.load(M+z+15*span, valid, 0)
    a0, a1, a2, a3 = m00+m10+m20, m01+m11+m21, m02+m12+m22, m03+m13+m23
    b0, b1, b2, b3 = m10-m20-m30, m11-m21-m31, m12-m22-m32, m13-m23-m33
    bias = tl.full((BLOCK,), 0, tl.float32)
    if HAS_BIAS:
        bias = tl.load(BIAS+k, valid, 0).to(tl.float32)
    gp = START+p
    nn, tt = gp//(OT*TH*TW), (gp//(TH*TW)) % OT
    yy, xx = 2*((gp//TW) % TH), 2*(gp % TW)
    base = nn*S0+k*S1+tt*S2+yy*S3+xx*S4
    tl.store(Y+base, a0+a1+a2+bias, valid & (yy<OH) & (xx<OW))
    tl.store(Y+base+S4, a1-a2-a3+bias, valid & (yy<OH) & (xx+1<OW))
    tl.store(Y+base+S3, b0+b1+b2+bias, valid & (yy+1<OH) & (xx<OW))
    tl.store(Y+base+S3+S4, b1-b2-b3+bias, valid & (yy+1<OH) & (xx+1<OW))


def run_spatial_winograd(x, u, bias, out, chunk):
    n, c, t, h, w = x.shape
    _, k, ot, oh, ow = out.shape
    th, tw = (oh+1)//2, (ow+1)//2
    total = n*ot*th*tw
    for start in range(0, total, chunk):
        p = min(chunk, total-start)
        v = torch.empty((16, 3*c, p), device=x.device, dtype=x.dtype)
        m = torch.empty((16, k, p), device=x.device, dtype=torch.float32)
        _input_transform[(triton.cdiv(3*c*p, 256),)](
            x, v, c, t, h, w, ot, th, tw, p, start, *x.stride(), BLOCK=256)
        _gemm[(triton.cdiv(k, 64), triton.cdiv(p, 128), 16)](
            u, v, m, k, 3*c, p, BK=32, BM=64, BN=128, num_warps=4, num_stages=3)
        _output_transform[(triton.cdiv(k*p, 256),)](
            m, bias if bias is not None else out, out, k, p, start,
            ot, oh, ow, th, tw, bias is not None, *out.stride(), BLOCK=256)
