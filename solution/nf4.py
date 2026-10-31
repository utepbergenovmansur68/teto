"""NF4 de-quantisation kernel in Triton (written from scratch).

bitsandbytes unpacks a 4-bit NF4 weight to a dense fp16 matrix before every matmul
(forward and backward, ~5% of a training step on this task).  This file does the same
unpacking with our own kernel: each byte holds two 4-bit codes (high nibble first),
every 64-element block has one fp32 scale ("absmax"; with double quantisation the
scales are first restored by bitsandbytes' own small routine), value = code[idx] * absmax.

Idea source: the NF4 format of QLoRA (Dettmers et al., 2023) and bitsandbytes' public
documentation of its layout.  The kernel itself is our own code.
Anything unusual (other quant type, transposed weight, out= given, ...) falls back to
the original bitsandbytes function.
"""
import torch
import triton
import triton.language as tl

_CODE = {}
_BROKEN = False


@triton.jit
def _nf4_deq(A, ABS, CODE, OUT, n_bytes, blocksize, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = idx < n_bytes
    b = tl.load(A + idx, mask=m, other=0).to(tl.int32)
    hi = b >> 4
    lo = b & 15
    scale = tl.load(ABS + (idx * 2) // blocksize, mask=m, other=0.0)
    c_hi = tl.load(CODE + hi)
    c_lo = tl.load(CODE + lo)
    tl.store(OUT + idx * 2, (c_hi * scale).to(OUT.dtype.element_ty), mask=m)
    tl.store(OUT + idx * 2 + 1, (c_lo * scale).to(OUT.dtype.element_ty), mask=m)

@triton.jit
def _nf4_deq2(A, ABS8, CODE2, ABS2, OFF, CODE, OUT, n_bytes, blocksize, bs2,
              BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)   # индекс байта
    m = idx < n_bytes
    b = tl.load(A + idx, mask=m, other=0).to(tl.int32)
    blk = (idx * 2) // blocksize
    q = tl.load(ABS8 + blk, mask=m, other=0).to(tl.int32)
    s2 = tl.load(ABS2 + blk // bs2, mask=m, other=0.0)
    scale = tl.load(CODE2 + q) * s2 + tl.load(OFF)
    j = tl.arange(0, 2)
    nib = (b[:, None] >> ((1 - j[None, :]) * 4)) & 15      # старший полубайт первым
    code = tl.load(CODE + nib)
    val = code * scale[:, None]
    o = idx[:, None] * 2 + j[None, :]
    tl.store(OUT + o, val.to(OUT.dtype.element_ty), mask=m[:, None])
                  
def _code(device):
    key = str(device)
    if key not in _CODE:
        import bitsandbytes.functional as BF
        _CODE[key] = BF.get_4bit_type("nf4", device=device).to(torch.float32).contiguous()
    return _CODE[key]


def dequantize_nf4(A, st):
    flat = A.reshape(-1)
    out = torch.empty(st.shape, dtype=st.dtype, device=A.device)
    n_bytes = flat.numel()
    block = 1024
    grid = (triton.cdiv(n_bytes, block),)
    if st.nested:
        s2 = st.state2
        offset = torch.as_tensor(st.offset, dtype=torch.float32, device=A.device).reshape(1)
        _nf4_deq2[grid](flat, st.absmax.contiguous(), s2.code.float().contiguous(),
                        s2.absmax.float().contiguous(), offset, _code(A.device), out,
                        n_bytes, st.blocksize, s2.blocksize, BLOCK=block, num_warps=4)
    else:
        absmax = st.absmax.float().contiguous()
        _nf4_deq[grid](flat, absmax, _code(A.device), out, n_bytes,
                       st.blocksize, BLOCK=block, num_warps=4)
    return out


def patch_bnb_dequant():
    import bitsandbytes.functional as BF

    orig = BF.dequantize_4bit
    if getattr(orig, "_soup_fused", False):
        return

    def dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=64,
                        quant_type="fp4"):
        global _BROKEN
        st = quant_state
        if (not _BROKEN and st is not None and absmax is None and out is None
                and A.is_cuda and A.dtype == torch.uint8
                and getattr(st, "quant_type", None) == "nf4"
                and st.dtype in (torch.float16, torch.float32)
                and st.blocksize % 2 == 0 and st.shape.numel() == 2 * A.numel()):
            try:
                res = dequantize_nf4(A, st)
                return res.t() if A.shape[0] == 1 else res
            except Exception:  # never break training because of the speed-up
                _BROKEN = True
        return orig(A, quant_state, absmax, out, blocksize, quant_type)

    dequantize_4bit._soup_fused = True
    BF.dequantize_4bit = dequantize_4bit
    import sys
    for name, mod in list(sys.modules.items()):
        if name.startswith("bitsandbytes") and getattr(mod, "dequantize_4bit", None) is orig:
            mod.dequantize_4bit = dequantize_4bit
