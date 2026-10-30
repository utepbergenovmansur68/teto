"""Fused RMSNorm for Qwen2 (Triton, forward + backward), written from scratch.

Idea source: the standard RMSNorm definition (Zhang & Sennrich, 2019) and the
usual "one program per row" Triton layout; no code copied from other projects.

The reference Qwen2RMSNorm does: cast to fp32 -> rsqrt(mean(x^2)+eps) -> cast back
to the input dtype -> multiply by the (fp32) weight.  Here all of that is ONE
kernel and the backward recomputes from x and the saved per-row rstd, so no fp32
copies of the activations are kept alive.  Norm weights are frozen under LoRA,
so no weight gradient is computed; if a weight requires grad we fall back.
With out_fp16=True the kernel writes fp16 directly (the projections run in fp16
under autocast anyway), which removes the fp32->fp16 cast kernels after the norm.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_fwd(X, W, Y, R, stride_x, stride_y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    msk = cols < N
    x = tl.load(X + row.to(tl.int64) * stride_x + cols, mask=msk, other=0.0)
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / N
    r = tl.rsqrt(var + eps)
    # same rounding step as the reference: (x * r).to(input_dtype)
    xn = (xf * r).to(X.dtype.element_ty).to(tl.float32)
    w = tl.load(W + cols, mask=msk, other=0.0).to(tl.float32)
    tl.store(Y + row.to(tl.int64) * stride_y + cols,
             (xn * w).to(Y.dtype.element_ty), mask=msk)
    tl.store(R + row, r)


@triton.jit
def _rms_bwd(DY, X, W, R, DX, stride_dy, stride_x, stride_dx, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    msk = cols < N
    dy = tl.load(DY + row.to(tl.int64) * stride_dy + cols, mask=msk, other=0.0).to(tl.float32)
    x = tl.load(X + row.to(tl.int64) * stride_x + cols, mask=msk, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=msk, other=0.0).to(tl.float32)
    r = tl.load(R + row)
    xn = x * r
    dxn = dy * w
    c = tl.sum(dxn * xn, axis=0) / N
    dx = r * (dxn - xn * c)
    tl.store(DX + row.to(tl.int64) * stride_dx + cols,
             dx.to(DX.dtype.element_ty), mask=msk)


def _cfg(N):
    block = triton.next_power_of_2(N)
    warps = 4 if block <= 2048 else 8
    return block, warps


class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, out_fp16=False):
        shape = x.shape
        N = shape[-1]
        x2 = x.reshape(-1, N)
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        out_dtype = torch.float16 if out_fp16 else torch.promote_types(x.dtype, weight.dtype)
        y = torch.empty(x2.shape, dtype=out_dtype, device=x.device)
        r = torch.empty(x2.shape[0], dtype=torch.float32, device=x.device)
        block, warps = _cfg(N)
        _rms_fwd[(x2.shape[0],)](x2, weight, y, r, x2.stride(0), y.stride(0), N, eps,
                                 BLOCK=block, num_warps=warps)
        ctx.save_for_backward(x2, weight, r)
        ctx.shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, r = ctx.saved_tensors
        N = x2.shape[-1]
        dy2 = dy.reshape(-1, N)
        if dy2.stride(-1) != 1:
            dy2 = dy2.contiguous()
        dx = torch.empty_like(x2)
        block, warps = _cfg(N)
        _rms_bwd[(x2.shape[0],)](dy2, x2, weight, r, dx, dy2.stride(0), x2.stride(0),
                                 dx.stride(0), N, BLOCK=block, num_warps=warps)
        return dx.view(ctx.shape), None, None, None


_SUPPORTED = (torch.float16, torch.float32)


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2 as m

    cls = m.Qwen2RMSNorm
    if getattr(cls, "_soup_fused", False):
        return
    orig = cls.forward

    def forward(self, hidden_states):
        w = self.weight
        if (hidden_states.is_cuda and w.is_cuda and not w.requires_grad
                and hidden_states.dtype in _SUPPORTED and w.dtype in _SUPPORTED
                and w.is_contiguous()):
            eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
            out16 = (getattr(self, "_soup_fp16_out", False)
                     and hidden_states.dtype == torch.float32
                     and torch.is_autocast_enabled("cuda"))
            return RMSNormFn.apply(hidden_states, w, float(eps), out16)
        return orig(self, hidden_states)

    cls.forward = forward
    cls._soup_fused = True


def patch_fp16_norm_outputs():
    """Norms inside decoder layers hand fp16 to the projections (final norm stays fp32)."""
    from transformers.models.qwen2 import modeling_qwen2 as m
    if getattr(m.Qwen2Model, "_soup_fp16_flags", False):
        return
    orig = m.Qwen2Model.__init__

    def __init__(self, *a, **k):
        orig(self, *a, **k)
        for layer in self.layers:
            layer.input_layernorm._soup_fp16_out = True
            layer.post_attention_layernorm._soup_fp16_out = True

    m.Qwen2Model.__init__ = __init__
    m.Qwen2Model._soup_fp16_flags = True


def patch_peft_input_cast():
    """peft upcasts fp16 input to fp32 for LoRA; autocast then casts it back. Skip both."""
    from peft.tuners.tuners_utils import BaseTunerLayer
    orig = BaseTunerLayer._cast_input_dtype
    if getattr(orig, "_soup", False):
        return

    def _cast_input_dtype(self, x, dtype):
        if x.dtype == torch.float16 and torch.is_autocast_enabled("cuda"):
            return x
        return orig(self, x, dtype)

    _cast_input_dtype._soup = True
    BaseTunerLayer._cast_input_dtype = _cast_input_dtype
