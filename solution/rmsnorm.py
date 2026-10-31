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

@triton.jit
def _add_rms_fwd(R, A, W, S, Y, RS, stride_r, stride_a, stride_s, stride_y, N, eps,
                 BLOCK: tl.constexpr):
    row = tl.program_id(0)
    ro = row.to(tl.int64)
    cols = tl.arange(0, BLOCK)
    msk = cols < N
    r = tl.load(R + ro * stride_r + cols, mask=msk, other=0.0).to(tl.float32)
    a = tl.load(A + ro * stride_a + cols, mask=msk, other=0.0).to(tl.float32)
    s = (r + a).to(S.dtype.element_ty)
    tl.store(S + ro * stride_s + cols, s, mask=msk)
    sf = s.to(tl.float32)  # exactly what the reference norm would read
    rr = tl.rsqrt(tl.sum(sf * sf, axis=0) / N + eps)
    xn = (sf * rr).to(S.dtype.element_ty).to(tl.float32)
    w = tl.load(W + cols, mask=msk, other=0.0).to(tl.float32)
    tl.store(Y + ro * stride_y + cols, (xn * w).to(Y.dtype.element_ty), mask=msk)
    tl.store(RS + row, rr)


@triton.jit
def _add_rms_bwd(DY, DS, S, W, RS, DR, DA, stride_dy, stride_ds, stride_s, stride_dr,
                 stride_da, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    ro = row.to(tl.int64)
    cols = tl.arange(0, BLOCK)
    msk = cols < N
    dy = tl.load(DY + ro * stride_dy + cols, mask=msk, other=0.0).to(tl.float32)
    ds = tl.load(DS + ro * stride_ds + cols, mask=msk, other=0.0).to(tl.float32)
    s = tl.load(S + ro * stride_s + cols, mask=msk, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=msk, other=0.0).to(tl.float32)
    rr = tl.load(RS + row)
    xn = s * rr
    dxn = dy * w
    c = tl.sum(dxn * xn, axis=0) / N
    dx = rr * (dxn - xn * c) + ds
    tl.store(DR + ro * stride_dr + cols, dx.to(DR.dtype.element_ty), mask=msk)
    tl.store(DA + ro * stride_da + cols, dx.to(DA.dtype.element_ty), mask=msk)


class AddRMSNormFn(torch.autograd.Function):
    """(res, attn) -> (res + attn, rmsnorm(res + attn)) in one kernel."""

    @staticmethod
    def forward(ctx, res, attn, weight, eps, out_fp16=False):
        shape = res.shape
        N = shape[-1]
        r2 = res.reshape(-1, N)
        a2 = attn.reshape(-1, N)
        if r2.stride(-1) != 1:
            r2 = r2.contiguous()
        if a2.stride(-1) != 1:
            a2 = a2.contiguous()
        s_dtype = torch.promote_types(res.dtype, attn.dtype)
        y_dtype = torch.float16 if out_fp16 else torch.promote_types(s_dtype, weight.dtype)
        s = torch.empty(r2.shape, dtype=s_dtype, device=res.device)
        y = torch.empty(r2.shape, dtype=y_dtype, device=res.device)
        rs = torch.empty(r2.shape[0], dtype=torch.float32, device=res.device)
        block, warps = _cfg(N)
        _add_rms_fwd[(r2.shape[0],)](r2, a2, weight, s, y, rs, r2.stride(0), a2.stride(0),
                                     s.stride(0), y.stride(0), N, eps,
                                     BLOCK=block, num_warps=warps)
        ctx.save_for_backward(s, weight, rs)
        ctx.shape, ctx.res_dtype, ctx.attn_dtype = shape, res.dtype, attn.dtype
        return s.view(shape), y.view(shape)

    @staticmethod
    def backward(ctx, ds, dy):
        s, weight, rs = ctx.saved_tensors
        N = s.shape[-1]
        ds2 = ds.reshape(-1, N)
        dy2 = dy.reshape(-1, N)
        if ds2.stride(-1) != 1:
            ds2 = ds2.contiguous()
        if dy2.stride(-1) != 1:
            dy2 = dy2.contiguous()
        dr = torch.empty(s.shape, dtype=ctx.res_dtype, device=s.device)
        da = torch.empty(s.shape, dtype=ctx.attn_dtype, device=s.device)
        block, warps = _cfg(N)
        _add_rms_bwd[(s.shape[0],)](dy2, ds2, s, weight, rs, dr, da, dy2.stride(0),
                                    ds2.stride(0), s.stride(0), dr.stride(0), da.stride(0),
                                    N, BLOCK=block, num_warps=warps)
        return dr.view(ctx.shape), da.view(ctx.shape), None, None, None


def patch_decoder_add_norm():
    """Qwen2DecoderLayer: fuse `residual + attn_out` with post_attention_layernorm."""
    import inspect
    from transformers.models.qwen2 import modeling_qwen2 as m

    cls = m.Qwen2DecoderLayer
    if getattr(cls, "_soup_addnorm", False):
        return
    orig = cls.forward
    src = inspect.getsource(orig)
    expected = ("hidden_states, _ = self.self_attn(", "self.post_attention_layernorm(",
                "past_key_values=past_key_values", "position_embeddings=position_embeddings",
                "return hidden_states")
    if src.count("residual + hidden_states") != 2 or not all(e in src for e in expected):
        raise RuntimeError("unexpected Qwen2DecoderLayer.forward layout, not patched")

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, cache_position=None,
                position_embeddings=None, **kwargs):
        norm = self.post_attention_layernorm
        w = norm.weight
        if not (hidden_states.is_cuda and w.is_cuda and not w.requires_grad
                and w.is_contiguous() and w.dtype in _SUPPORTED
                and hidden_states.dtype in _SUPPORTED):
            return orig(self, hidden_states, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=past_key_values,
                        use_cache=use_cache, cache_position=cache_position,
                        position_embeddings=position_embeddings, **kwargs)
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        attn_out, _ = self.self_attn(
            hidden_states=h, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position,
            position_embeddings=position_embeddings, **kwargs)
        if attn_out.shape != residual.shape or attn_out.dtype not in _SUPPORTED:
            hidden_states = residual + attn_out
            return hidden_states + self.mlp(norm(hidden_states))
        s_dtype = torch.promote_types(residual.dtype, attn_out.dtype)
        out16 = (getattr(norm, "_soup_fp16_out", False) and s_dtype == torch.float32
                 and torch.is_autocast_enabled("cuda"))
        eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
        hidden_states, normed = AddRMSNormFn.apply(residual, attn_out, w, eps, out16)
        return hidden_states + self.mlp(normed)

    cls.forward = forward
    cls._soup_addnorm = True
