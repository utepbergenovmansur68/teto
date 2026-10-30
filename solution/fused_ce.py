"""Triton cross-entropy over the big vocabulary + fused entropy (written from scratch).

* `cross_entropy_rows`: single-pass online-softmax loss per row; the backward
  writes (softmax - onehot) * grad in one pass.  Replaces transformers'
  ForCausalLMLoss (float() copy + log_softmax + nll + their backward copies).
* `fused_entropy`: per-row entropy of the logits in one pass, replacing TRL's
  `entropy_from_logits` (which TRL runs every step from the full logits).
  Same values (fp32 math), fewer passes and no temporary vocab-sized tensors.

Idea source: online softmax (Milakov & Gimelshein, 2018). No code copied.
"""
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_BLOCK = 4096


@triton.jit
def _ce_fwd(LOGITS, LABELS, LSE, LOSS, stride, V, IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    label = tl.load(LABELS + row)
    base = LOGITS + row.to(tl.int64) * stride
    m = tl.max(tl.full([BLOCK], float("-inf"), tl.float32), axis=0)
    s = tl.sum(tl.zeros([BLOCK], tl.float32), axis=0)
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(base + cols, mask=cols < V, other=float("-inf")).to(tl.float32)
        nm = tl.maximum(m, tl.max(x, axis=0))
        s = s * tl.exp(m - nm) + tl.sum(tl.exp(x - nm), axis=0)
        m = nm
    lse = m + tl.log(s)
    ignored = label == IGNORE
    safe = tl.where(ignored, 0, label)
    tgt = tl.load(base + safe).to(tl.float32)
    tl.store(LSE + row, lse)
    tl.store(LOSS + row, tl.where(ignored, 0.0, lse - tgt))


@triton.jit
def _ce_bwd(LOGITS, LABELS, LSE, GOUT, DX, stride, stride_dx, V,
            IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    label = tl.load(LABELS + row)
    lse = tl.load(LSE + row)
    g = tl.load(GOUT + row).to(tl.float32)
    g = tl.where(label == IGNORE, 0.0, g)
    base = LOGITS + row.to(tl.int64) * stride
    dbase = DX + row.to(tl.int64) * stride_dx
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        msk = cols < V
        x = tl.load(base + cols, mask=msk, other=float("-inf")).to(tl.float32)
        p = tl.exp(x - lse)
        p = tl.where(cols == label, p - 1.0, p) * g
        tl.store(dbase + cols, p.to(DX.dtype.element_ty), mask=msk)


@triton.jit
def _entropy_fwd(X, OUT, stride, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = X + row.to(tl.int64) * stride
    m = tl.max(tl.full([BLOCK], float("-inf"), tl.float32), axis=0)
    s = tl.sum(tl.zeros([BLOCK], tl.float32), axis=0)
    t = tl.sum(tl.zeros([BLOCK], tl.float32), axis=0)
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        msk = cols < V
        x = tl.load(base + cols, mask=msk, other=float("-inf")).to(tl.float32)
        nm = tl.maximum(m, tl.max(x, axis=0))
        a = tl.exp(m - nm)
        e = tl.exp(x - nm)
        xs = tl.where(msk, x, 0.0)
        s = s * a + tl.sum(e, axis=0)
        t = t * a + tl.sum(e * xs, axis=0)
        m = nm
    tl.store(OUT + row, m + tl.log(s) - t / s)


class _CERows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, ignore_index):
        rows, V = logits.shape
        lse = torch.empty(rows, dtype=torch.float32, device=logits.device)
        loss = torch.empty(rows, dtype=torch.float32, device=logits.device)
        _ce_fwd[(rows,)](logits, labels, lse, loss, logits.stride(0), V,
                         IGNORE=ignore_index, BLOCK=_BLOCK, num_warps=8)
        ctx.save_for_backward(logits, labels, lse)
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    def backward(ctx, gout):
        logits, labels, lse = ctx.saved_tensors
        rows, V = logits.shape
        gout = gout.to(torch.float32).contiguous()
        dx = torch.empty_like(logits)
        _ce_bwd[(rows,)](logits, labels, lse, gout, dx, logits.stride(0), dx.stride(0), V,
                         IGNORE=ctx.ignore_index, BLOCK=_BLOCK, num_warps=8)
        return dx, None, None


def cross_entropy_rows(logits2d, labels1d, ignore_index=-100):
    if logits2d.stride(-1) != 1:
        logits2d = logits2d.contiguous()
    return _CERows.apply(logits2d, labels1d.contiguous(), ignore_index)


def fused_entropy(logits):
    V = logits.shape[-1]
    x = logits.reshape(-1, V)
    if x.stride(-1) != 1:
        x = x.contiguous()
    out = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    _entropy_fwd[(x.shape[0],)](x, out, x.stride(0), V, BLOCK=_BLOCK, num_warps=8)
    return out.view(logits.shape[:-1])


# ----------------------------------------------------------------- patches
def patch_loss():
    """Replace transformers' ForCausalLMLoss with the fused version."""
    import transformers.loss.loss_utils as lu

    orig = lu.ForCausalLMLoss
    if getattr(orig, "_soup_fused", False):
        return

    def fused_loss(logits, labels, vocab_size=None, num_items_in_batch=None,
                   ignore_index=-100, shift_labels=None, **kwargs):
        if (not logits.is_cuda) or logits.dim() != 3 or kwargs.get("reduction", None) is not None:
            return orig(logits, labels, vocab_size, num_items_in_batch=num_items_in_batch,
                        ignore_index=ignore_index, shift_labels=shift_labels, **kwargs)
        V = logits.shape[-1]
        if shift_labels is None:
            labels = F.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:]
        shift_labels = shift_labels.to(logits.device).reshape(-1)
        rows = cross_entropy_rows(logits.reshape(-1, V), shift_labels, ignore_index)
        total = rows.sum()
        if num_items_in_batch is not None:
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(total.device)
            return total / num_items_in_batch
        n = (shift_labels != ignore_index).sum().clamp(min=1)
        return total / n

    fused_loss._soup_fused = True
    lu.ForCausalLMLoss = fused_loss
    mapping = getattr(lu, "LOSS_MAPPING", None)
    if isinstance(mapping, dict):
        for k, v in list(mapping.items()):
            if v is orig:
                mapping[k] = fused_loss
        mapping["ForCausalLM"] = fused_loss


def patch_entropy():
    """Replace TRL's entropy_from_logits everywhere it is already imported."""
    import trl.trainer.utils as tu

    orig = getattr(tu, "entropy_from_logits", None)
    if orig is None or getattr(orig, "_soup_fused", False):
        return

    def entropy_from_logits(logits, *args, **kwargs):
        if logits.is_cuda and logits.dim() >= 2 and (not torch.is_grad_enabled() or not logits.requires_grad):
            try:
                return fused_entropy(logits)
            except Exception:  # never break training because of the speed-up
                pass
        return orig(logits, *args, **kwargs)

    entropy_from_logits._soup_fused = True
    tu.entropy_from_logits = entropy_from_logits
    for name, mod in list(sys.modules.items()):
        if name.startswith("trl") and getattr(mod, "entropy_from_logits", None) is orig:
            mod.entropy_from_logits = entropy_from_logits
