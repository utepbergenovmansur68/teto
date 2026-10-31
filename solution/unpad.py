"""Skip pad rows in the Qwen2 MLP: gather real tokens, run the MLP, scatter back (Triton).

Idea source: the usual "unpadding" trick (as in FlashAttention's varlen path); own code.
"""
import threading

import torch
import triton
import triton.language as tl

STATE = threading.local()
_SEEN = [0]


@triton.jit
def _row_gather(X, IDX, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    src = tl.load(IDX + i)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    v = tl.load(X + src * N + cols, mask=m)
    tl.store(Y + i.to(tl.int64) * N + cols, v, mask=m)


@triton.jit
def _row_scatter(Y, IDX, X, N, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    dst = tl.load(IDX + i)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    v = tl.load(Y + i.to(tl.int64) * N + cols, mask=m)
    tl.store(X + dst * N + cols, v, mask=m)


def _gather(x2, idx):
    x2 = x2.contiguous()
    n = x2.shape[1]
    out = torch.empty(idx.numel(), n, dtype=x2.dtype, device=x2.device)
    _row_gather[(idx.numel(),)](x2, idx, out, n, BLOCK=triton.next_power_of_2(n), num_warps=4)
    return out


def _scatter(y2, idx, rows):
    y2 = y2.contiguous()
    n = y2.shape[1]
    out = torch.zeros(rows, n, dtype=y2.dtype, device=y2.device)
    _row_scatter[(idx.numel(),)](y2, idx, out, n, BLOCK=triton.next_power_of_2(n), num_warps=4)
    return out


class RowGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2, idx):
        ctx.idx, ctx.rows = idx, x2.shape[0]
        return _gather(x2, idx)

    @staticmethod
    def backward(ctx, g):
        return _scatter(g, ctx.idx, ctx.rows), None


class RowScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y2, idx, rows):
        ctx.idx = idx
        return _scatter(y2, idx, rows)

    @staticmethod
    def backward(ctx, g):
        return _gather(g, ctx.idx), None, None


def set_mask(mask):
    """Called by head_loss.compute_loss before the forward: remember which rows are real."""
    STATE.idx = mask.reshape(-1).nonzero().squeeze(1)
    STATE.total = mask.numel()



def clear():
    STATE.idx = None


def patch_mlp_unpad():
    from transformers.models.qwen2 import modeling_qwen2 as Q
    cls = Q.Qwen2MLP
    if getattr(cls, "_soup_unpad", False):
        return
    prev = cls.forward  # whatever is installed now (the SwiGLU example)

    def forward(self, x):
        idx = getattr(STATE, "idx", None)
        if (idx is None or not x.is_cuda or x.dim() != 3
                or x.shape[0] * x.shape[1] != STATE.total or idx.numel() == STATE.total
                or not torch.is_grad_enabled()):
            return prev(self, x)
        B, T, H = x.shape
        xs = RowGather.apply(x.reshape(-1, H), idx)
        y = prev(self, xs.unsqueeze(0))
        y = RowScatter.apply(y.reshape(-1, y.shape[-1]), idx, B * T)
        return y.view(B, T, -1)

    cls.forward = forward
    cls._soup_unpad = True
