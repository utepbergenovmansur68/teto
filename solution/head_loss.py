"""Chunked LM-head + cross-entropy + entropy + argmax in one Triton kernel (written from scratch).

The logits [rows, 151936] are never kept: rows go through the head 512 at a time, ONE kernel reads
a chunk of logits and produces per-row loss, entropy, argmax and (in a second pass over the same
row) the gradient (softmax - onehot) in place; then dh = dlogits @ W. Pad rows are dropped first.

The frozen tied embedding / lm_head is stored in fp16 (checked to be lossless at start-up);
the embedding output is cast back to fp32, so values stay the same.

Idea source: online softmax (Milakov & Gimelshein, 2018); the idea of chunking the head+loss is
the general "fused linear cross-entropy" idea. No code copied from other projects.
"""
import threading
import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_BLOCK = 8192
_SCALE = 1024.0  # keeps small fp16 gradients from underflowing; divided out in fp32
_STATE = threading.local()

ENABLE_EMB16 = True  # False = keep the embedding in fp32 (rollback switch)
_EMB_DONE = [False]


def _shrink_embedding(model):
    """Frozen tied embedding/lm_head: store fp16 (values identical), embedding output stays fp32."""
    if _EMB_DONE[0]:
        return
    _EMB_DONE[0] = True
    try:
        emb = model.get_input_embeddings()
        w = emb.weight
        if w.dtype != torch.float32 or w.requires_grad or not w.is_cuda:
            print("[solution] emb16: skipped (unexpected dtype/grad)")
            return
        before = torch.cuda.memory_allocated() >> 20
        w16 = w.data.to(torch.float16)
        for i in range(0, w.shape[0], 16384):  # chunked: no big fp32 temporary
            if not torch.equal(w16[i:i + 16384].float(), w.data[i:i + 16384]):
                print("[solution] emb16: not lossless, skipped")
                return
        w.data = w16
        orig_forward = emb.forward

        def forward(input_ids, *a, **k):
            return orig_forward(input_ids, *a, **k).float()

        emb.forward = forward
        del w16
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated() >> 20
        print(f"[solution] emb16: embedding+lm_head now fp16, alloc {before} -> {after} MiB")
    except Exception as exc:  # noqa: BLE001
        print(f"[solution] emb16: skipped ({exc!r})")


@triton.jit
def _head_ce(L, T, LOSS, ENT, ARG, stride, V, GS, IGN: tl.constexpr, BLOCK: tl.constexpr,
             WRITE_GRAD: tl.constexpr):
    row = tl.program_id(0)
    base = L + row.to(tl.int64) * stride
    label = tl.load(T + row)
    m = tl.max(tl.full([BLOCK], float("-inf"), tl.float32), axis=0)
    best = tl.max(tl.full([BLOCK], float("-inf"), tl.float32), axis=0)
    s = tl.sum(tl.zeros([BLOCK], tl.float32), axis=0)
    t = tl.sum(tl.zeros([BLOCK], tl.float32), axis=0)
    besti = tl.sum(tl.zeros([BLOCK], tl.int32), axis=0)
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        msk = cols < V
        x = tl.load(base + cols, mask=msk, other=float("-inf")).to(tl.float32)
        bm = tl.max(x, axis=0)
        bi = tl.argmax(x, axis=0)
        besti = tl.where(bm > best, off + bi, besti)
        best = tl.maximum(best, bm)
        nm = tl.maximum(m, bm)
        a = tl.exp(m - nm)
        e = tl.exp(x - nm)
        xs = tl.where(msk, x, 0.0)
        s = s * a + tl.sum(e, axis=0)
        t = t * a + tl.sum(e * xs, axis=0)
        m = nm
    lse = m + tl.log(s)
    ignored = label == IGN
    safe = tl.where(ignored, 0, label)
    tgt = tl.load(base + safe).to(tl.float32)
    tl.store(LOSS + row, tl.where(ignored, 0.0, lse - tgt))
    tl.store(ENT + row, lse - t / s)
    tl.store(ARG + row, besti)
    if WRITE_GRAD:
        for off in range(0, V, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            msk = cols < V
            x = tl.load(base + cols, mask=msk, other=float("-inf")).to(tl.float32)
            p = tl.exp(x - lse)
            p = tl.where(cols == label, p - 1.0, p)
            p = tl.where(ignored, 0.0, p) * GS
            tl.store(base + cols, p.to(L.dtype.element_ty), mask=msk)


class _HeadCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, tgt, sink, denom, chunk):
        N, H = h.shape
        V = w.shape[0]
        need = ctx.needs_input_grad[0]
        w16 = w.to(torch.float16)
        h16 = h.to(torch.float16)
        rows = torch.empty(N, dtype=torch.float32, device=h.device)
        ent = torch.empty(N, dtype=torch.float32, device=h.device)
        arg = torch.empty(N, dtype=torch.int32, device=h.device)
        dh = torch.empty_like(h) if need else None
        gs = _SCALE / denom
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            logits = h16[s:e] @ w16.t()
            _head_ce[(e - s,)](logits, tgt[s:e], rows[s:e], ent[s:e], arg[s:e], logits.stride(0),
                               V, gs, IGN=-100, BLOCK=_BLOCK, WRITE_GRAD=need, num_warps=4)
            if need:
                dh[s:e] = (logits @ w16).to(h.dtype) * (1.0 / _SCALE)
            del logits
        labelled = tgt != -100
        sink["rows"], sink["ent"] = rows, ent
        sink["correct"] = ((arg.long() == tgt) & labelled).sum()
        sink["n_lab"] = labelled.sum()
        sink["arg"] = arg
        ctx.dh = dh
        return rows.sum() / denom

    @staticmethod
    def backward(ctx, gout):
        return ctx.dh * gout.to(ctx.dh.dtype), None, None, None, None, None


def fused_head_ce(h, w, tgt, denom, chunk=512):
    """h [N,H], w [V,H] (frozen), tgt [N] (-100 = no loss). Returns (loss, stats)."""
    sink = {}
    loss = _HeadCE.apply(h, w, tgt, sink, float(denom), chunk)
    return loss, sink


def _head_loss_3d(hidden, weight, targets, valid, mask, denom):
    B, T, H = hidden.shape
    idx = valid.reshape(-1).nonzero().squeeze(1)
    h = hidden.reshape(-1, H).index_select(0, idx)
    tgt = targets.reshape(-1).index_select(0, idx)
    att = mask.reshape(-1).index_select(0, idx).to(torch.float32)
    loss, s = fused_head_ce(h, weight, tgt, denom)
    tl_full = torch.zeros(B * T, dtype=torch.float32, device=hidden.device)
    tl_full = tl_full.index_copy(0, idx, s["rows"]).view(B, T)
    s.update(loss=loss, w=att, token_loss=tl_full)
    return s


def patch_head_loss():
    from transformers.models.qwen2 import modeling_qwen2 as Q
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from trl.trainer.sft_trainer import SFTTrainer
    from soup_cli.utils.stream_loss_chunk import STEP_ATTR

    if getattr(SFTTrainer.compute_loss, "_soup_head", False):
        return
    cls = Q.Qwen2ForCausalLM
    orig_fwd = cls.forward
    orig_loss = SFTTrainer.compute_loss

    def fwd(self, *args, **kwargs):
        ctx = getattr(_STATE, "ctx", None)
        w = self.lm_head.weight
        if (ctx is None or args or kwargs.get("labels") is not None
                or kwargs.get("logits_to_keep", 0) not in (0, None)
                or not self.training or w.requires_grad or not w.is_cuda):
            return orig_fwd(self, *args, **kwargs)
        kwargs.pop("labels", None)
        kwargs.pop("logits_to_keep", None)
        out = self.model(**kwargs)
        res = _head_loss_3d(out.last_hidden_state, w, ctx["targets"], ctx["valid"],
                            ctx["mask"], ctx["denom"])
        ctx["result"] = res
        return CausalLMOutputWithPast(loss=res["loss"], logits=None)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        a = self.args
        ok = (not getattr(self, "_soup_head_off", False)
              and isinstance(model, torch.nn.Module) and model.training
              and isinstance(inputs, dict) and "labels" in inputs and "attention_mask" in inputs
              and "shift_labels" not in inputs and inputs["labels"].is_cuda
              and not getattr(a, "use_liger_kernel", False)
              and getattr(self, "num_virtual_tokens", 0) == 0
              and not getattr(self, "aux_loss_enabled", False)
              and getattr(self, "compute_loss_func", None) is None
              and getattr(self, "label_smoother", None) is None
              and getattr(a, "loss_type", "nll") == "nll")
        if not ok:
            return orig_loss(self, model, inputs, return_outputs=return_outputs,
                             num_items_in_batch=num_items_in_batch)
        labels, mask = inputs["labels"], inputs["attention_mask"]
        targets = F.pad(labels, (0, 1), value=-100)[..., 1:]
        valid = mask.bool() | (targets != -100)
        if num_items_in_batch is None:
            denom = float((targets != -100).sum().clamp(min=1))
        else:
            denom = float(num_items_in_batch)
        fwd_inputs = {k: v for k, v in inputs.items()
                      if k not in ("labels", "_prediction_loss_only")}
        fwd_inputs["use_cache"] = False
        if ENABLE_EMB16:
            _shrink_embedding(model)

        from . import unpad
        unpad.set_mask(mask)

        _STATE.ctx = {"targets": targets, "valid": valid, "mask": mask, "denom": denom}
        torch.cuda.empty_cache()
        try:
            outputs = model(**fwd_inputs)
        finally:
            unpad.clear()
            c = _STATE.ctx
            _STATE.ctx = None

        res = c.get("result")
        if res is None:
            print("[solution] head_loss: fast path not used, falling back")
            self._soup_head_off = True
            return orig_loss(self, model, inputs, return_outputs=return_outputs,
                             num_items_in_batch=num_items_in_batch)
        if not getattr(self, "_soup_head_said", False):
            print("[solution] head_loss: chunked fast path is active")
            self._soup_head_said = True
        with torch.no_grad():
            ent = (res["ent"] * res["w"]).sum() / mask.sum()
            entropy = self.accelerator.gather_for_metrics(ent).mean().item()
            self._metrics["train"]["entropy"].append(entropy)
            n_tok = self.accelerator.gather_for_metrics(mask.sum()).sum().item()
            self._total_train_tokens += n_tok
            self._metrics["train"]["num_tokens"] = [self._total_train_tokens]
            correct = self.accelerator.gather_for_metrics(res["correct"])
            total = self.accelerator.gather_for_metrics(res["n_lab"])
            tsum = total.sum()
            acc = (correct.sum() / tsum).item() if tsum > 0 else 0.0
            self._metrics["train"]["mean_token_accuracy"].append(acc)
        setattr(self, STEP_ATTR, types.SimpleNamespace(
            token_loss=res["token_loss"], token_entropy=None,
            correct=res["correct"], labelled=res["n_lab"]))
        loss = res["loss"]
        return (loss, outputs) if return_outputs else loss

    compute_loss._soup_head = True
    SFTTrainer.compute_loss = compute_loss
    cls.forward = fwd
