"""Glue: call apply_extra() at the END of solution.apply() (after the example's patch)."""
import warnings

_DONE = False


def apply_extra():
    global _DONE
    if _DONE:
        return
    try:
        import torch
        if not torch.cuda.is_available():  # Soup's unit tests may run without a GPU
            return
        from . import fused_ce, head_loss, nf4, rmsnorm
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"solution extras not loaded: {e}")
        return
    for fn in (rmsnorm.patch_qwen2_rmsnorm, rmsnorm.patch_fp16_norm_outputs,
               rmsnorm.patch_peft_input_cast, fused_ce.patch_loss, fused_ce.patch_entropy,
               nf4.patch_bnb_dequant, head_loss.patch_head_loss):
        try:
            fn()
            print(f"[solution] OK: {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"[solution] SKIPPED {fn.__name__}: {e}")
    _DONE = True
