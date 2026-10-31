import pytest
import torch
import torch.nn.functional as F

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def ref_rmsnorm(x, w, eps):
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dt)


@cuda
@pytest.mark.parametrize("xdt", [torch.float16, torch.float32])
def test_rmsnorm(xdt):
    from solution.rmsnorm import RMSNormFn
    torch.manual_seed(0)
    x = (torch.randn(3, 37, 1536, device="cuda") * 2).to(xdt).requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    w = torch.rand(1536, device="cuda") + 0.5
    dy = torch.randn(3, 37, 1536, device="cuda")
    y = RMSNormFn.apply(x, w, 1e-6, False)
    y_ref = ref_rmsnorm(x_ref, w, 1e-6)
    assert y.dtype == y_ref.dtype
    torch.testing.assert_close(y, y_ref, rtol=2e-3, atol=2e-3)
    y.backward(dy)
    y_ref.backward(dy)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=3e-2, atol=3e-2)


@cuda
@pytest.mark.parametrize("ldt", [torch.float32, torch.float16])
def test_ce_and_entropy(ldt):
    from solution.fused_ce import cross_entropy_rows, fused_entropy
    torch.manual_seed(0)
    V = 151936
    logits = (torch.randn(2 * 19, V, device="cuda") * 3).to(ldt).requires_grad_()
    ref = logits.detach().clone().requires_grad_()
    labels = torch.randint(0, V, (2 * 19,), device="cuda")
    labels[::5] = -100
    rows = cross_entropy_rows(logits, labels)
    ref_rows = F.cross_entropy(ref.float(), labels, ignore_index=-100, reduction="none")
    torch.testing.assert_close(rows, ref_rows, rtol=1e-4, atol=1e-4)
    g = torch.rand_like(rows)
    (rows * g).sum().backward()
    (ref_rows * g).sum().backward()
    torch.testing.assert_close(logits.grad.float(), ref.grad.float(), rtol=1e-2, atol=1e-5)
    p = torch.log_softmax(logits.detach().float(), -1)
    ent_ref = -(p.exp() * p).sum(-1)
    torch.testing.assert_close(fused_entropy(logits.detach()), ent_ref, rtol=1e-4, atol=1e-4)


@cuda
@pytest.mark.parametrize("nested", [True, False])
def test_nf4_dequant(nested):
    import bitsandbytes.functional as BF
    from solution.nf4 import dequantize_nf4
    torch.manual_seed(0)
    W = torch.randn(1536, 1536, device="cuda", dtype=torch.float16)
    q, st = BF.quantize_4bit(W, blocksize=64, compress_statistics=nested, quant_type="nf4")
    ref = BF.dequantize_4bit(q, st)
    out = dequantize_nf4(q, st)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

@cuda
def test_head_ce():
    from solution.head_loss import fused_head_ce
    torch.manual_seed(0)
    h = torch.randn(37, 64, device="cuda", requires_grad=True)
    w = torch.randn(5000, 64, device="cuda") * 0.2
    tgt = torch.randint(0, 5000, (37,), device="cuda")
    tgt[::4] = -100
    loss, s = fused_head_ce(h, w, tgt, denom=20.0, chunk=16)
    lg = (h.detach().half() @ w.half().t()).float()
    lp = torch.log_softmax(lg, -1)
    ref = F.cross_entropy(lg, tgt, ignore_index=-100, reduction="sum") / 20
    torch.testing.assert_close(loss, ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(s["ent"], -(lp.exp() * lp).sum(-1), rtol=2e-3, atol=2e-3)
    assert (s["arg"].long() == lg.argmax(-1)).float().mean() > 0.97
    loss.backward()
    h2 = h.detach().clone().requires_grad_()
    lg2 = (h2.half() @ w.half().t()).float()
    (F.cross_entropy(lg2, tgt, ignore_index=-100, reduction="sum") / 20).backward()
    torch.testing.assert_close(h.grad, h2.grad, rtol=3e-2, atol=1e-3)

@cuda
def test_rmsnorm_fp16_out():
    from solution.rmsnorm import RMSNormFn
    torch.manual_seed(0)
    x = (torch.randn(3, 37, 1536, device="cuda") * 2).requires_grad_()  # fp32 input
    x_ref = x.detach().clone().requires_grad_()
    w = torch.rand(1536, device="cuda") + 0.5
    dy = torch.randn(3, 37, 1536, device="cuda")
    y = RMSNormFn.apply(x, w, 1e-6, True)
    y_ref = ref_rmsnorm(x_ref, w, 1e-6)
    assert y.dtype == torch.float16
    torch.testing.assert_close(y.float(), y_ref, rtol=5e-3, atol=5e-3)
    y.backward(dy.half())
    y_ref.backward(dy)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=3e-2, atol=3e-2)

@cuda
@pytest.mark.parametrize("out16", [True, False])
def test_add_rmsnorm(out16):
    from solution.rmsnorm import AddRMSNormFn
    torch.manual_seed(0)
    r = (torch.randn(3, 37, 1536, device="cuda") * 2).requires_grad_()
    a = torch.randn(3, 37, 1536, device="cuda").half().requires_grad_()
    r_ref = r.detach().clone().requires_grad_()
    a_ref = a.detach().clone().requires_grad_()
    w = torch.rand(1536, device="cuda") + 0.5
    dy = torch.randn(3, 37, 1536, device="cuda")
    ds = torch.randn(3, 37, 1536, device="cuda")
    s, y = AddRMSNormFn.apply(r, a, w, 1e-6, out16)
    s_ref = r_ref + a_ref
    y_ref = ref_rmsnorm(s_ref, w, 1e-6)
    assert s.dtype == torch.float32
    assert y.dtype == (torch.float16 if out16 else torch.float32)
    torch.testing.assert_close(s, s_ref)
    torch.testing.assert_close(y.float(), y_ref, rtol=5e-3, atol=5e-3)
    ((s * ds).sum() + (y.float() * dy).sum()).backward()
    ((s_ref * ds).sum() + (y_ref * dy).sum()).backward()
    torch.testing.assert_close(r.grad, r_ref.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(a.grad.float(), a_ref.grad.float(), rtol=3e-2, atol=3e-2)

@cuda
def test_unpad_roundtrip():
    from solution.unpad import RowGather, RowScatter
    torch.manual_seed(0)
    x = torch.randn(20, 64, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    idx = torch.tensor([0, 1, 2, 5, 6, 12, 13, 19], device="cuda")
    y = RowScatter.apply(RowGather.apply(x, idx) * 2, idx, 20)
    m = torch.zeros(20, 1, device="cuda")
    m[idx] = 1
    y_ref = x_ref * 2 * m
    torch.testing.assert_close(y, y_ref)
    g = torch.randn_like(y)
    y.backward(g); y_ref.backward(g)
    torch.testing.assert_close(x.grad, x_ref.grad)
    
@cuda
@pytest.mark.parametrize("nested", [True, False])
def test_nf4_patch_transposed(nested):
    import bitsandbytes.functional as BF
    from solution.nf4 import patch_bnb_dequant
    torch.manual_seed(0)
    W = torch.randn(256, 1536, device="cuda", dtype=torch.float16)
    q, st = BF.quantize_4bit(W, blocksize=64, compress_statistics=nested, quant_type="nf4")
    orig = BF.dequantize_4bit
    patch_bnb_dequant()
    for A in (q, q.t()):
        ref = orig(A, st)
        out = BF.dequantize_4bit(A, st)
        assert out.shape == ref.shape
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
