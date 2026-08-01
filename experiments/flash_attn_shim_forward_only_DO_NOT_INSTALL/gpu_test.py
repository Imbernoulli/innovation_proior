import os, sys, traceback
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cuda"
assert torch.cuda.is_available(), "need GPU"
print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0))

import flash_attn
from flash_attn import flash_attn_varlen_func, flash_attn_func
from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis
print("flash_attn", flash_attn.__version__, "from", flash_attn.__file__)

results = {}


def sdpa_varlen_ref(q, k, v, cu, causal, scale):
    # q,k,v: (total, H, D) fp32 ; per-sequence sdpa; returns (total, Hq, D)
    outs = []
    Hq = q.shape[1]
    Hk = k.shape[1]
    for i in range(len(cu) - 1):
        s, e = cu[i].item(), cu[i + 1].item()
        qi = q[s:e].transpose(0, 1)  # (Hq, L, D)
        ki = k[s:e].transpose(0, 1)  # (Hk, L, D)
        vi = v[s:e].transpose(0, 1)
        if Hk != Hq:
            rep = Hq // Hk
            ki = ki.repeat_interleave(rep, dim=0)
            vi = vi.repeat_interleave(rep, dim=0)
        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal, scale=scale)
        outs.append(oi.transpose(0, 1))  # (L, Hq, D)
    return torch.cat(outs, 0)


def run_forward(name, seqlens, Hq, Hk, D, causal):
    total = sum(seqlens)
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=DEV)
    cu[1:] = torch.tensor(seqlens, device=DEV).cumsum(0)
    maxs = max(seqlens)
    scale = 1.0 / (D ** 0.5)
    q = torch.randn(total, Hq, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(total, Hk, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(total, Hk, D, device=DEV, dtype=torch.bfloat16)
    out = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                 max_seqlen_q=maxs, max_seqlen_k=maxs,
                                 causal=causal, softmax_scale=scale)
    ref = sdpa_varlen_ref(q.float(), k.float(), v.float(), cu, causal, scale)
    diff = (out.float() - ref).abs().max().item()
    ok = diff < 2e-2
    results[name] = (ok, diff)
    print(f"[FWD] {name:38s} causal={causal} Hq={Hq} Hk={Hk} D={D}  max|diff|={diff:.4e}  {'PASS' if ok else 'FAIL'}")


# ---- 1. varlen forward numerics ----
for causal in (False, True):
    run_forward(f"varlen_MHA_c{int(causal)}", [13, 47, 100, 5], 8, 8, 128, causal)
    run_forward(f"varlen_GQA_c{int(causal)}", [16, 64, 33], 8, 2, 128, causal)

# ---- 2. non-varlen flash_attn_func forward ----
try:
    B, S, H, D = 3, 40, 8, 128
    scale = 1.0 / (D ** 0.5)
    q = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16)
    for causal in (False, True):
        out = flash_attn_func(q, k, v, causal=causal, softmax_scale=scale)
        ref = F.scaled_dot_product_attention(
            q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2),
            is_causal=causal, scale=scale).transpose(1, 2)
        diff = (out.float() - ref).abs().max().item()
        ok = diff < 2e-2
        results[f"flash_attn_func_c{int(causal)}"] = (ok, diff)
        print(f"[FWD] flash_attn_func c={int(causal)}                     max|diff|={diff:.4e}  {'PASS' if ok else 'FAIL'}")
except Exception as e:
    results["flash_attn_func"] = (False, str(e))
    print("[FWD] flash_attn_func ERROR:", e)

# ---- 3. padding round-trip: unpad -> varlen -> pad ----
try:
    B, S, H, D = 4, 50, 8, 128
    scale = 1.0 / (D ** 0.5)
    hidden = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16)
    lens = torch.tensor([50, 30, 17, 3], device=DEV)
    mask = (torch.arange(S, device=DEV)[None, :] < lens[:, None]).int()  # (B,S)
    q = hidden.clone(); k = torch.randn_like(hidden); v = torch.randn_like(hidden)
    unpad = unpad_input(q.reshape(B, S, H * D), mask)
    q_un, indices, cu_q, max_q = unpad[0], unpad[1], unpad[2], unpad[3]
    k_un = index_first_axis(k.reshape(B * S, H * D), indices)
    v_un = index_first_axis(v.reshape(B * S, H * D), indices)
    q_un = q_un.reshape(-1, H, D); k_un = k_un.reshape(-1, H, D); v_un = v_un.reshape(-1, H, D)
    mq = int(max_q) if not torch.is_tensor(max_q) else int(max_q.item())
    o_un = flash_attn_varlen_func(q_un, k_un, v_un, cu_seqlens_q=cu_q, cu_seqlens_k=cu_q,
                                  max_seqlen_q=mq, max_seqlen_k=mq, causal=True, softmax_scale=scale)
    o_pad = pad_input(o_un.reshape(-1, H * D), indices, B, S).reshape(B, S, H, D)
    # reference per valid sequence
    ref = torch.zeros_like(o_pad, dtype=torch.float32)
    for i in range(B):
        L = int(lens[i])
        qi = q[i, :L].float().transpose(0, 1); ki = k[i, :L].float().transpose(0, 1); vi = v[i, :L].float().transpose(0, 1)
        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True, scale=scale).transpose(0, 1)
        ref[i, :L] = oi
    validmask = mask.bool()[..., None, None].expand_as(o_pad)
    diff = (o_pad.float()[validmask] - ref[validmask]).abs().max().item()
    ok = diff < 2e-2
    results["padding_roundtrip"] = (ok, diff)
    print(f"[PAD] unpad->varlen->pad valid-token             max|diff|={diff:.4e}  {'PASS' if ok else 'FAIL'}")
except Exception as e:
    results["padding_roundtrip"] = (False, str(e))
    print("[PAD] ERROR:", e); traceback.print_exc()

# ---- 4. BACKWARD (decisive) ----
print("\n=== BACKWARD / autograd test (required for actor training) ===")
try:
    seqlens = [13, 47, 20]
    total = sum(seqlens); D = 128; H = 8
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=DEV)
    cu[1:] = torch.tensor(seqlens, device=DEV).cumsum(0)
    maxs = max(seqlens); scale = 1.0 / (D ** 0.5)
    q = torch.randn(total, H, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(total, H, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(total, H, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    out = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                 max_seqlen_q=maxs, max_seqlen_k=maxs, causal=True, softmax_scale=scale)
    print("  forward out.requires_grad =", out.requires_grad, "| grad_fn =", out.grad_fn)
    loss = out.float().pow(2).sum()
    loss.backward()
    gq_ok = q.grad is not None and torch.isfinite(q.grad).all().item()
    results["backward"] = (gq_ok, "grad produced" if gq_ok else "no/invalid grad")
    print(f"  backward SUCCEEDED: q.grad is {'present & finite' if gq_ok else 'MISSING/NaN'}")
except Exception as e:
    results["backward"] = (False, f"{type(e).__name__}: {e}")
    print(f"  backward FAILED -> {type(e).__name__}: {e}")

# ---- 5. transformers integration ----
print("\n=== transformers integration ===")
try:
    import transformers
    from transformers.utils import is_flash_attn_2_available
    fa2 = is_flash_attn_2_available()
    print("  transformers", transformers.__version__, "is_flash_attn_2_available() =", fa2)
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
    print("  imported _flash_attention_forward OK")
    results["transformers_fa2_available"] = (bool(fa2), fa2)
    # try a forward+backward through transformers' packed path
    try:
        from transformers.modeling_flash_attention_utils import lazy_import_flash_attention
        lazy_import_flash_attention("flash_attention_2")
        B, S, H, D = 2, 24, 8, 128
        qs = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
        ks = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
        vs = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
        o = _flash_attention_forward(qs, ks, vs, attention_mask=None, query_length=S,
                                     is_causal=True, attn_implementation="flash_attention_2")
        print("  _flash_attention_forward ran; out.requires_grad =", o.requires_grad)
        o.float().sum().backward()
        print("  transformers path backward SUCCEEDED; qs.grad finite =",
              (qs.grad is not None and torch.isfinite(qs.grad).all().item()))
        results["transformers_backward"] = (True, "ok")
    except Exception as e:
        results["transformers_backward"] = (False, f"{type(e).__name__}: {e}")
        print(f"  transformers path backward FAILED -> {type(e).__name__}: {e}")
except Exception as e:
    results["transformers_fa2_available"] = (False, str(e))
    print("  transformers integration ERROR:", e); traceback.print_exc()

# ---- summary ----
print("\n================ SUMMARY ================")
allpass = True
for k_, (ok, info) in results.items():
    print(f"  {k_:34s} {'PASS' if ok else 'FAIL'}   {info}")
    allpass = allpass and ok
print("ALL PASS:", allpass)
