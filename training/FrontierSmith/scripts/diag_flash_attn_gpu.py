"""GPU-side proof that flash-attn actually computes (login node has no GPU).

The prebuilt wheel we install is built against torch 2.10 while the env runs
torch 2.11; `import flash_attn` succeeding only proves ABI compatibility of the
symbol table. This runs a real kernel and checks the result against a reference,
so a silently-wrong kernel cannot pass.
"""
import torch
import flash_attn
from flash_attn import flash_attn_func

print("flash_attn", flash_attn.__version__, "| torch", torch.__version__,
      "| gpu", torch.cuda.get_device_name(0))

torch.manual_seed(0)
B, S, H, D = 2, 4096, 8, 128
q, k, v = (torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") for _ in range(3))
out = flash_attn_func(q, k, v, causal=True)
assert torch.isfinite(out).all(), "flash-attn produced non-finite values"

ref = torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(),
    is_causal=True).transpose(1, 2).to(out.dtype)
err = (out - ref).abs().max().item()
print(f"kernel OK  out={tuple(out.shape)}  max|flash-sdpa|={err:.4f}")
assert err < 0.05, f"flash-attn disagrees with the reference: {err}"
print("FLASH_ATTN_GPU_VERIFIED")
