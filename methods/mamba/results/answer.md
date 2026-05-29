# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## The problem

Match the modeling quality of attention while scaling **linearly** in sequence length and keeping a **constant-size** state during autoregressive inference. Attention is effective because it compresses nothing (the KV cache stores the whole context), which is also why it costs O(L²) to train and O(L) memory per step to decode. Prior subquadratic models (linear attention, S4/S4D, S5, H3, Hyena) are efficient but underperform on dense discrete data like text. The diagnosis: they are **linear time-invariant (LTI)** — their dynamics are constant across time — so they cannot perform **content-based selection** (decide, per token, what to keep or discard). This is exactly what tasks like Selective Copying and Induction Heads require.

## The key idea

Make the SSM parameters **functions of the input** (the *selection mechanism*). Starting from the continuous SSM h′=Ah+Bx, y=Ch, exact zero-order hold gives a discretized diagonal SSM

  h_t = Ā_t h_{t−1} + B̄_t x_t,  y_t = C_t h_t,  with Ā = exp(Δ A), B̄ = (ΔA)^{−1}(exp(ΔA) − I)·ΔB,

and the scalar check is h′=ah+bx with a<0 and x held constant, so h(Δ)=e^{aΔ}h(0)+((e^{aΔ}−1)/a)·b·x.

let Δ, B, C depend on the current token (A stays static — it only acts through Ā = exp(ΔA), so selective Δ already makes the transition selective):

  s_B(x) = Linear_N(x),  s_C(x) = Linear_N(x),  r_Δ(x) = Linear_R(x),  s_Δ(x) = Linear_D(r_Δ(x)),  Δ = softplus(b_Δ + s_Δ(x)).

The learned b_Δ is added exactly once: the forward pass forms s_Δ with the projection weight only, then the scan adds b_Δ immediately before softplus. This turns the model time-varying, which **breaks the convolution form** (no single fixed kernel exists when Ā_t, B̄_t, C_t vary). The implementation keeps Ā=exp(ΔA) and uses the first-order input update B̄≈ΔB, so the scan computes B̄x_t as Δ_t B_t x_t. Efficiency is recovered with a **hardware-aware selective scan**:

- **Parallel scan** — the recurrence is first-order *linear*, so it is an associative prefix scan under (a,b)•(a′,b′) = (a′a, a′b+b′): O(L) work, O(log L) depth.
- **Kernel fusion** — load (Δ,A,B,C) from HBM to SRAM, discretize and scan and multiply by C entirely in SRAM, write back only y of size (B,L,D); the expanded (B,L,D,N) state never touches slow memory (≈N× less IO).
- **Recomputation** — don't store intermediate scan states for backward; recompute them from the small inputs. Activation memory ≈ FlashAttention.

A special case makes the Δ choice principled: with N=1, A=−1, B=1, the leaky integrator h′=−h+x discretized with exact ZOH and input-dependent Δ=softplus(Linear(x)) gives Ā=1−σ(Linear(x)), B̄=σ(Linear(x)), i.e. **h_t=(1−g_t)h_{t−1}+g_t x_t with g_t=σ(Linear(x_t))** — the classic RNN gate. So selective Δ generalizes gating; softplus and the input-linear form are not arbitrary.

## The architecture

Fuse the H3 SSM block and the Transformer MLP block into one homogeneous block (inspired by the gated attention unit): expand width D by E=2 into two branches; the main branch runs short causal conv → SiLU → selective SSM; the gate branch runs SiLU and multiplies in; project back to D; wrap in pre-norm residual. Parameters ≈ 6D² per block (E=2), so two blocks ≈ a Transformer layer's 12D² (MHA+MLP). Defaults: real-valued diagonal A (S4D-Real init A_n=−(n+1)), N=16, Δ-bias init so softplus∈[0.001,0.1], low-rank Δ projection dt_rank≈D/16.

## Working code

```python
# Selective SSM block + reference selective scan.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True):
    """Reference (pure-PyTorch) selective scan. The fused CUDA kernel does the same
    arithmetic, but discretizes/scans in SRAM and recomputes states in the backward pass.
      u, delta: (B, D, L)   A: (D, N)   B, C: (B, N, L)   D: (D,)   z: (B, D, L)
    """
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)                                    # Δ = softplus(delta_bias + low-rank Linear(x)) > 0
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    B, C = B.float(), C.float()

    deltaA   = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))     # Ā = exp(Δ A)
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)        # B̄ x_t ≈ (Δ ⊙ B) x_t

    x = A.new_zeros((batch, dim, dstate))
    ys = []
    for i in range(u.shape[2]):                                      # linear recurrence == associative scan
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]                  # h_t = Ā_t h_{t-1} + B̄_t x_t
        ys.append(torch.einsum('bdn,bn->bd', x, C[:, :, i]))         # y_t = C_t h_t (selective read-out)
    y = torch.stack(ys, dim=2)                                       # (B, D, L)

    out = y if D is None else y + u * rearrange(D, "d -> d 1")       # skip connection
    if z is not None:
        out = out * F.silu(z)                                        # multiplicative gate (z branch)
    return out.to(dtype=dtype_in)


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, conv_bias=True, bias=False):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)   # -> [x, z]
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=conv_bias)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)  # -> [dt, B, C]
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)                # low-rank Δ -> per-channel steps

        # Δ-bias init so softplus(bias) ∈ [dt_min, dt_max]
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)   # S4D-Real
        self.A_log = nn.Parameter(torch.log(A))                          # A = -exp(A_log) < 0
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states):                                    # (B, L, D)
        b, l, _ = hidden_states.shape
        xz = rearrange(self.in_proj(hidden_states), "b l d2 -> b d2 l")
        x, z = xz.chunk(2, dim=1)

        x = self.act(self.conv1d(x)[..., :l])                           # short causal conv + SiLU

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # dt: weight only here; the bias is folded into the scan via delta_bias (avoids double-counting).
        dt = rearrange(self.dt_proj.weight @ dt.t(), "d (b l) -> b d l", l=l)  # selective Δ (per channel)
        B  = rearrange(B, "(b l) n -> b n l", l=l).contiguous()          # selective B
        C  = rearrange(C, "(b l) n -> b n l", l=l).contiguous()          # selective C
        A  = -torch.exp(self.A_log.float())

        y = selective_scan_ref(x, dt, A, B, C, D=self.D.float(), z=z,
                               delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        return self.out_proj(rearrange(y, "b d l -> b l d"))


class Block(nn.Module):
    """Pre-norm residual wrapper; the Mamba block IS the homogeneous layer (no separate MLP)."""
    def __init__(self, d_model, norm_cls=nn.LayerNorm):
        super().__init__()
        self.norm = norm_cls(d_model)
        self.mixer = Mamba(d_model)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class MambaLM(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layer=24):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([Block(d_model) for _ in range(n_layer)])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight                     # tie weights

    def forward(self, input_ids):
        h = self.embedding(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.norm_f(h))
```

For autoregressive inference, the same block runs one token at a time as a true recurrence: cache the conv window and the SSM state h (size D·N), and per step compute Ā=exp(ΔA), B̄≈ΔB, h←Ā·h+B̄·x, y=C·h+D·x, y←y·SiLU(z) — constant work and constant memory per token, no growing cache.
