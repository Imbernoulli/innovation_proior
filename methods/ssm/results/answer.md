# Selective state space model (selective SSM / S6), distilled

A selective state space model is a structured SSM whose dynamics are made input-dependent: the
step size `Δ` and the input/output matrices `B`, `C` are produced by linear projections of the
sequence at each position, while the diagonal state matrix `A` stays static. This single change
restores content-based selection — the ability to write a token into the recurrent state, ignore
it, or reset the state — which strictly time-invariant SSMs cannot do. Because input-dependent
parameters break the convolution form, the recurrence is evaluated directly, but kept linear-time
and memory-efficient with a hardware-aware fused parallel scan. Wrapped in one homogeneous block
(the Mamba block) it scales linearly in sequence length and uses a constant-size state at
inference.

## Problem it solves

A subquadratic sequence-model backbone that matches attention's quality on dense discrete data
(language, genomics) while scaling linearly in length and keeping a constant per-step inference
state. The capability efficient predecessors lacked was content-based selection: the ability to
filter or retain information conditioned on the actual tokens.

## Key idea

Sequence modeling is compressing context into a state; the compression must be content-dependent.
Make the SSM time-varying by letting `Δ, B, C` depend on the input:

```
B_t = s_B(x_t) = Linear_N(x_t)            # (B, L, N), selective
C_t = s_C(x_t) = Linear_N(x_t)            # (B, L, N), selective
Δ_t = softplus(Param + s_Δ(x_t)),  s_Δ = low-rank: Linear_D(Linear_R(x))   # (B, L, D)
A   : static, diagonal                    # (D, N); only enters via Ā = exp(ΔA)
```

`A` is left static because it affects the model only through `Ā = exp(Δ A)`; selectivity in `Δ`
already makes `Ā` (and `B̄`) selective, so making `A` selective is redundant.

For the mathematical SSM, zero-order-hold discretization gives `Ā = exp(Δ A)` and
`B̄ = (Δ A)^{-1}(exp(Δ A) − I)·Δ B`. The canonical code keeps the exact transition
`Ā = exp(Δ A)` but uses the simplified input update `B̄ x_t ≈ (Δ B_t) x_t`, so the scan forms
`delta * B * u`. The recurrence is `h_t = Ā_t h_{t−1} + B̄_t x_t`, `y_t = C_t h_t`.

## Connection to gating (why softplus + linear Δ)

With `N = 1, A = −1, B = 1, s_Δ = Linear(x), τ_Δ = softplus`, the continuous system is the leaky
integrator `h'(t) = −h(t) + x(t)`. Then `Δ_t = softplus(Linear(x_t))` and ZOH gives

```
Ā_t = exp(−Δ_t) = 1/(1 + exp(Linear(x_t))) = 1 − σ(Linear(x_t)),
B̄_t = −(exp(−Δ_t) − 1) = 1 − Ā_t = σ(Linear(x_t)),
```

so with `g_t = σ(Linear(x_t))` the recurrence is the classical RNN gate
`h_t = (1 − g_t) h_{t−1} + g_t x_t`. The heuristic gate is the special case of selective
discretization: large `Δ` (g→1) forgets the previous state and focuses on the current input,
while small `Δ` (g→0) persists/ignores. A literal zero reset also requires the current boundary
input/write term to be zeroed or learned appropriately. This is the principled reason for the
softplus and linear-in-input form; the minimal construction uses a one-dimensional broadcast gate,
and the implementation uses its low-rank generalization.

## Efficient computation

Input-dependent `(Ā_t, B̄_t, C_t)` make the coefficient of `y_t` on `x_{t−k}` depend on `t`, so
there is no single convolution kernel — the FFT-convolution training path is lost. Recover
efficiency with three classical techniques:

- **Parallel scan.** The recurrence is first-order *linear*, so it is a prefix scan under the
  associative operator `(a, b) • (a', b') = (a' a, a' b + b')` (composition of affine maps
  `h ↦ a h + b`): `O(L)` work, `O(log L)` depth. FLOPs `O(BLDN)` (linear in `L`, low constant)
  versus `O(BLD log L)` for the FFT convolution.
- **Kernel fusion.** The expanded state has size `B·L·D·N` (a factor of `N` larger than the
  `B·L·D` input/output). Load the smaller `Δ, A, B, C` tensors (scaling like
  `O(BLD + BLN + DN)` in the variable-B/C path), discretize, scan, and multiply by `C` entirely
  in SRAM, then write back only `y` of size `B·L·D`. This cuts asymptotic HBM IO by `O(N)`;
  reported kernel speedups are 20–40× versus a standard scan implementation.
- **Recomputation.** Do not store the intermediate states for the backward pass; recompute them
  in SRAM from the reloaded inputs and output gradient, avoiding `O(BLDN)` HBM reads. Net
  activation memory matches an optimized attention implementation.

## Architecture (homogeneous block)

Combine the H3-style SSM block with the MLP block into one repeated block. Expand `D` by factor
`E` (use `E = 2`) into two branches; the main branch goes through a short causal depthwise
conv1d, SiLU, then the selective SSM; the gate branch goes through SiLU and multiplies the main
branch (so the gated MLP is a SwiGLU-style unit); an output projection returns to width `D`;
wrap in a pre-norm residual. Most parameters are the projections (`3ED²`/block; the SSM's
`Δ, B, C` projections and `A` are tiny). Two such blocks (`≈12D²`) match one Transformer layer
(`4D²` MHA + `8D²` MLP).

## Defaults and why

- `d_state = N` (e.g. 16): the per-channel recurrent state; larger `N` compresses more context
  but multiplies materialized state by `N` — affordable only because of the fused scan.
- `expand = E = 2`: capacity-matches a Transformer layer with two blocks.
- `dt_rank = ceil(d_model / 16)`: low-rank `Δ` projection; negligible parameters next to the main
  projections, generalizing the one-dimensional broadcast gate.
- `A = −exp(A_log)`, `A_log = log(arange(1, N+1))`: S4D-Real init, `A_n = −(n+1)`, a spread of
  decay rates; stored in log space to keep `A < 0`.
- Δ bias initialized so `softplus(bias) ∈ [0.001, 0.1]`: a spread of starting memory horizons.
- real-valued state: simpler and works well on discrete (text/DNA) modalities, the target here.

## Working code

Selective scan (sequential reference for the real-valued variable-B/C path used by the Mamba
block; the canonical reference also handles constant, grouped, and complex `B, C` with the same
state update):

```python
import torch
import torch.nn.functional as F


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True):
    # u: (B, D, L) main signal; delta: (B, D, L) step before softplus
    # A: (D, N) static diagonal (A = -exp(A_log) < 0); B, C: (B, N, L) selective
    # D: (D,) skip; z: (B, D, L) gate branch
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)                                    # Δ > 0
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]

    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))       # Ā = exp(Δ A)
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)        # B̄ x ≈ (Δ ⊙ B) x

    h = A.new_zeros((batch, dim, dstate))
    ys = []
    for t in range(u.shape[2]):
        h = deltaA[:, :, t] * h + deltaB_u[:, :, t]                  # h_t = Ā_t h_{t-1} + B̄_t x_t
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, :, t]))        # y_t = C_t h_t
    y = torch.stack(ys, dim=2)                                       # (B, D, L)

    if D is not None:
        y = y + u * D[..., None]
    if z is not None:
        y = y * F.silu(z)
    return y
```

Reference-case map: if `B` is constant the canonical function uses
`einsum("bdl,dn,bdl->bdln", delta, B, u)`; if `B` or `C` are grouped 4D tensors it repeats groups
across channels before the same recurrence; if `A` is complex it views paired real tensors as
complex and returns `2 * real(y)`. The Mamba block above uses the real variable-B/C path.

The Mamba block:

```python
import math
import torch
import torch.nn as nn
from einops import rearrange


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        super().__init__()
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)   # -> (x, z)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Δ bias init so softplus(bias) ~ Uniform([dt_min, dt_max]); inverse-softplus.
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit = True

        # S4D-Real: A = -exp(A_log), A_n = -(n+1); stored in log space, kept in fp32.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))                   # per-channel skip
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, hidden_states):           # (B, L, D)
        b, l, _ = hidden_states.shape
        xz = rearrange(self.in_proj(hidden_states), "b l two_d -> b two_d l")
        x, z = xz.chunk(2, dim=1)

        x = self.act(self.conv1d(x)[..., :l])   # causal depthwise conv (truncate pad) + SiLU

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.A_log.shape[1], self.A_log.shape[1]], dim=-1)
        dt = rearrange(self.dt_proj.weight @ dt.t(), "d (b l) -> b d l", l=l)  # weight only; bias -> scan
        B = rearrange(B, "(b l) n -> b n l", l=l)
        C = rearrange(C, "(b l) n -> b n l", l=l)
        A = -torch.exp(self.A_log.float())

        y = selective_scan_ref(x, dt, A, B, C, D=self.D.float(), z=z,
                               delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        return self.out_proj(rearrange(y, "b d l -> b l d"))
```

## Properties of selection

- *Variable spacing:* the gate can drive `g_t → 0` to filter noise tokens between relevant
  inputs (the Selective Copying mechanism; language fillers like "um").
- *Filtering context:* a selective model has a mechanism to ignore transient inputs or flush old
  history, unlike LTI models whose fixed kernel cannot condition on the content being seen.
- *Boundary resetting:* with large `Δ_t` (or `g_t → 1` in the exact gate case), the previous state
  can be forgotten at sequence boundaries; the current boundary input/write term determines
  whether that is a literal zero reset or an overwrite by a learned boundary state.
