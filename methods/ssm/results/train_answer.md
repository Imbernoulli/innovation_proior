I want a single sequence-model backbone I can drop under a foundation model for any modality — language, audio, DNA — and at scale the only thing that really works is self-attention. Attention works because every position reads directly from every other through a softmax over query-key dot products; nothing is summarized away, the routing is dense and content-dependent. But that is exactly the bill it keeps charging: training is $O(L^2)$ in length because all pairwise interactions are materialized, autoregressive decoding keeps the whole past resident as an $O(L)$ key/value cache so per-step cost grows without bound, and there is no notion of anything outside the window. The whole zoo of cheaper variants — linear attention, gated convolutions, structured state space models like S4/S4D and S5, the H3 and Hyena blocks — answers the efficiency half: they scale linearly or near-linearly in length. But on information-dense *discrete* data, text above all, they have consistently underperformed attention. So the problem is sharper than "make attention cheaper": I need to name the *specific* capability these efficient models lack on discrete data and supply it without giving up linear-time scaling and a constant-size inference state.

The reframing that makes the landscape legible is that a sequence model compresses context into a state and acts on that state. Attention is the degenerate case that compresses nothing — the cache *is* the uncompressed context — which is why it is both powerful and expensive. A recurrent model with a bounded state sits at the other extreme: cheap, constant work per step, but only as good as whatever that bounded state retained. The whole efficiency-versus-quality axis is therefore one question about the state: how much it can hold, and what governs which information it keeps. The escape is not a bigger state but a *smarter* compression — keep what is relevant, drop what is not — and what counts as relevant depends on the actual tokens. That word, content, is the whole story. The most principled cheap model on the table, the structured SSM, is built from a continuous linear system $h'(t) = A h(t) + B x(t)$, $y(t) = C h(t)$ discretized by zero-order hold to $\bar A = \exp(\Delta A)$, $\bar B = (\Delta A)^{-1}(\exp(\Delta A) - I)\,\Delta B$, giving the linear recurrence $h_t = \bar A h_{t-1} + \bar B x_t$, $y_t = C h_t$. With a HiPPO choice of $A$ the state is a near-optimal online summary of history, genuine long-range memory rather than a decay heuristic. The trouble lies in what buys its efficiency: freezing $(\Delta, A, B, C)$ across all time steps and unrolling from $h_0 = 0$ gives $y_t = \sum_k C \bar A^k \bar B \, x_{t-k}$, a convolution by a single fixed kernel $\bar K = (C\bar B, C\bar A \bar B, C \bar A^2 \bar B, \dots)$ that can be evaluated by FFT in $O(L \log L)$ at training and run as a recurrence at inference. But that single reusable kernel exists *only* because $(\bar A, \bar B, C)$ do not depend on $t$. Time-invariance is the efficiency engine — and time-invariance is exactly the disease. On Selective Copying the gap between a relevant input and its output varies per example because noise tokens fall in between, so a static kernel with fixed lags cannot work, and a constant recurrent transition cannot decide per token whether to keep it or ignore it; Induction Heads, content-determined associative recall, is the same lesson. The time-invariance that gives the convolution is the very thing that forbids content-based selection. They are one property seen from two sides, so the fix is forced: the dynamics must depend on the input.

I propose the selective state space model (the selective SSM, or S6), wrapped in a homogeneous block I call Mamba. The single change is to make the dynamics input-dependent: the step size $\Delta$ and the input/output matrices $B, C$ are produced by linear projections of the sequence at each position, while the diagonal state matrix $A$ stays static. Concretely $B_t = \mathrm{Linear}_N(x_t)$ and $C_t = \mathrm{Linear}_N(x_t)$, each shape $(B, L, N)$ and genuinely per-coordinate because they control *which* state coordinates to write and read, while $\Delta_t = \mathrm{softplus}(b_\Delta + s_\Delta(x_t))$ of shape $(B, L, D)$. I leave $A$ static deliberately: it touches the computation only through $\bar A = \exp(\Delta A)$, and if $\Delta$ is already input-dependent then $\bar A = \exp(\Delta_t A)$ is already input-dependent through $\Delta_t$, so making $A$ selective is a redundant second route to the same place — I keep the principled diagonal HiPPO/S4D memory matrix and let $\Delta$ carry the selection. Why $\Delta$ is the right knob, and why softplus-of-a-linear, is not arbitrary: take the smallest case $N = 1, A = -1, B = 1$, the leaky integrator $h'(t) = -h(t) + x(t)$, with $\Delta_t = \mathrm{softplus}(\mathrm{Linear}(x_t))$. Then $\bar A_t = \exp(-\Delta_t)$, and since $\exp(-\mathrm{softplus}(z)) = 1/(1+e^z) = 1 - \sigma(z)$, while $\bar B_t = -(\exp(-\Delta_t) - 1) = 1 - \bar A_t = \sigma(\mathrm{Linear}(x_t))$, the recurrence collapses to
$$h_t = (1 - g_t)\, h_{t-1} + g_t\, x_t, \qquad g_t = \sigma(\mathrm{Linear}(x_t)),$$
the classical RNN gate. The heuristic gate that LSTMs and GRUs reached for is exactly the special case of selective discretization: a large $\Delta_t$ ($g_t \to 1$) overwrites the old state with the current input, a small $\Delta_t$ ($g_t \to 0$) persists the state and ignores the token. That is the Selective Copying solution — filler tokens get $\Delta \to 0$ and slide past, data tokens get a large $\Delta$ and are written — and it tells me the parametrization of $\Delta$: shared low-dimensional evidence for "write" versus "keep," then a channelwise map to per-channel steps. So $s_\Delta$ is low-rank, projecting to a small rank $R \approx d_\mathrm{model}/16$ and expanding, with the learned base timescale $b_\Delta$ added exactly once right before the softplus. For the discrete modalities I am targeting I use a real-valued state (simpler and hardware-friendly; complex states help only on smooth perceptual signals), a diagonal $A$ with the S4D-Real init $A_n = -(n+1)$ stored in log space to keep $A < 0$, and a $\Delta$ bias initialized so $\mathrm{softplus}(b_\Delta) \in [0.001, 0.1]$, a spread of starting memory horizons.

The moment $\bar A_t, \bar B_t, C_t$ vary with position the unrolling no longer collapses — the coefficient relating $y_t$ to $x_{t-k}$ becomes $C_t \bar A_t \bar A_{t-1}\cdots \bar A_{t-k+1} \bar B_{t-k}$, which depends on $t$, not just on the lag — so there is no single kernel, the FFT-convolution path is gone, and I am forced back onto the recurrence with its two apparent killers: it is sequential, and its expanded hidden state has size $B \cdot L \cdot D \cdot N$, a factor of $N$ larger than the $B \cdot L \cdot D$ input. Neither is actually fatal. The recurrence is sequential but *linear*, and a first-order linear recurrence parallelizes: with diagonal $A$ everything decouples into scalar recurrences, and the pairs $(a, b)$ compose under
$$(a_1, b_1) \bullet (a_2, b_2) = (a_2 a_1,\; a_2 b_1 + b_2),$$
which is composition of the affine maps $h \mapsto a h + b$ and is therefore associative — so computing all $h_t$ from steps $(a_t, b_t) = (\bar A_t, \bar B_t x_t)$ is a prefix scan with $O(L)$ work and $O(\log L)$ depth. On FLOPs the scan costs $O(BLDN)$ versus the convolution's $O(BLD\log L)$, so abandoning the convolution is not even a loss in arithmetic for a modest $N$. The memory blowup I solve with the same discipline that IO-aware attention used: on a GPU everything but dense matmul is bandwidth-bound, so I never write the big intermediate to slow HBM. The inputs $\Delta, A, B, C$ are small (scaling like $O(BLD + BLN + DN)$); the discretized $\bar A, \bar B$ and the running states are the big $B\cdot L\cdot D\cdot N$ object. So I load $\Delta, A, B, C$ into SRAM, do the discretization there, run the scan there keeping the expanded state on chip, multiply by $C$ and sum over $N$ to collapse it, and write back only $y$ of size $B\cdot L\cdot D$ — cutting asymptotic HBM traffic by a factor on the order of $N$, a reported 20–40$\times$ kernel speedup over a standard scan. And to keep the backward pass from reintroducing the blowup, I do not store the intermediate states; I recompute them in SRAM from the reloaded inputs and the output gradient. One implementation note that the gate derivation explains but the code economizes: the canonical scan keeps the exact transition $\bar A = \exp(\Delta A)$ but uses the first-order input update $\bar B x_t \approx (\Delta \odot B)\, x_t$, which is why the code forms $\mathrm{delta} \cdot B \cdot u$. Finally I fold the architecture into one homogeneous block rather than the H3 SSM-block-then-MLP-block alternation: project width $D$ up by $E = 2$ into a main branch and a gate branch, run the main branch through a short causal depthwise conv1d (the cheap local mixing H3 framed as a shift-SSM), a SiLU, then the selective SSM; pass the gate branch through SiLU and multiply it in (a SwiGLU-style gated MLP); project back to $D$; wrap in a pre-norm residual. Most parameters are the projections, about $3ED^2$ per block, so two of these blocks ($\approx 12D^2$) match one Transformer layer. The result is linear-time in training, constant-state at inference, and content-aware — and selectivity gives it a concrete mechanism to ignore transient inputs, flush old history, and reset at sequence boundaries that no fixed-kernel LTI model has.

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
