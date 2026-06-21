I want a single sequence model that can serve as the backbone of a foundation model across language, audio, and genomics, and the uncomfortable fact is that at scale only attention really works. Attention is effective because it compresses nothing: every position looks directly at every other position, the routing is dense and content-dependent, and the key/value cache literally *is* the uncompressed context. That same property is the bill. Training is quadratic in the sequence length $L$ because I form interactions among all $L^2$ pairs; autoregressive decoding must keep the entire past resident, so generating token $t$ costs work and memory growing with $t$; and there is no notion of anything outside the window. A large zoo of subquadratic alternatives — linear attention, gated convolutions, recurrent models, the structured state space models S4/S4D/S5, and the SSM-architecture blocks H3 and Hyena — answers the efficiency half: they are cheap and scale linearly. But on information-dense discrete modalities, language above all, they have consistently underperformed attention. So the right question is not "how do I approximate attention more cheaply" but "what specific capability are the efficient models missing, and can I restore it without surrendering linear-time scaling."

The reframing that makes the landscape legible is that a sequence model compresses context into a state and then acts on that state. Attention is the degenerate case where the compression is the identity — it keeps everything, which is why it is both powerful and expensive. A recurrent model with a fixed-size state sits at the other extreme — cheap, constant work per step, no growing cache — but only as good as whatever its bounded state managed to hold. So the entire efficiency-versus-quality axis collapses to one question: how well is a bounded state used. The structured SSMs are the most principled cheap model on the table. They start from the continuous linear system $h'(t)=Ah(t)+Bx(t)$, $y(t)=Ch(t)$, discretized with a step $\Delta$ by zero-order hold into $\bar A=\exp(\Delta A)$, $\bar B=(\Delta A)^{-1}(\exp(\Delta A)-I)\cdot\Delta B$, giving the recurrence $h_t=\bar A h_{t-1}+\bar B x_t$, $y_t=Ch_t$. (The scalar ZOH check: $h'=ah+bx$ with $x$ held constant over $[0,\Delta]$ integrates to $h(\Delta)=e^{a\Delta}h(0)+\frac{e^{a\Delta}-1}{a}\,b\,x_t$, matching the formula.) Their power comes from a special $A$ — the HiPPO matrix — that makes the state a near-optimal online summary of history, and S4D showed a plain diagonal $A$ of $N$ numbers per channel works essentially as well. Crucially, when $(\Delta,A,B,C)$ are frozen across time, unrolling collapses into a single fixed convolution kernel $\bar K=(C\bar B, C\bar A\bar B, C\bar A^2\bar B,\dots)$ with $y=x*\bar K$, which an FFT evaluates in $O(L\log L)$ during training. That convolution is the whole efficiency engine — and it exists *only* because the dynamics are time-invariant, so one kernel can be reused at every position. That very time-invariance is the disease. On Selective Copying, the tokens to remember sit at random positions with noise interspersed, so the input-to-output spacing depends on content; a static kernel has fixed lags that cannot adapt, and a constant recurrence applies the same transition to signal and garbage alike, unable to decide "this is noise, do not write it." Induction Heads is the same lesson in associative-recall clothing. The property I love is exactly the property that forbids content-based selection.

I propose Mamba, a selective state space model. The fix is forced: make the SSM parameters functions of the input, so the model can look at $x_t$ and change how it treats $x_t$. Concretely I let $\Delta$, $B$, and $C$ depend on the current token while $A$ stays static: $s_B(x)=\mathrm{Linear}_N(x)$, $s_C(x)=\mathrm{Linear}_N(x)$, and for the step $\Delta=\mathrm{softplus}(b_\Delta+\mathrm{Linear}_D(\mathrm{Linear}_R(x)))$. I deliberately leave $A$ static because $A$ acts only through $\bar A=\exp(\Delta A)$; with $\Delta_t$ already input-dependent, $\bar A_t$ is already selective, so making $A$ selective would be a redundant second route to the same place — fewer parameters, nothing lost. Why $\Delta$, $B$, $C$ specifically: $B$ controls how the current input is written into the state, $C$ how the state is read out, and $\Delta$ — the timescale — controls how much the state persists versus gets overwritten, which is precisely the lever for "ignore this token" versus "keep it." That selection on $\Delta$ is not arbitrary, and the cleanest way to see it is the minimal case $N=1$, $A=-1$, $B=1$, the leaky integrator $h'=-h+x$, with $\Delta_t=\mathrm{softplus}(\mathrm{Linear}(x_t))$. Then $\bar A_t=\exp(-\Delta_t)=\exp(-\mathrm{softplus}(z))=\sigma(-z)=1-\sigma(z)$, and ZOH gives $\bar B_t=1-\bar A_t=\sigma(z)$. Writing $g_t=\sigma(\mathrm{Linear}(x_t))$, the recurrence becomes
$$h_t=(1-g_t)\,h_{t-1}+g_t\,x_t,$$
the exact classical RNN gate — a convex combination of keep-the-old-state and write-the-new-input. I did not put a gate in by hand; it fell out of input-dependent discretization. Large $\Delta_t$ (gate $\to 1$) overwrites with the current token, small $\Delta_t$ (gate $\to 0$) persists and lets filler slide past, and $\Delta\to\infty$ resets the state at a boundary — exactly the Selective Copying solution. This also fixes the form of the projections: $\Delta$ is generated low-rank ($r=\mathrm{Linear}_R(x)$ with rank $\approx D/16$, then a channelwise map to per-channel steps) because the "write or keep" evidence is shared and low-dimensional, while $B$ and $C$ are full width $N$ because they genuinely select which state coordinates to write and read. The bias $b_\Delta$ is the learned base timescale, so it must be added exactly once, immediately before the softplus.

The price of selection is that the moment $\bar A_t,\bar B_t,C_t$ vary with position the coefficient relating $y_t$ to $x_{t-k}$ becomes $C_t\bar A_t\bar A_{t-1}\cdots\bar A_{t-k+1}\bar B_{t-k}$, which depends on $t$, not just on the lag $k$ — there is no single kernel, the convolution and its FFT are gone. This is the trade the whole field had been avoiding, and it is why every structured SSM stayed time-invariant. I recover the lost efficiency with a hardware-aware selective scan built from three classical ideas. First, the recurrence is still *linear* in $h$, and first-order linear recurrences parallelize: viewing each step as the affine map $h\mapsto a h+b$ with $(a_t,b_t)=(\bar A_t,\bar B_t x_t)$, composition gives the associative operator $(a_1,b_1)\bullet(a_2,b_2)=(a_2a_1,\ a_2b_1+b_2)$, and a work-efficient prefix scan computes every $h_t$ in $O(L)$ work and $O(\log L)$ depth — associativity is all the scan needs, the time-varying coefficients are no obstacle. Second, the memory blowup is real: the expanded state has shape $B\cdot L\cdot D\cdot N$, a factor $N$ larger than the $B\cdot L\cdot D$ data, and naively materializing $\bar A,\bar B$ and the running states in slow HBM is what kills it. So I fuse the kernel: load the small inputs $(\Delta,A,B,C)$ from HBM into fast SRAM, do the discretization, the scan, and the multiply-by-$C$ entirely in SRAM, collapse the state dimension away, and write back only $y$ of size $B\cdot L\cdot D$. The $N$-times-larger state is born and dies on chip, cutting HBM traffic by roughly $N$. Third, to keep the backward pass from reintroducing that blowup I do not store the intermediate scan states; I recompute them from the small inputs, which is cheaper than the HBM reads it avoids and lands activation memory near an optimized attention implementation. The scan also costs $O(BLDN)$ versus the convolution's $O(BLD\log L)$, so for long sequences and modest $N$ it is not even more arithmetic. The implementation keeps the exact transition $\bar A=\exp(\Delta A)$ but uses the first-order input update $\bar B\approx\Delta B$, so the scan forms $\bar B x_t$ as $\Delta_t B_t x_t$. Real-valued diagonal $A$ with the S4D-Real init $A_n=-(n+1)$ is the default — complex states help oscillatory perceptual signals but real is simpler, hardware-friendlier, and fine for the discrete data where LTI SSMs failed — and the $\Delta$ bias is initialized so $\mathrm{softplus}(b_\Delta)\in[0.001,0.1]$, a spread of memory horizons.

Finally the block. Rather than the H3-shaped mixer-then-MLP pair, I fuse the SSM path and the gated MLP into one homogeneous block (in the spirit of the gated attention unit). The input of width $D$ is projected up by an expansion $E=2$ into two branches of width $E\cdot D$: a main branch $x$ and a gate branch $z$. The main branch runs a short causal depthwise conv1d (the cheap local mixing that H3's shift-SSM did), a SiLU, then the selective SSM where all content-dependent sequence mixing happens; the gate branch passes through SiLU and multiplies in elementwise (making the gated part behave like a SwiGLU). The result projects back to $D$, wrapped in a pre-norm residual. Each block is $\approx 6D^2$ parameters at $E=2$, so two blocks $\approx$ one Transformer layer's $\approx 12D^2$ — the right way to match capacity is two Mamba blocks per Transformer layer. Stacking this single block homogeneously, with norms and residuals, is the entire architecture: no attention, no separate MLP. Selectivity also buys a quiet property the old models lacked — being able to drive $\Delta\to\infty$ and wipe irrelevant history means quality can improve monotonically with context and stitched sequences can be reset at boundaries instead of bleeding into each other.

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
