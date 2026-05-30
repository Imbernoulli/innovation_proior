# RWKV (Receptance Weighted Key Value)

## Problem

Train and deploy large autoregressive language models with **parallel training like a Transformer** and **linear-time, constant-memory inference like an RNN** — and do it *without* approximating attention. Self-attention is `O(T²d)` time and keeps a growing KV cache at decode; an RNN is `O(Td)` time with `O(d)` inference state but its recurrence is sequential in time and gradient-unstable.

## Key idea

Start from the Attention Free Transformer's dot-product-free reweighting `Σ_i e^{w_{t,i}+k_i} ⊙ v_i / Σ_i e^{w_{t,i}+k_i}`, then constrain the pairwise position bias to be **linear in the time gap and per-channel**, `w_{t,i} = -(t-i)w` with `w ∈ (ℝ_{≥0})^d`. This turns the normalized weighted average of values into an exact two-vector running state (a linear-attention recurrence), giving an architecture that is a Transformer at training time and an RNN at inference time. The four elements: **R**eceptance (gate for incoming context), **W**eight (per-channel time decay), **K**ey, **V**alue.

## Final method

**WKV operator** (the linear attention). With per-channel decay `w` and current-token bonus `u`:

- Explicit form: `wkv_t = [Σ_{i<t} e^{-(t-1-i)w+k_i} ⊙ v_i + e^{u+k_t} ⊙ v_t] / [Σ_{i<t} e^{-(t-1-i)w+k_i} + e^{u+k_t}]`
- Recurrent form (`a_0=b_0=0`):
  - `wkv_t = (a_{t-1} + e^{u+k_t} ⊙ v_t) / (b_{t-1} + e^{u+k_t})`
  - `a_t = e^{-w} ⊙ a_{t-1} + e^{k_t} ⊙ v_t`
  - `b_t = e^{-w} ⊙ b_{t-1} + e^{k_t}`

Read past state with the current token added at bonus `u`; emit; then fold the current token into the state with its ordinary weight `e^{k_t}` so it decays for future steps. A running-max (log-sum-exp) rescaling of `a, b` by a shared exponent `p` keeps the exponentials from overflowing; inference state per layer is `{x, a, b, p}`, each a `d`-vector — `O(d)`, constant in context.

**Token shift.** Every projection sees a per-channel blend of the current and previous token: `r_t = W_r(μ_r ⊙ x_t + (1-μ_r) ⊙ x_{t-1})` (and likewise `k, v`, and `r', k'` in channel-mixing). Implemented as a one-step sequence shift.

**Time-mixing block.** Token-shift → project `r, k, v` → `wkv` → gate `σ(r) ⊙ wkv` → output projection: `o_t = W_o(σ(r_t) ⊙ wkv_t)`.

**Channel-mixing block** (position-wise FFN analog). Token-shift → expand `k'` → squared-ReLU → project → receptance gate: `o'_t = σ(r'_t) ⊙ (W'_v · max(k'_t, 0)²)`.

**Block / model.** Pre-LN residuals `x ← x + TimeMix(LN(x))`, `x ← x + ChannelMix(LN(x))`; embedding (tiny `U(±10⁻⁴)` init + extra LayerNorm), `L` stacked blocks, final LayerNorm + linear head, cross-entropy. Gradients are bounded by construction: because `wkv` is a normalized average, its Jacobians are bounded expectations/covariances rather than a runaway product, so the stack trains deep and scales to billions of parameters.

**Two modes.** *Time-parallel* training: the `W_{r,k,v,o}` projections are per-token GEMMs (`O(BTd²)`, fully parallel); only the cheap `O(BTd)` `wkv` scan is sequential, written as a custom CUDA kernel. *Time-sequential* inference: the recurrence above, `O(d)` memory per layer, `O(Td)` total.

## Code

```python
import torch, torch.nn as nn

def wkv(time_decay, time_first, k, v):
    # time_decay in log-space: effective per-step decay = exp(-exp(time_decay)); time_first = bonus u
    B, T, C = k.shape
    w, u = torch.exp(time_decay), time_first
    y = torch.empty_like(v)
    a = torch.zeros(B, C, device=k.device)            # numerator state
    b = torch.zeros(B, C, device=k.device)            # denominator state
    p = torch.full((B, C), -1e38, device=k.device)    # running max exponent
    for t in range(T):
        kt, vt = k[:, t], v[:, t]
        q  = torch.maximum(p, u + kt)                  # output: past + current(bonus u)
        e1, e2 = torch.exp(p - q), torch.exp(u + kt - q)
        y[:, t] = (e1 * a + e2 * vt) / (e1 * b + e2)
        q2 = torch.maximum(p - w, kt)                  # advance state (current decays for future)
        e1, e2 = torch.exp(p - w - q2), torch.exp(kt - q2)
        a, b, p = e1 * a + e2 * vt, e1 * b + e2, q2
    return y

def token_shift(x):
    return nn.functional.pad(x, (0, 0, 1, -1))          # x_{t-1}

class TimeMix(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mix_r = nn.Parameter(torch.ones(1, 1, d))
        self.mix_k = nn.Parameter(torch.ones(1, 1, d))
        self.mix_v = nn.Parameter(torch.ones(1, 1, d))
        self.time_decay = nn.Parameter(torch.zeros(d))  # W
        self.time_first = nn.Parameter(torch.zeros(d))  # u
        self.key   = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.receptance = nn.Linear(d, d, bias=False)
        self.output = nn.Linear(d, d, bias=False)
    def forward(self, x):
        xx = token_shift(x)
        xk = x * self.mix_k + xx * (1 - self.mix_k)
        xv = x * self.mix_v + xx * (1 - self.mix_v)
        xr = x * self.mix_r + xx * (1 - self.mix_r)
        k, v, r = self.key(xk), self.value(xv), self.receptance(xr)
        rwkv = torch.sigmoid(r) * wkv(self.time_decay, self.time_first, k, v)
        return self.output(rwkv)

class ChannelMix(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mix_k = nn.Parameter(torch.ones(1, 1, d))
        self.mix_r = nn.Parameter(torch.ones(1, 1, d))
        self.key   = nn.Linear(d, 4 * d, bias=False)
        self.value = nn.Linear(4 * d, d, bias=False)
        self.receptance = nn.Linear(d, d, bias=False)
    def forward(self, x):
        xx = token_shift(x)
        xk = x * self.mix_k + xx * (1 - self.mix_k)
        xr = x * self.mix_r + xx * (1 - self.mix_r)
        k = torch.square(torch.relu(self.key(xk)))      # squared ReLU
        return torch.sigmoid(self.receptance(xr)) * self.value(k)

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att, self.ffn = TimeMix(d), ChannelMix(d)
    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class RWKV(nn.Module):
    def __init__(self, vocab, d, n_layers):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.ln_emb = nn.LayerNorm(d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(n_layers))
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        nn.init.uniform_(self.emb.weight, -1e-4, 1e-4)
    def forward(self, idx):
        x = self.ln_emb(self.emb(idx))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_out(x))
```

At inference the same `wkv` runs one token at a time carrying `(a, b, p)` forward — `O(d)` state, `O(1)` work per token, no growing KV cache.
