# Value Residual Learning (ResFormer / value embeddings), distilled

Value Residual Learning gives every deep Transformer layer direct, un-smoothed access to the
initial token-level information by adding a residual connection on the **value** path: before
attention, each layer mixes in the first layer's value `V_1` (a linear map of the token
embedding), and runs the ordinary attention on the mixed value so `V_1` is aggregated by the
*same* learned attention matrix. The architecture is **ResFormer**; the practical "value
embedding" specialization replaces `V_1` with a dedicated, gated token-to-value lookup table
injected at a few selected layers.

## Problem it solves

Self-attention is a smoothing (low-pass) operation, so stacking attention layers drives token
representations toward uniformity (over-smoothing): in deep layers, sequence-level features
dominate and localized token-level information from the initial embedding is diluted. The plain
hidden residual carries the initial embedding forward but feeds all of Q, K, V, so re-injecting
whole hidden states perturbs the learned attention distribution. Needed: cheaply restore
un-smoothed early information to deep layers **without** disturbing their attention pattern.

## Key idea

A single self-attention update is one gradient-descent step minimizing the nonlocal smoothing
functional `J(u) = ½∬||u(x)−u(y)||² k(x,y) dx dy`, whose minimizer is a constant — over-smoothing
is its fixed point. The variational repair is to descend a *regularized* functional
`E = J(u) + (λ/2)∫||u−f||²dx` whose convex fidelity term anchors the output to an un-smoothed
reference `f`. Its gradient flow is `du/dt = -∇J(u) - λ(u-f)`, so an Euler step with
`u(0)=v` and `λ = λ̃/Δt` adds `+λ̃(f-v)`. With `f=V_1`, the direct output correction is
`Attn(Q_n,K_n,V_n) + λ(V_1 - V_n)`.

Value Residual Learning keeps the same source but moves it into the **value** path before
attention, and decouples the useful early-value carrier from the negative current-value term:

```
V_n' = λ_{n,1} · V_1 + λ_{n,2} · V_n,      V_1 = H_0 W^V_1,   V_n = H_{n-1} W^V_n
U_n  = Attn(Q_n, K_n, V_n')                 # same A_n; only the value changed
```

Three points carry the design:
- **Value path only.** `A_n` is computed from `Q_n, K_n` and is untouched, so the learned
  attention distribution (the abstract, sequence-level mixing depth bought) is preserved;
  changing `V` only changes *what* is aggregated. Adding the residual to Q, K, or the attention
  matrix would change the attention distribution itself.
- **Share the current attention matrix** (mix `V_1` in *before* attention) rather than adding a
  raw `V_1` term to the output: `V_1` is then read from the positions each query actually
  attends to, at no extra compute. Recomputing a separate cross-layer attention for `V_1` is
  more expensive; adding `V_1` with an identity (no attention) throws away the query-specific
  routing decision.
- **Positive value mix, not a signed difference.** The earlier `λ(V_1 − V_n)`
  output-correction both adds `V_1` and subtracts the layer's own value, making it fragile in
  `λ`; `λ_{n,1}V_1 + λ_{n,2}V_n` decouples the early-value carrier from any negative
  current-value term.

**Source = `V_1`, and concentrate in late layers.** `V_1 = H_0 W^V_1` is the least-smoothed,
least-redundant signal (`V_2` and later are reachable through the ordinary hidden residual).
Over-smoothing is worst deep, so learnable `λ` can give later layers more access to `V_1`, and
sparse variants can zero the `V_1` term outside selected late layers.

**Why it works (mechanism).** `V_1` has no "value-state drain" (the large-value-norm sink-token
pathology is learned in deep layers). Injecting drain-free `V_1` should weaken the
mutual-reinforcement loop between value-state drains and attention sinks, and it lets each deep
layer learn a smaller correction `ΔV` on top of a clean token-level value. The key diagnostic
distinction from a pure optimization shortcut is whether first-layer learning-rate boosts can
replicate the behavior; the mechanism predicts they should not.

## Variants

- **Identity-ResFormer:** `λ_{n,1}=λ_{n,2}=0.5`, fixed (no new parameters beyond `W^V`).
- **Learnable-ResFormer:** `λ` trainable, init 0.5, so each layer can choose its own mix.
- **Constant-ResFormer:** fixed choices such as `λ_{n,1}=2`, `λ_{n,2}=1`.
- **Sparse-ResFormer:** `λ_{n,1}=0` except selected layers, typically late layers.
- **Dense-ResFormer:** `V_n' = λ_{n,n}V_n + Σ_{i<n} λ_{n,i}V_i` (most general; dilutes `V_1`).
- **Value embeddings (modded-nanogpt specialization):** replace `V_1` with a dedicated
  token-to-value lookup table `E_v` with its own parameters — `V_n' = V_n + λ_n·E_v(token)` —
  gated by learnable `λ`, small init, placed at layers `1`, `2`, `N-3`, `N-2`, and `N-1` in the
  simplified `value_embed` edit. Same mechanism, with the injected signal decoupled from layer
  1's value projection.

## Code — canonical ResFormer form

```python
import torch
import torch.nn as nn


def attention(q, k, v, scale):
    A = torch.softmax((q @ k.transpose(-2, -1)) * scale + causal_mask(q, k), dim=-1)
    return A @ v                                   # V_1 rides the SAME attention matrix as V_n


class ResBlock(nn.Module):
    """Pre-norm decoder block with a value residual to the first layer's value V_1."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.Wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wk = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wv = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wo = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.scale = (config.n_embd // config.n_head) ** -0.5
        if layer_idx > 0:                              # init 0.5/0.5 -> Identity-ResFormer
            self.lam1 = nn.Parameter(torch.tensor(0.5))   # weight on V_1
            self.lam2 = nn.Parameter(torch.tensor(0.5))   # weight on this layer's value

    def forward(self, x, v_first):
        h = self.ln1(x)
        q, k, v = self.Wq(h), self.Wk(h), self.Wv(h)      # V_n = H_{n-1} W^V_n
        if self.layer_idx == 0:
            v_first = v                                    # cache V_1 = H_0 W^V_1
        else:
            v = self.lam1 * v_first + self.lam2 * v        # V_n' = lam1*V_1 + lam2*V_n
        u = attention(q, k, v, self.scale)                # ordinary attention on mixed value
        x = x + self.Wo(u)
        x = x + self.mlp(self.ln2(x))
        return x, v_first
```

## Code — value embeddings (modded-nanogpt form, modular embedding intervention)

```python
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token + position embedding, plus gated value embeddings for selected layers.
    get_value_embed(i) -> lambda_i * E_v_i(token) (or None); the attention block does
    v = v + that, BEFORE attention, so the residual rides the layer's attention matrix."""

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        # A few dedicated token -> value-space tables (one partitioned table), small init.
        self.n_ve = 5
        self.vte = nn.Embedding(config.vocab_size * self.n_ve, config.n_embd)
        nn.init.normal_(self.vte.weight, mean=0.0, std=0.01)
        self.ve_lambda = nn.Parameter(torch.full((self.n_ve,), 0.5))   # learnable gate
        self._ve_layers = None
        self._cached_ve = None

    def forward(self, idx):
        b, t = idx.size()
        tok_emb = self.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)
        if self._ve_layers is None:                       # first + last few layers
            self._ve_layers = [1, 2, self.n_layer - 3, self.n_layer - 2, self.n_layer - 1]
        vs = self.vocab_size
        self._cached_ve = {}
        for i, layer_idx in enumerate(self._ve_layers):
            offset_idx = idx + i * vs                      # partition i of the joint table
            self._cached_ve[layer_idx] = self.vte(offset_idx)
        return self.drop(tok_emb + pos_emb)

    def get_value_embed(self, layer_idx):
        if self._cached_ve is None or layer_idx not in self._cached_ve:
            return None
        ve_idx = self._ve_layers.index(layer_idx)
        return self.ve_lambda[ve_idx] * self._cached_ve[layer_idx]    # lambda * E_v(token)

    def get_lm_head_weight(self):
        return self.wte.weight

    def get_num_pos_params(self):
        return self.wpe.weight.numel()
```

Inside the attention block, the residual returned by `get_value_embed(layer_idx)` enters the value
path before attention:

```python
        q, k, v = self.Wq(h), self.Wk(h), self.Wv(h)
        if value_embed is not None:                       # lambda * E_v(token), or None
            v = v + value_embed                           # value residual rides A as v does
        u = attention(q, k, v, self.scale)
```
