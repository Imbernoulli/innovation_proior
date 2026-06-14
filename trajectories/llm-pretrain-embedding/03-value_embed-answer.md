**Problem (from step 2).** The bigram rung beat untied (2.3058 → 2.2877) and lifted local-cue downstream
(ARC-Easy 54.80 → 56.40), but it injected *one* noisy hashed feature gated by a single scalar per layer —
WikiText-2 barely moved (44.97) and LAMBADA slightly regressed (71.11 → 71.43). The diagnosis: the
*content* matters less than how richly the per-layer injection is learned and placed by depth.

**Key idea.** Deep transformers over-smooth: self-attention is one gradient step on a smoothing functional
whose fixed point is uniform token representations, so localized token-level information from the initial
embedding `H_0` is washed out in deep layers. The variational repair re-supplies an un-smoothed reference
(the first layer's value `V_1 = H_0 W^V_1`) to deep layers. Since `V_1` is functionally just a
token-indexed lookup, replace it with **dedicated, freely-learned value-embedding tables** injected at a
few selected layers — a richer, depth-aware per-layer injection than one hashed bigram cue.

**Why it works / vs bigram.** A clean per-token learned value has no value-state drain (the
large-norm sink-token pathology is deep-layer-learned), so injecting it at the deep layers weakens the
attention-sink loop and lets each deep layer learn a smaller correction. Targeting the *last three layers*
directly addresses where over-smoothing hurts long-range completion — the place the bigram cue could not
help (LAMBADA). Five independent learned tables at distinct depths is the "richer injection" the bigram
diagnosis called for.

**Harness reality (same-name vs paper).** The canonical value-residual mixes `V_1` into the *value path
before attention* so it rides the layer's attention matrix. This harness exposes only
`get_value_embed(layer_idx)`, whose return is added to the **residual stream** before the block
(`x = x + ve`) — not the value path. So this is a residual-stream injection motivated by the
value-residual mechanism; it also feeds `Q`/`K`, so the gate + small init keep early perturbation benign.

**Hyperparameters.** `n_ve = 5` tables held in one partitioned `nn.Embedding(vocab_size·5, n_embd)`,
small init (`std 0.01`); per-table learnable gate `ve_lambda` init 0.5; injection layers `[1, 2,
n_layer−3, n_layer−2, n_layer−1]`; table `i` read via `offset_idx = idx + i·vocab_size`. Output head stays
tied to `wte`; token+position stream unchanged.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 115-140) — step 3: value embeddings
class TokenEmbedding(nn.Module):
    """Token + position embedding with value embeddings for selected layers."""
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        # Value embeddings: 5 tables injected into selected layers (like modded-nanogpt)
        self.n_ve = 5
        self.vte = nn.Embedding(config.vocab_size * self.n_ve, config.n_embd)
        nn.init.normal_(self.vte.weight, mean=0.0, std=0.01)
        # Per-VE learnable blending coefficient (lambda)
        self.ve_lambda = nn.Parameter(torch.full((self.n_ve,), 0.5))
        # Injection layers: layer 1, 2, and last 3 layers
        self._ve_layers = None
        self._cached_ve = None

    def forward(self, idx):
        b, t = idx.size()
        tok_emb = self.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)
        # Compute injection layer indices: layer 1, 2, and last 3 layers
        if self._ve_layers is None:
            self._ve_layers = [1, 2, self.n_layer - 3, self.n_layer - 2, self.n_layer - 1]
        # Cache per-VE value embeddings (5 separate lookups into partitioned table)
        vs = self.vocab_size
        self._cached_ve = {}
        for i, layer_idx in enumerate(self._ve_layers):
            offset_idx = idx + i * vs  # offset into partition i
            self._cached_ve[layer_idx] = self.vte(offset_idx)
        return self.drop(tok_emb + pos_emb)

    def get_value_embed(self, layer_idx):
        """Get value embedding residual for a given layer, or None."""
        if self._cached_ve is None or layer_idx not in self._cached_ve:
            return None
        ve_idx = self._ve_layers.index(layer_idx)
        lamb = self.ve_lambda[ve_idx]
        return lamb * self._cached_ve[layer_idx]

    def get_lm_head_weight(self):
        return self.wte.weight

    def get_num_pos_params(self):
        return self.wpe.weight.numel()
```

The surrounding `GPT` is unchanged: before each block it adds `get_value_embed(i)` to the residual stream
when non-`None`, wires the (tied) head from `get_lm_head_weight()`, and trains with cross-entropy.
