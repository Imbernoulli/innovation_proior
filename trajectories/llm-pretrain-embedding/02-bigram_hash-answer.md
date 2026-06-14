**Problem (from step 1).** Untying decoupled the input/output matrices but added *no new signal* — every
token still enters as the same vector regardless of context (val_loss floor 2.3058). The cheapest piece
of information the token table structurally cannot carry is local order: the (previous, current) token
pair. A literal bigram table is unstorable (`vocab_size² ≈ 2.5×10^9` rows).

**Key idea.** Inject a *hashed* bigram embedding. Feature hashing maps each ordered pair into one of `B`
buckets via a cheap GPU hash, so memory is `B · n_embd` regardless of how many pairs exist (no
dictionary). The looked-up bigram vector is added, gated per layer, into the residual stream before each
block — giving every layer access to left-context the unigram table omits.

**Why it works / why it is safe.** Bigrams are heavily Zipfian, so a table of `B = 5 · vocab_size` keeps
*frequent*-bigram collisions rare by the birthday bound (`p_col ≈ 1 − exp(−T/B)` on the occurring pairs).
The table is **zero-initialized** and each layer's injection is gated by a small learnable scalar (init
0.1), so the augmentation starts as an exact no-op — it cannot harm the tuned base — and training grows
the bigram signal only at the depths where gradients ask for it.

**Harness specialization.** The full hash-embedding construction (k independent hashes + a
`vocab_size × k` per-feature importance matrix) collapses onto this harness's single exposed degree of
freedom: one hashed row per position, one learnable gate per layer. Hash:
`index = (r1·curr XOR r2·prev) mod (B−1)`, `r1 = 36313`, `r2 = 27191` (large multipliers spread ids
across int32 before the XOR; order-sensitive); position 0 (no previous token) → reserved bucket `B−1`.
The injection lands on the **residual stream** before each block (the only per-layer hook the harness
exposes — `get_value_embed(i)` is added to `x`, not into the value path). The output head stays **tied**
to `wte`; the bigram is a pure additive input-side signal. Unsigned table (no sign lookup).

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 115-140) — step 2: bigram hash
class TokenEmbedding(nn.Module):
    """Token + position + bigram hash embedding."""
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        # Bigram hash embedding: 5x vocab for hash collision reduction
        self.bigram_vocab_size = config.vocab_size * 5
        self.bigram_embed = nn.Embedding(self.bigram_vocab_size, config.n_embd)
        nn.init.zeros_(self.bigram_embed.weight)
        self.n_layer = config.n_layer
        # Per-layer learnable scaling for bigram embedding injection
        self.bigram_lambdas = nn.Parameter(torch.full((config.n_layer,), 0.1))
        self._cached_bigram = None

    def _bigram_hash(self, idx):
        """Compute bigram hash indices from consecutive token pairs."""
        rand_int_1 = 36313
        rand_int_2 = 27191
        mod = self.bigram_vocab_size - 1
        x = idx.to(torch.int32)
        out = torch.zeros_like(x)
        # Position 0: no previous token, use reserved index
        out[:, 0] = mod
        # Positions 1+: XOR hash of (current, previous) token pair
        out[:, 1:] = torch.bitwise_xor(
            rand_int_1 * x[:, 1:],
            rand_int_2 * x[:, :-1]
        ) % mod
        return out.long()

    def forward(self, idx):
        b, t = idx.size()
        tok_emb = self.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)
        self._cached_bigram = self.bigram_embed(self._bigram_hash(idx))
        return self.drop(tok_emb + pos_emb)

    def get_value_embed(self, layer_idx):
        """Inject bigram embedding at every layer with learnable scaling."""
        if self._cached_bigram is None or layer_idx >= self.n_layer:
            return None
        return self.bigram_lambdas[layer_idx] * self._cached_bigram

    def get_lm_head_weight(self):
        return self.wte.weight

    def get_num_pos_params(self):
        return self.wpe.weight.numel()
```

The surrounding `GPT` is unchanged: before block `i` it adds `get_value_embed(i)` to the residual stream
when non-`None`, wires the (tied) head from `get_lm_head_weight()`, and trains with cross-entropy.
