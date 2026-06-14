# Hash embeddings (and the bigram-hash specialization), distilled

A **hash embedding** represents each token (or `n`-gram) by a small learnable combination of vectors
drawn from a *shared pool*, where the pool rows are selected by `k` independent hash functions and
combined with `k` trainable per-token *importance weights*. It is an interpolation between a standard
embedding table and the feature-hashing trick — both are special cases — that keeps memory bounded and
independent of the true feature-space size while letting the model repair the collisions hashing
introduces. The **bigram-hash** embedding is its specialization for a causal language model: the
(previous, current) token pair is hashed into a fixed table and its vector is injected, gated by a
learnable per-layer scalar, into the residual stream.

## Problem it solves

A standard embedding table costs `K · d` parameters and grows linearly with the vocabulary `K`; under
Zipf most rows are rarely updated, a dictionary must be fixed before training, and an `n`-gram feature
space of size `K^n` (e.g. ~`2.5 × 10^9` ordered bigrams for `K ≈ 50k`) is simply unstorable. The goal:
a bounded-memory, dictionary-free, end-to-end-trainable token representation whose cost does not blow
up with the feature space, and that does not destroy useful distinctions when features share storage.

## Key idea

Feature hashing maps a feature into one of `B` buckets, each with an embedding row: memory `B · d`, no
dictionary, but a single fixed hash forces collisions among distinct features and offers no gradient to
escape them (the hash codomain is discrete). Hash embeddings fix this with two moves:

1. **`k` independent hashes for collision resistance.** Describing a feature by the tuple
   `(h_1(w), …, h_k(w))` behaves like one hash with range `B^k`, so the per-feature collision
   probability collapses (birthday math: `|T|=10^8, B=10^6` drops `p_col` from `≈1` at `k=1` to
   `≈10^{-4}` at `k=2`) at a cost only linear in `k`, not `B^k`.
2. **Trainable per-feature importance weights for collision repair.** Pull the `k` selected component
   vectors and combine them by a learned weight vector `p_w ∈ R^k`. The weights are continuous (nothing
   discrete to differentiate), so features that collide on one hash are pulled apart through the others;
   `p_w → 0` for unimportant features both mutes them and removes them from the effective collision set
   (implicit feature selection / regularization).

## Final construction

For token `w`, with `k` hash functions `H_i(w) = E[ D_{2,i}(D_1(w)) ]` selecting rows of a shared
`B × d` pool `E`, and importance weights `p_w` (rows of a `K × k` matrix `P`):

```
c_w = (H_1(w), …, H_k(w))            # k component vectors, each d-dim
p_w = (p_w^1, …, p_w^k) ∈ R^k        # trainable importance weights
ê_w = sum_{i=1}^{k} p_w^i  H_i(w)    # d-dim hash embedding
e_w = ê_w  ⊕  p_w                    # optional: concatenate the weights
```

- `D_1: T → {1,…,K}` is a token-to-id map (dictionary, or a hash if the space is too large/dynamic);
  each `D_{2,i}: {1,…,K} → {1,…,B}` is an independent id-to-bucket hash. Same id indexes `P`.
- **Parameter count `B · d + K · k`** vs `K · d` for a standard table. Defaults: `k = 2`, `K > 10 · B`.
- **Special cases.** `k = 1` with all `p_w^1 = 1` ⇒ the hashing trick. Additionally `B = |T|` with `h_1`
  the identity ⇒ a standard embedding table.

**Collision theory (birthday).** For a hash with range `K`, the probability a token collides with at
least one other is `p_col = 1 - (1 - 1/K)^{|T|-1} ≈ 1 - exp(-|T|/K)`, expected `C_tot = |T| · p_col`
tokens in some collision. Using `k` component hashes approximates a joint range `B^k`, slashing
`p_col`. Importance weights make the effective collision set the important subset, so the same birthday
count is applied to `|T_imp|` rather than all of `|T|`.

**Unbiased collisions (the sign trick, from feature hashing).** With an independent sign hash
`xi: T → {±1}`, the signed map `phi_i^{(h,xi)}(x) = sum_{j:h(j)=i} xi(j) x_j` has
`E[⟨x,x'⟩_phi] = ⟨x,x'⟩` (the diagonal `j=l` gives `xi(j)^2=1` ⇒ the true product; off-diagonal terms
have `E[xi(j)xi(l)]=0`), with variance `O(1/B)`. Random signs make colliding contributions cancel in
expectation rather than coherently inflate a shared row.

## Bigram-hash specialization (causal LM)

The feature is the ordered pair (previous token, current token); its space `K^2` is unstorable, so it
is the hashing trick's home turf.

- **Table size** `B = 5 · vocab_size`: a small constant multiple of the token table; occurring bigrams
  are Zipfian and far fewer than `K^2`, so frequent-bigram collisions stay rare.
- **Hash** (cheap, GPU-friendly, order-sensitive): `index = (r1·curr XOR r2·prev) mod (B-1)` with large
  fixed multipliers `r1 = 36313`, `r2 = 27191` spreading ids across int32 before the XOR; position 0 has
  no previous token ⇒ reserved bucket `B-1`.
- **Harness specialization of the importance weights:** the full per-feature `K × k` matrix is collapsed
  to one learnable scalar per layer over a **zero-initialized** table. The augmentation starts as an exact
  no-op and training grows the bigram signal only where, and at the depths where, gradients ask for it.
- The output projection stays **tied** to `wte`; the bigram signal is an additive input/value-side
  intervention, so it remains a pure embedding-level change.
- The compact implementation is unsigned; the signed-hash identity above explains the unbiased variant,
  but no sign table is instantiated below.

## Working code

Fills the `forward` / `get_value_embed` slots of the fixed `TokenEmbedding` harness:

```python
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token + position embedding augmented with a hashed bigram embedding.

    The (previous, current) token pair is hashed into a B = 5*vocab_size table; the
    looked-up vector is injected into every layer, gated by a learnable per-layer
    scalar. The bigram table is zero-initialised so the
    augmentation begins as a no-op.
    """

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer

        # bigram component pool: 5x vocab buckets for collision reduction
        self.bigram_vocab_size = config.vocab_size * 5
        self.bigram_embed = nn.Embedding(self.bigram_vocab_size, config.n_embd)
        nn.init.zeros_(self.bigram_embed.weight)             # start as a no-op residual

        # per-layer gate on the bigram injection
        self.bigram_lambdas = nn.Parameter(torch.full((config.n_layer,), 0.1))
        self._cached_bigram = None

    def _bigram_hash(self, idx):
        # index = (r1*curr XOR r2*prev) mod (B-1); position 0 -> reserved bucket B-1
        rand_int_1 = 36313
        rand_int_2 = 27191
        mod = self.bigram_vocab_size - 1
        x = idx.to(torch.int32)
        out = torch.zeros_like(x)
        out[:, 0] = mod                                       # "no previous token" slot
        out[:, 1:] = torch.bitwise_xor(
            rand_int_1 * x[:, 1:],                            # r1 * current token
            rand_int_2 * x[:, :-1]                            # r2 * previous token
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
        # inject the bigram signal at every layer, gated by its learnable weight
        if self._cached_bigram is None or layer_idx >= self.n_layer:
            return None
        return self.bigram_lambdas[layer_idx] * self._cached_bigram

    def get_lm_head_weight(self):
        return self.wte.weight                                # tied output projection

    def get_num_pos_params(self):
        return self.wpe.weight.numel()                        # excluded from the param count
```

## Relation to prior methods

- **Standard embedding table** = hash embedding with `B = |T|`, identity hash, `k = 1`.
- **Feature hashing / hashing trick** (Weinberger et al. 2009; bigram-hashed in fastText, Joulin et al.
  2016) = hash embedding with `k = 1` and frozen unit weights. Hash embeddings add `k` hashes (the
  `B^k` collision cliff) and trainable importance weights (collision repair + implicit feature
  selection). The signed kernel can be used for unbiased collisions in variants with a sign table; the
  compact code above does not instantiate that table.
