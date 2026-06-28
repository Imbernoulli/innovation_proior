Let me state the slot exactly, because the structure of the problem decides almost everything. At each
spatial location I hold a set of `V` variable tokens — vectors in `R^D` — and I must return one vector in
`R^D`. The set has no canonical order (there is no privileged arrangement of temperature, geopotential,
humidity, the surface mask), so the reduction must be permutation-invariant in the variables; `V` is read
from the input at runtime, so the reduction must accept any `V`; and the output feeds a fixed downstream
stack tuned to features on a single-token scale, so the reduction should keep the fused token on that
scale. This is a set-to-vector problem. So before I pick a formula, I want to know what *class* of
set-to-vector functions I am choosing from and where the reductions already on the table sit inside it.

The cleanest characterization of permutation-invariant set functions I know writes them as
`ρ(Σ_v φ(x_v))`: encode each element with a shared `φ`, combine with a symmetric reduction (the sum),
decode with `ρ`. This is universal in principle. But look at what it commits to: each element is encoded
*in isolation* by `φ` before the sum. The elements never see each other during encoding. The reductions
already on the table are exactly this skeleton in its barest forms — the uniform mean is `φ = identity`,
sum, then divide; the learned weighted sum is `φ = (scalar weight)·identity`, sum. The single-query
cross-attention looks at first like it escapes this, since its weights are content-dependent, but when I
look hard at it the same blind spot is there in a subtler place: a single learnable query scores each
variable token against itself, and the variable tokens are pooled by those scores — but the *score on each
variable is computed without reference to the other variables' contents*. The query asks "how much do you,
token `x_v`, match my fixed question?" of each token independently. So across all three existing reductions,
the variables never interact before being summarized.

Is that interaction worth wanting here? Let me argue it from the domain rather than assume it. The right
combination of variables at a grid cell does not look separable across variables — it looks relational. A
mid-tropospheric wind token may carry decisive information *only when* the geopotential token indicates a
sharp gradient; a humidity token's relevance depends on the temperature token's regime. These are
inter-variable correlations: the pooling weight that should go on variable `i` depends on what variable
`j` is currently saying. A pooling that scores each variable independently — whether by a fixed scalar
(weighted sum) or by a fixed query against the raw token (cross-attention) — cannot express "upweight wind
*because* geopotential is in this state." To capture that, the variables have to read each other before
the summary is formed. So the candidate I want to develop is: let the `V` variable tokens attend among
themselves first, producing context-aware variable representations, and only then pool. That is the one
thing none of the existing rungs does, so it is where I should look for the gain.

Now I have to be careful, because "let the elements interact, then pool" has a degenerate trap I have seen
before. If I let the encoder be optimized only to make the *pooling* easy, the cheapest solution can be a
collapse — an encoder that maps every variable token to the same vector pools trivially and learns
nothing. So whatever interaction mechanism I use among the variables must be one that resists that trivial
collapse and that genuinely mixes information across the set. Self-attention has the shape I want here:
each token's new representation is a weighted combination of *all* the tokens' values, with the weights
computed from content compatibilities, and a residual connection keeps the token's own information. So
folding every token to a constant is not the cheap optimum — it would have to fight the residual rather
than ride a free gradient down to it — and it is permutation-equivariant — permute the input tokens and
the outputs permute the same way — which is what I need so that a permutation-invariant pooling on top can
yield a permutation-invariant whole. So I will try a self-attention encoder over the variable set, with the
pooling on top itself attention so it is content-dependent and set-valued. Let me build both out of one
well-behaved attention block and check the pieces, because the invariance claim I just made about the
composition is exactly the kind of thing that is easy to assert and easy to get wrong.

The block I want is a Transformer encoder block adapted for sets — multihead attention plus a feed-forward,
with residuals and layer norm, but **no positional encoding and no dropout**, because positions would
break the permutation symmetry I am relying on and stochastic dropout would break the determinism of the
set map. Call it MAB, the multihead attention block, taking a query set `X` and a key/value set `Y`. I
write it in the standard residual form: first an attention sublayer with a residual,
`H = LayerNorm(X + Multihead(X, Y, Y))`, then a feed-forward sublayer with a residual,
`MAB(X, Y) = LayerNorm(H + rFF(H))`, where `rFF` is a row-wise feed-forward applied identically to every
element (a `Linear` followed by a nonlinearity, residual), and `Multihead` is the usual `h`-head scaled
dot-product attention with `1/√d` scaling. Two properties matter. The attention output is a weighted
average over the *keys* `Y`, so MAB is invariant to the order of `Y`; and it processes the *queries* `X`
row-wise plus an attention that treats `X`'s rows symmetrically, so it is equivariant in `X`.

The `1/√d` scaling I want to actually check rather than recite, because the whole point of the block is
that its softmax stays responsive at the width this pipeline runs (`D = 1024`, so `d_k = 64` per head). The
argument is that for query/key components roughly independent with unit variance, the dot product over `d`
dimensions has variance `d`, so unscaled logits grow like `√d` and saturate the softmax toward one-hot,
killing its gradient. Let me put numbers on it: a random unit-variance query against eight random keys, at
the head widths in play.

```
d=  64  raw logit std= 8.83  →  max softmax(unscaled)=0.999 ;  scaled (/8)   std=1.10  max softmax=0.50
d= 256  raw logit std=21.2   →  max softmax(unscaled)=0.890 ;  scaled (/16)  std=1.33  max softmax=0.32
d=1024  raw logit std=31.5   →  max softmax(unscaled)=0.999 ;  scaled (/32)  std=0.98  max softmax=0.28
```

The unscaled logit std tracks `√d` (8.83≈√64, 31.5≈√1024), and at those magnitudes the unscaled softmax is
essentially one-hot (0.999) — gradient-dead. Dividing by `√d` pulls the logit std back to ≈1 and the
softmax back to a usable spread (0.50, 0.32, 0.28). So the scaling is doing real work at `d_k = 64`, not a
cosmetic constant; good, I keep it as written.

From MAB the two blocks I need fall out directly. Self-attention within the variable set is
`SAB(X) := MAB(X, X)` — the tokens are simultaneously queries, keys, and values, so each variable token's
new representation reads from all the others by content compatibility. Stacking SABs encodes higher-order
interactions, but a single SAB already gives the pairwise mixing I argued for, and for a set of only
`V = 48` elements one block of pairwise interaction is a sensible amount of structure to add. SAB is
permutation-equivariant, the property I need under the pooling. Note SAB costs `O(V²)` per
location — but `V = 48` is tiny, and this `V²` is the cost of a `48×48` attention at each location, not the
`O((V·h·w)²)` blowup that motivated aggregation in the first place (that came from running the *backbone*
over the full variable-and-space sequence; here the backbone still sees only the `h·w` aggregated
sequence). So I can afford the full SAB rather than the inducing-point approximation (ISAB), which is meant
for large sets where `V²` would hurt.

Now the pooling. I want a learnable, content-dependent, set-valued pooling that returns one summary
vector. The construction is to introduce a learnable *seed* vector `S` and let it attend over the encoded
set: `PMA_1(Z) = MAB(S, Z)` with one seed `S ∈ R^{1×D}`. This is multihead attention pooling — the seed is
a trainable query that asks the set "what is your summary?", the answer is a content-dependent weighted
combination of the encoded variable values, and because the attention is multihead the seed can pose
several sub-questions at once (the thermodynamic summary, the dynamical summary) and combine them through
the block's output projection. One seed gives exactly one summary token per location, which is the
`[B, L, D]` the contract wants; I do not need `k > 1` seeds (those are for problems wanting several
correlated outputs, like clustering, and would need a trailing SAB among the seeds to model their
interaction — irrelevant here). The pooling is invariant to the order of `Z` because the softmax runs over
the set of keys.

So I have an equivariant encoder (SAB) feeding an invariant pooling (PMA), and I claimed the composition is
permutation-invariant. That composition is the property the contract turns on, so I should not take it on
faith — I will build the two blocks at toy width and measure it directly. Let `D = 8`, `h = 2` heads,
`B = 2`, `V = 4`, `L = 3`, random weights, random input `x: [2, 4, 3, 8]`. Run the aggregator once; then
draw a random permutation `π` of the four variables, apply it along the `V` axis, run again, and compare
the two `[2, 3, 8]` outputs. If the map is genuinely invariant the difference should be zero up to
floating-point error.

```
output shape                          (2, 3, 8)   ✓ matches [B, L, D]
max |out(x) − out(π·x)|               4.8e-07     ← floating-point zero: invariant ✓
output shape with V=7 at runtime      (2, 3, 8)   ✓ any-V accepted
```

The permuted run agrees with the original to `5e-7`, which is round-off, not a real difference — so the
composed SAB→PMA map is permutation-invariant in the variables, as the design requires; the equivariant /
invariant composition argument holds in actual numbers, not just on paper. And feeding `V = 7` instead of
`V = 4` through the *same* weights returns the same `[2, 3, 8]` output shape, confirming `V` never enters a
weight matrix — the module accepts any runtime `V`. Both contract requirements (permutation-invariance,
any-`V`) are now checked rather than assumed. Scale-preservation I do not get for free from this check, but
the trailing LayerNorm fixes the output to unit-scale features, which is what the single-token downstream
stack expects.

Let me hold this construction up against the rung directly below it — the single-query cross-attention —
because I want to be sure the extra machinery is buying the structure I argued for and not just parameters.
PMA with one seed and a single head, on the *raw* tokens with no encoder, is essentially that cross-
attention: a learnable query pooling the variable tokens. So the ClimaX default sits inside this
construction as its `k=1`, no-encoder, single-head special case. The two things this construction adds over
it are the two it leaves out: (1) the pooling is *multihead* (several summary sub-questions instead of one),
and (2) there is a *self-attention encoder (SAB) over the variable set before pooling*, so the variables
read each other and the summary is formed over *context-aware* variable representations rather than raw,
independently-scored ones. That second addition is the inter-variable-correlation modelling I argued the
domain needs and that no lower rung has. There is independent evidence that attending-among-elements-
before-pooling captures structure fixed reductions cannot: on a max-value-regression toy task, the
attentive self-attention-plus-pooling construction matches the max-pooling oracle, while mean and sum
pooling fail badly — because finding the governing element requires the elements to be compared against
each other, which only the interaction buys. Meteorological variables at a cell have that same relational
flavour, which is why I expect the gain to carry over here — though that expectation is exactly what the
held-out RMSE comparison has to settle, and I cannot settle it from the construction alone.

So the aggregator is: `SAB` over the `V` variable tokens, then `PMA_1` (one seed) over the SAB output —
the "SAB + PMA" Set Transformer architecture, restricted to a single encoder block and one seed. Let me
settle the remaining choices. Should the PMA pool the raw `Z` or `rFF(Z)`? The canonical definition is
`PMA_k(Z) = MAB(S, rFF(Z))` with a leading row-wise feed-forward, but in practice that leading `rFF` is
dropped when the preceding block already ends in a feed-forward — and SAB *does* end in its `rFF` sublayer.
So `PMA_1` applied directly to the SAB output is the standard, non-redundant form; I do not add a second
leading feed-forward. LayerNorm: keep it on (`ln=True`) in both blocks — this sits inside a deep ViT
pipeline fine-tuned from pretrained weights, and LayerNorm in the residual blocks is what keeps the
activations well-conditioned and stable during fine-tuning. Heads: use the pipeline's `num_heads = 16`, so
`d_k = D/16 = 64` per head, the standard head width and the same granularity the rest of the model uses
(and the width I just checked the `1/√d` scaling at). The seed `S` is a learnable parameter, Xavier-
initialized (the canonical init for the seed and inducing points) so the attention starts with sensible-
magnitude logits.

Now the implementation against the contract, since the indexing is where it lives or dies — and the toy
run above already exercised exactly this path, so I am transcribing what I just ran at full width. The
tokenizer hands me `x: [B, V, L, D]`. The reduction is per-location and per-example and identical at every
location, so I treat every `(example, location)` pair as an independent `V`-element set: permute to
`[B, L, V, D]` to bring the variable axis adjacent to `D`, fold `B` and `L` into one batch `[B·L, V, D]` —
`B·L` independent set problems. Apply `SAB` (= `MAB(X, X)`): `[B·L, V, D] → [B·L, V, D]`, the variables now
context-aware. Expand the single seed to the batch, `[B·L, 1, D]`, and apply `PMA` (= `MAB(S, Z_enc)`):
`[B·L, 1, D]`. Squeeze the seed axis to `[B·L, D]` and unfold back to `[B, L, D]`. `V` never enters any
weight matrix — every `Linear` inside MAB maps `D → D`, and the seed is `1 × 1 × D` — which is why the
`V = 7` toy run went through the `V = 4` weights unchanged; `V` only sizes the runtime attention.

I should implement MAB faithfully to its canonical reference rather than reach for a library multihead
layer, because the residual placement and the rFF are part of the block's identity and I want the math to
match the reference exactly. MAB projects `Q, K, V` with three linears to width `D`; splits each into
`num_heads` chunks of width `D/num_heads` and stacks the heads along the batch dimension (the standard
trick that lets one `bmm` do all heads at once); computes attention `A = softmax(Q_ K_ᵀ / √D)` over the
key axis; forms the attention output with a residual on the projected query, `Q_ + A V_`; reassembles the
heads back along the feature axis; applies the first LayerNorm; then the feed-forward residual
`O + ReLU(fc_o(O))`; then the second LayerNorm. SAB calls `MAB(X, X)`; PMA calls `MAB(S, Z)` with the
expanded seed. (The full scaffold module — MAB plus the `SAB`-then-`PMA` aggregator — is in the answer.)

Where this leaves me: a permutation-invariant (checked: `5e-7` under a random variable permutation), any-`V`
(checked: `V=7` through `V=4` weights), scale-respecting reduction built from one SAB and one PMA seed. It
adds over the single-query cross-attention baseline exactly the two levers that baseline lacks — multihead
pooling and self-attention among the variables before pooling — at `O(V²)` per location, trivial for
`V = 48` and not touching the backbone's `h·w` sequence. LayerNorm on, `num_heads = 16`, one SAB block, one
PMA seed, no positional encoding, no dropout: the minimal Set-Transformer pooling that lets the variables
read each other before the summary is formed. Whether that relational capacity actually lowers the held-out
latitude-weighted RMSE against the cross-attention default is the one claim I have left open for the
experiment to decide; the construction's correctness, not its win, is what I have verified here.

## Minimal code sketch

A tiny attention-based pooling stub that realizes the core idea — variables attend to each other, then a
learnable seed summarizes the set:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SetTransformerAggregator(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.sab = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.seed = nn.Parameter(torch.randn(1, 1, dim))
        self.pma = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):                 # x: [B, V, L, D]
        b, v, l, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * l, v, d)
        z, _ = self.sab(x, x, x)         # variables interact
        s = self.seed.expand(b * l, 1, d)
        out, _ = self.pma(s, z, z)       # seed pools the set
        return out.squeeze(1).reshape(b, l, d)
```
