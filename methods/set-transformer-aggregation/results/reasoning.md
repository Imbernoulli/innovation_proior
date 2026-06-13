Let me state the slot exactly, because the structure of the problem decides almost everything. At each
spatial location I hold a set of `V` variable tokens — vectors in `R^D` — and I must return one vector in
`R^D`. The set has no canonical order (there is no privileged arrangement of temperature, geopotential,
humidity, the surface mask), so the reduction must be permutation-invariant in the variables; `V` is read
from the input at runtime, so the reduction must accept any `V`; and the output feeds a fixed downstream
stack tuned to features on a single-token scale, so the reduction should keep the fused token on that
scale. This is a set-to-vector problem, and the right way to think about it is to ask what *class* of
set-to-vector functions I should draw from, then pick the most expressive member that respects the
constraints.

The cleanest characterization of permutation-invariant set functions I know writes them as
`ρ(Σ_v φ(x_v))`: encode each element with a shared `φ`, combine with a symmetric reduction (the sum),
decode with `ρ`. This is universal in principle. But look at what it commits to: each element is encoded
*in isolation* by `φ` before the sum. The elements never see each other during encoding. The reductions
already on the table are exactly this skeleton in its barest forms — the uniform mean is `φ = identity`,
sum, then divide; the learned weighted sum is `φ = (scalar weight)·identity`, sum. Even the single-query
cross-attention, when I look hard at it, has the same blind spot in a subtler place: a single learnable
query scores each variable token against itself, and the variable tokens are pooled by those scores — but
the *score on each variable is computed without reference to the other variables' contents*. The query
asks "how much do you, token `x_v`, match my fixed question?" of each token independently. So none of the
existing reductions lets the variables *interact* before being summarized.

Is that interaction worth wanting here? I think it is exactly the structure this domain has. The right
combination of variables at a grid cell is not separable across variables — it is relational. A
mid-tropospheric wind token may carry decisive information *only when* the geopotential token indicates a
sharp gradient; a humidity token's relevance depends on the temperature token's regime. These are
inter-variable correlations: the pooling weight that should go on variable `i` depends on what variable
`j` is currently saying. A pooling that scores each variable independently — whether by a fixed scalar
(weighted sum) or by a fixed query against the raw token (cross-attention) — cannot express "upweight wind
*because* geopotential is in this state." To capture that, the variables have to read each other before
the summary is formed. So the move I want is: let the `V` variable tokens attend among themselves first,
producing context-aware variable representations, and only then pool. That is the one structural lever the
existing rungs all leave unpulled.

Now I have to be careful, because "let the elements interact, then pool" has a degenerate trap I have seen
before. If I let the encoder be optimized only to make the *pooling* easy, the cheapest solution can be a
collapse — an encoder that maps every variable token to the same vector pools trivially and learns
nothing. So whatever interaction mechanism I use among the variables must be one that resists that trivial
collapse and that genuinely mixes information across the set. Self-attention is exactly that: each token's new
representation is a weighted combination of *all* the tokens' values, with the weights computed from
content compatibilities, and a residual connection keeps the token's own information. So folding every
token to a constant is not the cheap optimum — it would have to fight the residual rather than ride a free
gradient down to it — and it is permutation-equivariant — permute the input
tokens and the outputs permute the same way — which is precisely what I need so that a permutation-
invariant pooling on top yields a permutation-invariant whole. So the encoder over the variable set should
be self-attention, and the pooling on top should itself be attention so it is content-dependent and
set-valued. Let me build both out of one well-behaved attention block and check the pieces.

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
row-wise plus an attention that treats `X`'s rows symmetrically, so it is equivariant in `X`. The
`1/√d` is load-bearing: for query/key components roughly independent with unit variance, the dot product
over `d` dimensions has variance `d`, so unscaled logits grow like `√d`, saturate the softmax toward
one-hot, and kill its gradient; dividing by `√d` keeps the softmax responsive.

From MAB the two blocks I need fall out directly. Self-attention within the variable set is
`SAB(X) := MAB(X, X)` — the tokens are simultaneously queries, keys, and values, so each variable token's
new representation reads from all the others by content compatibility. Stacking SABs encodes higher-order
interactions, but a single SAB already gives the pairwise mixing I argued for, and for a set of only
`V = 48` elements one block of pairwise interaction is a sensible amount of structure to add. SAB is
permutation-equivariant, exactly the property I need under the pooling. Note SAB costs `O(V²)` per
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
interaction — irrelevant here). The pooling is permutation-invariant in `Z` (the softmax runs over the set
of keys), so composing the equivariant SAB encoder with the invariant PMA pooling yields a
permutation-invariant aggregator, as required.

Let me check this against the rung directly below it — the single-query cross-attention — because I want to
be sure I am adding the *right* thing and not just more machinery. PMA with one seed and a single head, on
the *raw* tokens with no encoder, is essentially that cross-attention: a learnable query pooling the
variable tokens. So the ClimaX default is the `k=1`, no-encoder, special case of this construction. The two
things I am adding over it are exactly the two levers it leaves unpulled: (1) the pooling is *multihead*
(several summary sub-questions instead of one), and (2) — the decisive one — there is a *self-attention
encoder (SAB) over the variable set before pooling*, so the variables read each other and the summary is
formed over *context-aware* variable representations rather than raw, independently-scored ones. That
second lever is precisely the inter-variable-correlation modelling I argued the domain needs and that no
lower rung has. There is direct evidence that attending-among-elements-before-pooling captures structure
fixed reductions cannot: on a max-value-regression toy task, the attentive self-attention-plus-pooling
construction matches the max-pooling oracle, while mean and sum pooling fail badly — because finding the
governing element requires the elements to be compared against each other, which only the interaction
buys. That is the same kind of relational structure I expect among meteorological variables.

So the aggregator is: `SAB` over the `V` variable tokens, then `PMA_1` (one seed) over the SAB output —
the "SAB + PMA" Set Transformer architecture, restricted to a single encoder block and one seed, which is
the minimal form that adds inter-variable interaction and multihead pooling over the cross-attention
baseline. Let me settle the remaining choices. Should the PMA pool the raw `Z` or `rFF(Z)`? The canonical
definition is `PMA_k(Z) = MAB(S, rFF(Z))` with a leading row-wise feed-forward, but in practice that
leading `rFF` is dropped when the preceding block already ends in a feed-forward — and SAB *does* end in
its `rFF` sublayer. So `PMA_1` applied directly to the SAB output is the standard, non-redundant form; I do
not add a second leading feed-forward. LayerNorm: keep it on (`ln=True`) in both blocks — this sits inside
a deep ViT pipeline fine-tuned from pretrained weights, and LayerNorm in the residual blocks is what keeps
the activations well-conditioned and stable during fine-tuning. Heads: use the pipeline's
`num_heads = 16`, so `d_k = D/16 = 64` per head, the standard head width and the same granularity the rest
of the model uses. The seed `S` is a learnable parameter, Xavier-initialized (the canonical init for the
seed and inducing points) so the attention starts with sensible-magnitude logits.

Now the implementation against the contract, since the indexing is where it lives or dies. The tokenizer
hands me `x: [B, V, L, D]`. The reduction is per-location and per-example and identical at every location,
so I treat every `(example, location)` pair as an independent `V`-element set: permute to `[B, L, V, D]` to
bring the variable axis adjacent to `D`, fold `B` and `L` into one batch `[B·L, V, D]` — `B·L` independent
set problems. Apply `SAB` (= `MAB(X, X)`): `[B·L, V, D] → [B·L, V, D]`, the variables now context-aware.
Expand the single seed to the batch, `[B·L, 1, D]`, and apply `PMA` (= `MAB(S, Z_enc)`):
`[B·L, 1, D]`. Squeeze the seed axis to `[B·L, D]` and unfold back to `[B, L, D]`. `V` never enters any
weight matrix — every `Linear` inside MAB maps `D → D`, and the seed is `1 × 1 × D` — so the module accepts
any `V`, exactly as the contract requires; `V` only sizes the runtime attention.

I should implement MAB faithfully to its canonical reference rather than reach for a library multihead
layer, because the residual placement and the rFF are part of the block's identity and I want the math to
match the reference exactly. MAB projects `Q, K, V` with three linears to width `D`; splits each into
`num_heads` chunks of width `D/num_heads` and stacks the heads along the batch dimension (the standard
trick that lets one `bmm` do all heads at once); computes attention `A = softmax(Q_ K_ᵀ / √D)` over the
key axis; forms the attention output with a residual on the projected query, `Q_ + A V_`; reassembles the
heads back along the feature axis; applies the first LayerNorm; then the feed-forward residual
`O + ReLU(fc_o(O))`; then the second LayerNorm. SAB calls `MAB(X, X)`; PMA calls `MAB(S, Z)` with the
expanded seed. (The full scaffold module — MAB plus the `SAB`-then-`PMA` aggregator — is in the answer.)

Let me trace the causal chain to be sure it holds. I need a permutation-invariant, any-`V`, scale-
respecting reduction of the variable tokens at each location. The Deep-Sets skeleton and the existing
reductions (mean, weighted sum, single-query cross-attention) all encode/score each variable *in
isolation*, so none can model inter-variable correlations — yet the right combination of meteorological
variables at a cell is relational, depending on what the other variables say. The fix is to let the `V`
tokens attend among themselves before pooling: a self-attention block (SAB = MAB(X, X)), which is
permutation-equivariant and resistant to the trivial collapse, mixes the variables into context-aware representations; then a
learnable-seed multihead attention pooling (PMA with one seed = MAB(S, ·)), permutation-invariant and
content-dependent, summarizes them into one token. The single-query cross-attention baseline is exactly the
`k=1`, no-encoder, single-head special case; this adds the two levers it lacks — multihead pooling and,
decisively, self-attention among the variables before pooling — at `O(V²)` per location, which is trivial
for `V = 48` and does not touch the backbone's `h·w` sequence. LayerNorm on, `num_heads = 16`, one SAB
block, one PMA seed, no positional encoding, no dropout: the minimal Set-Transformer pooling that captures
inter-variable structure the fixed reductions cannot.
