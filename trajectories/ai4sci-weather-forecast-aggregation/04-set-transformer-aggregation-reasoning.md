The cross-attention rung confirmed the whole ladder's logic and, in doing so, drew a sharp line under what
content-dependent pooling buys. Against the weighted sum's 345.79 / 2.5934 / 3.4245, the learnable-query
cross-attention landed 255.85 / 2.1214 / 3.1455 — a large drop on every target, and crucially it pulled
wind10m-7day back under the mean's floor (3.1455 < 3.3991), exactly the falsifiable bar I had set for it.
So the diagnosis held: the weighted sum's wind regression really was the cost of a frozen, content-
independent weighting, and letting the weight on each variable react to the token contents fixed it. The
z500-3day collapse from 345.79 to 255.85 is the headline — content-dependent, multi-head, value-
re-projecting pooling captures the state-dependent dynamical structure the global split could not. The
ladder is internally consistent: uniform → fixed global weighting → content-dependent attention, each rung
repairing the previous one's measured defect. Now I have to ask what the *cross-attention* rung itself
still leaves on the table, because that is where the next move lives.

Let me look hard at what the single query actually does, token by token. At one location it forms one
learnable query `q`, scores each variable token by `(W^Q q)·(W^K x_v)/√d_k`, softmaxes over the `V`
variables, and returns `Σ_v α_v (W^V x_v)`. The weights are content-dependent — they react to what each
token says — and that is the win I just measured. But notice *how* each weight is computed: the score on
variable `v` is a compatibility between the fixed query and `x_v` *alone*. The query asks the same fixed
question of every token independently; the score on the geopotential token does not depend on what the
wind token is currently saying, and vice versa. The variable tokens never read each other before they are
pooled. So cross-attention reacts to each variable's content in isolation, but it cannot express a
*relational* rule — "upweight the wind token *because* the geopotential token indicates a sharp gradient
here." That kind of inter-variable conditioning is exactly the structure I'd expect to matter most on the
hardest target: wind10m-7day is the longest lead (7 days) and the smallest absolute headroom (3.1455),
and at long lead the informative combination of variables is plausibly the most state-coupled — which
variable carries the signal depends on the joint atmospheric configuration, not on any one variable's
token in isolation. So the cross-attention number, good as it is, is the number I'd expect from a pooling
that scores variables independently; the residual structure it can't reach is the correlations *among* the
variables.

That points at the one lever every rung so far has left unpulled. The mean encodes each variable in
isolation (identity, then average); the weighted sum encodes each in isolation (scalar weight, then sum);
the cross-attention scores each in isolation (query-vs-token, then weighted sum of re-projected values).
None of them lets the `V` variable tokens *interact* before the summary is formed. If I want the pooling
weight on variable `i` to be able to depend on variable `j`'s contents, the variables have to attend to
*each other* first, producing context-aware representations, and only then be pooled. That is a different
and strictly more expressive construction than scoring-then-pooling, and it is precisely the gap the
cross-attention rung's isolation-scoring leaves.

I have to be careful, because "let the elements interact, then pool" has a collapse trap: if the
interaction stage is optimized only to make pooling easy, the cheapest solution can be an encoder that maps
every variable token to the same vector — trivially poolable, useless. So the interaction mechanism must be
one that genuinely mixes information across the set and resists the trivial collapse. Self-attention is exactly that:
each token's new representation is a content-weighted combination of *all* the tokens' values, with a
residual that carries the token's own information forward, so the identity-collapse the bare encoder
risked is not the cheap optimum — folding every token to a constant would have to fight the residual rather
than ride a free gradient down to it — and it is permutation-equivariant — permute the inputs and the outputs permute the
same way — which is what I need so that a permutation-invariant pooling on top yields a permutation-
invariant whole. So the move is: a self-attention block over the `V` variable tokens (the variables read
each other), then an attention pooling on top (content-dependent, set-valued, returning one token).

Let me build both from one well-behaved block and check the pieces against the constraints the contract and
the pretrained backbone impose. The block is a Transformer encoder block adapted for sets — multihead
attention plus a feed-forward, residuals, LayerNorm, but **no positional encoding and no dropout** (a
position would break the permutation symmetry I'm relying on; stochastic dropout would break the
determinism of the set map). Call it MAB(X, Y) for a query set `X` over key/value set `Y`:
`H = LayerNorm(X + Multihead(X, Y, Y))`, then `MAB(X, Y) = LayerNorm(H + rFF(H))`, with `rFF` a row-wise
feed-forward and the usual `1/√d` scaling so the softmax doesn't saturate (the same responsiveness
argument that mattered at the cross-attention rung). From MAB, two blocks fall out. Self-attention within
the variable set is `SAB(X) = MAB(X, X)` — tokens are queries, keys, and values at once, so each variable
reads from all the others by content compatibility; this is the inter-variable interaction the lower rungs
lack, and one block of pairwise mixing is a sensible amount of structure to add for a set of `V = 48`. The
pooling is `PMA` with one learnable *seed* vector `S`: `PMA(Z) = MAB(S, Z)`, the seed a trainable query
that asks the encoded set for its summary, returning one token; multihead so it can pose several summary
sub-questions (the thermodynamic and dynamical summaries) and combine them through the block's output
projection. One seed gives exactly the one token per location the contract wants — I do not need more
seeds, which are for problems wanting several correlated outputs.

Now the key consistency check that tells me this is the *right* next rung and not a sideways move: how does
this relate to the cross-attention I'm trying to beat? A one-seed, single-head PMA on the *raw* tokens with
no encoder is essentially the cross-attention rung — a learnable query pooling the variable tokens. So the
ClimaX cross-attention is the no-encoder, `k=1`, single-head special case of this construction. The two
things I'm adding over it are exactly the two it lacks: the pooling is multihead, and — the decisive one —
there is a self-attention encoder (SAB) over the variable set *before* pooling, so the summary is formed
over context-aware variable representations rather than independently-scored raw ones. That is the
inter-variable-correlation modelling I argued wind10m-7day most needs. And this is not idle: the
attend-among-elements-then-pool construction is known to match a max-pooling oracle on a max-value-
regression task where mean and sum pooling fail badly — because identifying the governing element requires
the elements to be compared against each other, the same relational structure I expect among
meteorological variables. So the next rung is `SAB` then one-seed `PMA`: the minimal Set-Transformer
pooling that adds interaction and multihead pooling over the cross-attention baseline.

Cost and conditioning, since this sits inside a deep backbone fine-tuned from pretrained weights. The SAB
is `O(V²)` per location — but `V = 48`, so this is a `48×48` attention at each of the 512 locations, tiny,
and it does *not* reintroduce the `O((V·h·w)²)` blowup that motivated aggregation in the first place (that
came from the backbone over the full variable-and-space sequence; here the backbone still sees only the
`h·w` aggregated sequence). So I can afford the full SAB rather than the inducing-point bottleneck (ISAB),
which is meant for large sets. I keep LayerNorm on in both blocks — inside a deep ViT fine-tune, LayerNorm
in the residual blocks is what keeps activations well-conditioned and training stable — and use the
pipeline's `num_heads = 16` (`d_k = 64`/head, the standard width the rest of the model uses). The seed is a
learnable parameter, Xavier-initialized so the pooling attention starts with sensible-magnitude logits.
Unlike the cross-attention rung I cannot zero-initialize to recover the mean, because the SAB encoder makes
the map nonlinear in the tokens — but the residual structure and LayerNorm keep the starting point
well-behaved, and fine-tuning from pretrained ClimaX weights means the backbone adapts to the new
aggregator's output distribution.

The shapes follow the same per-location folding as the cross-attention rung. The tokenizer hands me
`x: [B, V, L, D]`; permute to `[B, L, V, D]` and fold to `[B·L, V, D]` so each `(example, location)` is an
independent `V`-set. Apply SAB: `[B·L, V, D] → [B·L, V, D]`, the variables now context-aware. Expand the
seed to `[B·L, 1, D]` and apply PMA: `[B·L, 1, D]`. Squeeze and unfold to `[B, L, D]`. `V` never enters any
weight — every `Linear` inside MAB maps `D → D` and the seed is `1×1×D` — so the module accepts any `V`,
as the contract requires. I implement MAB faithfully to its canonical reference (three projections to width
`D`, heads split and stacked along the batch dim for one `bmm`, residual `Q_ + A V_`, LayerNorm, then the
feed-forward residual `O + ReLU(fc_o(O))`, LayerNorm); SAB calls `MAB(x, x)`, PMA calls `MAB(seed, z)`.
(The full scaffold module — MAB plus the SAB-then-PMA aggregator — is in the answer.)

So the delta from the cross-attention rung is precise: where that rung scored each variable token
independently against one fixed query and pooled the re-projected values, I now first let the `V` variable
tokens attend among themselves (SAB) so each is represented in the context of the others, then pool the
context-aware tokens with a learnable-seed multihead attention (PMA). The pooling weight on a variable can
now depend on what the other variables are saying — the inter-variable correlation the cross-attention
cannot express.

Let me close on the bar this has to clear, stated against the real cross-attention numbers, since this rung
is the endpoint and carries no measurement of its own. To justify the added SAB block, it must beat
255.85 / 2.1214 / 3.1455 on z500-3day / t850-5day / wind10m-7day. The sharpest, most falsifiable test is
wind10m-7day: that is the longest-lead, most state-coupled target and the smallest residual headroom
(3.1455), and it is precisely where I argued inter-variable correlation should matter most — so if the
Set-Transformer pooling cannot get wind10m below 3.1455, then attending-among-variables-before-pooling is
*not* what that target needed and the extra block is unjustified. On z500-3day and t850-5day I'd expect
smaller relative gains than the jump cross-attention already delivered (those targets' biggest defect — the
frozen global split — was already fixed by content-dependent pooling), but still a measurable improvement
if relational structure among the dynamical and thermodynamic variables is real. The risk I'd watch is
overfitting: SAB roughly triples the aggregator's parameters over the single cross-attention layer, and on
a single-seed, single-block budget fine-tuned from pretrained weights, a too-expressive aggregator could
fit noise and *fail* to clear the cross-attention bar on the easier targets even while helping wind10m. So
the validation I would run is exactly the three-target comparison against 255.85 / 2.1214 / 3.1455, with
wind10m-7day as the decisive rung: clear it on all three (and especially wind10m) and the ladder's final
step — from independent-scoring pooling to interaction-then-pooling — is earned.
