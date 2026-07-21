Plain RoPE is the strongest step so far, and the way it won tells me where the remaining loss is hiding.
Stripping the QK-Norm half edged the combined model out exactly as I predicted: validation loss 2.2570 vs
2.2589, WikiText-2 43.17 vs 43.44, LAMBADA 65.81 vs 67.20 — plain RoPE best on every language-modeling
metric, confirming that once position is relative, pinning q/k to the `sqrt(d_k)·cos` ceiling costs more
sharpness than it buys in stability. The split I flagged showed up downstream: the combined form kept a
slight edge on ARC-Easy (57.83 vs 57.32) and PIQA (64.74 vs 64.42), plain RoPE took HellaSwag (34.48 vs
34.24) and WinoGrande (51.70 vs 50.67). So plain RoPE is the strongest *language model* and the right base
to build on. Now read the ladder as a trajectory: the val_loss numbers are 2.2885 → 2.2589 → 2.2570. Fixing
position bought 0.0296; freeing the pinned sharpness bought 0.0019, an order of magnitude less. The scale
and position axes have given what they are going to give, and they are giving less each time. Another
0.03-class move will not come from refining scale or position further; it has to come from an axis I have not
touched at all.

So what do all three steps have in common? QK-Norm changed the logit *scale*; RoPE changed how *position*
enters the logit; neither touched the *shape of the attention distribution itself*. All three end in the
identical operation — a single softmax over the context, then a weighted average of values — and that single
softmax has a structural defect no amount of better position or scaling can reach, and I think it is exactly
where the difference between another 0.002 and another 0.03 is hiding.

The defect, precisely: the weights are `a_{m,n} = exp(s_{m,n})/sum_{n'} exp(s_{m,n'})`, and every `exp` is
strictly positive. No finite logit drives a weight to exactly zero, and the softmax has no mechanism to take
mass *back* off a position once it is placed. Over a 1024-token context, even if every irrelevant token gets
a weight of order `1/T`, the *aggregate* of those weights is most of the mass, so the output
`o_m = sum_n a_{m,n} v_n` is part signal and part a low-level average of the irrelevant remainder. RoPE put
the *peak* attention in the right relative location; it did nothing about the noise floor smeared under the
peak, because that floor is a property of softmax positivity, not of where the peak sits.

Whether that is worth a step depends on how the contamination is *structured*. Split the context into the
relevant set `S` with mass `p` and the irrelevant tail with mass `1 - p`, and write each value as
`v_n = μ + w_n`, where `μ` is a component *shared* across many tail tokens — and there really is such a
component, because frequent tokens (punctuation, function words, the attention-sink first position) recur
all over the context with value vectors pointing a common direction — and `w_n` is the idiosyncratic,
roughly zero-mean remainder. The tail's contribution splits:
`sum_tail a_n v_n = (1 - p)μ + sum_tail a_n w_n`. The two behave oppositely as context grows. The second
term is a weighted sum of zero-mean vectors; with roughly uniform tail weights `ε = (1 - p)/N` its norm is
about `(1 - p)||w||/sqrt(N)`, which *shrinks* as the context lengthens — the junk partly averages itself out.
But `(1 - p)μ` does *not* shrink with `N`: it is the entire tail mass pointed in the common direction,
surviving no matter how long the context. That common-mode term is the real contamination — a fixed fraction
`(1 - p)` of a shared value direction poured into every token's representation regardless of relevance.

And this is not something plain RoPE can already solve, which is the check on whether I need a new operator
at all. Plain RoPE handed the q/k magnitudes back, so a head *can* sharpen and shrink `1 - p`. But two things
stop sharpening from removing the floor. It cannot drive `1 - p` to zero: on genuinely ambiguous contexts the
content-similarity gap is bounded, and many heads must stay diffuse to hedge across several plausible tokens.
And more fundamentally, `(1 - p)μ` is *linear* in the residual tail mass, so a strictly-positive, sum-to-one
map can only reduce its coefficient, never cancel it — placing *negative* weight on the common direction is
the one thing it cannot do. Concentration reduces the coefficient of a term that only subtraction can remove.
That is the structural reason the last gain was small and why the next lever has to *change* the operator
rather than tune it.

So can I change the operator so it can *subtract* mass from irrelevant positions? Where does "cancel a common
floor by subtracting two correlated signals" already work — a differential amplifier rejects the voltage
common to its two inputs and amplifies only their difference; noise-cancelling headphones subtract an
estimate of the ambient floor. The principle is precise: the noise is *common* to two channels, the signal
*differs* between them, so the difference keeps the signal and kills the noise. Build attention the same way:
form *two* softmax maps over the same context from two query/key projections and take the differential
`A1 - lambda·A2` with a learnable scalar `lambda`, then average values with the signed map —
`(softmax(Q1 K1^T/sqrt d) - lambda·softmax(Q2 K2^T/sqrt d)) V`, with `Q = [Q1; Q2]`, `K = [K1; K2]`. Both
maps see the same content, so over the irrelevant tail their mass on `μ` is *correlated*, and `A1 - lambda·A2`
drives the tail's aggregate mass toward zero — `(1 - p1) - lambda(1 - p2) ≈ 0` even when each map alone still
puts substantial mass on the tail — so the `(tail mass)·μ` term is cancelled *without* either map reaching
`p → 1` on the signal. On the relevant tokens the model has an incentive to make the first map spike where
the second does not, so the difference is large there. The resulting weights no longer sum to one and are no
longer all positive — they are *signed*, exactly the new power I wanted: the operator can push an irrelevant
value's contribution to zero or below.

Before taking that as the move, the other operators on this axis, because "make the map able to zero things
out" has more than one answer and I want the one that fits the *structure* of the contamination, not just the
positivity symptom. The most direct is a sparse normalization — sparsemax or entmax — which assigns *exact*
zeros to low-scoring tokens. Tempting, but it still produces a *convex combination*: it can zero an
individual token but cannot place negative weight, and the contamination is not one loud irrelevant token, it
is `(1 - p)μ` smeared across the whole tail in a shared direction. To kill that, sparsemax would have to zero
essentially the entire tail — the over-committing a diffuse head cannot afford — so it fights the
coefficient, not the term, and on the engineering side it needs a sort or bisection at every one of the 24
layers with no fused kernel. A hard top-k mask has the same convex-combination limit plus a k to tune on a
frozen budget and a non-differentiable selection. Gating the *output* — a learned per-token gate — scales the
whole result including its signal and does nothing to the internal mix. The differential map is the only one
that produces genuinely *signed* weights, the exact algebraic power the common-mode cancellation requires —
not one option among several but the one whose new capability matches the shape of the defect.

One check before the engineering: will the subtraction be used, or will the optimizer just drive `lambda → 0`
and collapse this back to a single half-width softmax? At `lambda = 0` the operator is `A1·V`, ordinary
attention on a `head_dim/2`-wide head — a valid configuration the optimizer *could* choose, so I need a
reason it will not. The reason is that cancelling the common-mode floor genuinely lowers loss: a cleaner
average over the relevant values predicts the next token better, and the decomposition says the floor is a
real `(1 - p)μ`, not noise that averages away. As long as the loss gradient in `lambda` at 0 is negative, the
optimizer moves off zero — and I stack the deck with the init, centering `lambda` at `lambda_init ≈ 0.8` so
training *starts* already using strong subtraction and would have to actively learn `lambda → 0` to disable
it. The burden is on *disabling* the mechanism, the right default when I believe it helps.

Now the engineering that makes it a fair, stable drop-in, because a naive version would double the cost and
destabilize training. Three pieces. First, `lambda`: not a free scalar, which is badly conditioned and can
drift the subtraction into a wild magnitude that blows the output scale. I reparameterize
`lambda = exp(λq1·λk1) - exp(λq2·λk2) + lambda_init`, with four learnable `head_dim` vectors initialized
`N(0, 0.1)`. At init, with `head_dim = 32`, each dot `λq1·λk1` is a sum of 32 products of two independent
`N(0, 0.1)` scalars — each product mean 0, variance `10^-4`, so the sum has variance `3.2×10^-3`, standard
deviation ≈ 0.057 — so each exponential is `≈ 1 ± 0.057` and their difference has std `≈ 0.08`. So
`lambda ≈ lambda_init ± 0.08`: the two exponentials sit near 1 and roughly cancel, centering the whole
reparameterization on `lambda_init` with small, well-scaled signed gradients feeding the four vectors — the
controlled start a bare scalar would not give.

Second, the budget. Doubling the queries and keys, I *halve the head dimension* to stay matched: each logical
head uses `head_dim = n_embd/n_head/2 = 32`, with `2·n_head = 32` query/key sub-heads of dim 32 and
`n_head = 16` value heads of dim 64. The widths close exactly — q/k width `32·32 = 1024 = n_embd`, v width
`16·64 = 1024` — so the fused `c_attn` stays `n_embd → 3·n_embd`, not a single projection parameter moves,
and the doubling is absorbed by the halving. The only genuinely new parameters are the four lambda vectors
(`4·32 = 128` per layer) and the per-head output RMSNorm (`64` per layer), 4608 across 24 layers — against
355M that is `1.3×10^-5` of the model. And the FLOPs match: the `QK^T` cost is `n_maps·T²·d_per_map` —
vanilla `16·T²·64`, differential `32·T²·32` — identical, and the value-average is `16·T²·64` either way. So
any improvement is purely from the changed attention *shape*, not from added capacity or compute — the honest
way to claim the win.

Third, scale. The subtraction changes the operator's gain, and I have to compensate it in a way the frozen
optimizer can absorb. `A1`'s rows sum to 1 and `lambda·A2`'s sum to `lambda`, so the signed map's rows sum to
`1 - lambda`; at init `≈ 0.2`, meaning on the common direction `μ` the operator passes it at gain 0.2 instead
of the 1 a single softmax would — a 5× suppression of exactly the floor I target, the row-sum arithmetic
confirming the mechanism does what the decomposition promised. But that `1 - lambda` row-sum also shrinks the
*signal* part and *varies* head-to-head as `lambda` drifts, so I cannot leave it. I per-head normalize each
head's `2·head_dim` output with an RMSNorm — removing the variable, lambda-dependent shrinkage and restoring
unit scale — then rescale by the *fixed* constant `(1 - lambda_init)`. Fixed, not the live `lambda`: dividing
by the live `(1 - lambda)` would couple the residual-branch scale to the subtraction strength as it moves, a
moving target the frozen learning rate and cosine schedule were never tuned for. The fixed constant gives the
residual branch a predictable contribution matching what a standard single-softmax head would deliver at this
operating point, so the frozen optimizer, learning rate, and schedule transfer unchanged — which matters
because the loop is fixed and I cannot retune.

Now the two compromises the edit surface forces, which I have to name honestly. The editable region is only
`CausalSelfAttention(config)`, and `config` does not carry the *layer index*. The natural refinement scales
the subtraction with depth — gentle early, strong deep — via `lambda_init = 0.8 - 0.6·exp(-0.3·(l-1))`, which
needs `l`, and every one of the 24 blocks constructs from the same `config` with no `l`. That schedule wants
`lambda ≈ 0.20` at `l = 1` (row-sum ≈ 0.8, mostly the first map), `0.73` at `l = 8`, `≈ 0.80` at `l = 24`. I
can set only one number, so I set the deep-layer asymptote `lambda_init = 0.8`, which is also where the
reparameterization is centered. The early layers then cancel far harder than the schedule would prescribe —
0.8 where it wanted 0.2 — a genuine omission forced by the edit surface, not a choice: if the early layers
are hurt by over-cancellation, that is the missing depth schedule showing, not the method failing. The second
compromise: the fused SDPA path returns only the final averaged output, not the two softmax maps I need to
subtract, so I take the manual masked-softmax path the default code already provides, compute the `2·n_head`
maps explicitly, reshape to `(B, n_head, 2, T, T)`, subtract, and matmul with v — paying the memory of the
explicit `T×T` maps over 32 maps rather than 16, unavoidable because the differential subtraction lives
*between* the softmax and the value-average, exactly where the fused kernel hides its internals. Position
stays the strongest baseline's: RoPE on the doubled q/k sub-heads (`use_pos_emb = False`, the split-half
rotation from step 3 applied to the `head_dim`-wide sub-heads). So this step is the strongest position scheme
with the single softmax replaced by a differential one. The full module is in the answer.

The bar, against the real numbers with no feedback to lean on since this is the finale. Beat plain RoPE:
val_loss 2.2570, WikiText-2 43.17, LAMBADA 65.81, ARC-Easy 57.32, HellaSwag 34.48, PIQA 64.42, WinoGrande
51.70. The operator is parameter- and FLOP-matched, so any gain is purely the changed shape. My expectation
follows from the decomposition: since the common-mode floor is a real `(1 - p)μ` fraction of the output over
a 1024-token context and the subtraction removes it structurally rather than by concentration, val_loss and
both perplexities should fall below the RoPE numbers — LAMBADA most of all, since its long-passage last-word
prediction is exactly the retrieval-under-noise the cancellation targets. Downstream should move with the LM
gain, the retrieval-flavored tasks likeliest to lift. Two risks I would actually validate before believing
the win. The fixed `lambda_init = 0.8` may over-cancel the early layers, per the schedule values above — so a
modest-but-positive result might be the depth-schedule omission rather than a failure of the mechanism, and
passing `l` would recover more. And the signed map can, at init, produce a higher-variance output than a
single softmax, being a difference of two unit-sum maps — I would watch the first few hundred iterations for
an instability the `(1 - lambda_init)` compensation and the per-head norm are there to prevent. The honest
fallback if val_loss does *not* drop below 2.2570: at 355M parameters, 7.1B tokens, and a 1024-token context
the attention-noise floor may simply not be binding yet — the common-mode term is largest, and the
cancellation most valuable, when the context is long enough that the irrelevant tail dominates the mass, and
1024 is modest. In that case the strongest configuration remains plain RoPE, and the differential operator is
a bet that the floor becomes binding at larger scale and longer context than this run.
