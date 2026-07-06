Plain RoPE is the strongest rung so far, and the way it won tells me where the remaining loss is hiding.
Stripping the QK-Norm half edged the combined model out exactly as I predicted: validation loss 2.2570
against RoPE + QK-Norm's 2.2589, WikiText-2 43.17 vs 43.44, LAMBADA 65.81 vs 67.20 — plain RoPE best on
every language-modeling metric, confirming that once position is relative, pinning q/k to the
`sqrt(d_k)·cos` ceiling cost more sharpness than it bought in stability. The split I flagged did show up on
downstream: RoPE + QK-Norm kept a slight edge on ARC-Easy (57.83 vs 57.32) and PIQA (64.74 vs 64.42), the
multiple-choice tasks, while plain RoPE took HellaSwag (34.48 vs 34.24) and WinoGrande (51.70 vs 50.67). So
plain RoPE is the strongest *language model* and the right rung to build on, with the combined form not
strictly dominated. Now let me read the whole ladder as a trajectory, because the *shape* of the gains tells
me whether there is anything left on the axes I have been working. The three val_loss numbers are 2.2885 →
2.2589 → 2.2570. The first step, fixing position, bought 0.0296. The second, freeing the sharpness the
ceiling had pinned, bought 0.0019 — an order of magnitude smaller. The scale axis and the position axis have
given what they are going to give, and they are giving less each time. If I want another 0.03-class move
rather than another 0.002-class one, I cannot get it by refining scale or position further; I have to touch
an axis I have not touched at all.

So step back and look at what *every* rung so far has in common, because that is the seam none of them
touched. QK-Norm changed the logit *scale*. RoPE changed how *position* enters the logit. Neither touched
the *shape of the attention distribution itself* — all three rungs end in the identical operation: a single
softmax over the context, then a weighted average of values. And that single softmax has a structural defect
that no amount of better position or better scaling can reach, and I think it is exactly where the
difference between "another 0.002 by refining an old axis" and "another 0.03 by opening a new one" is hiding.

Here is the defect, stated precisely. The attention weights are `a_{m,n} = exp(s_{m,n}) / sum_{n'}
exp(s_{m,n'})`, and every `exp(·)` is strictly positive. So no finite logit can drive a weight to exactly
zero — the softmax can make an irrelevant token's weight small, but never zero, and it has no mechanism to
take mass *back off* a position once it has been placed. Over a 1024-token context, even if every irrelevant
token gets a weight of order `1/T`, the *aggregate* of those weights is most of the mass. The output
`o_m = sum_n a_{m,n} v_n` is therefore part signal — the few tokens that matter — and part a low-level
average of the irrelevant remainder. RoPE made the model place its *peak* attention in the right relative
location; it did nothing about the noise floor smeared under that peak, because the floor is a property of
the softmax's positivity, not of where the peak sits. The 2.2570 is, in part, the cost of averaging junk
into every token's representation.

Let me make "noise floor" concrete rather than leave it as a worry, because whether it is worth a whole rung
depends on how the contamination is *structured*, and the structure is what decides whether the fix has to
be subtraction or whether plain RoPE's sharpness freedom already handles it. Split the context into the
relevant set `S` (a few tokens) with total mass `p` and the irrelevant tail with mass `1 − p`. Write each
value vector as `v_n = μ + w_n`, where `μ` is a component *shared* across many tail tokens — and there really
is such a component, because frequent tokens (punctuation, function words, the attention-sink first position)
recur all over the context and their value vectors point in a common direction — and `w_n` is the
idiosyncratic, roughly zero-mean remainder. Then the tail's contribution to the output splits:
`sum_tail a_n v_n = (sum_tail a_n) μ + sum_tail a_n w_n = (1 − p) μ + sum_tail a_n w_n`. The two terms behave
completely differently as the context grows. The second term is a weighted sum of zero-mean idiosyncratic
vectors; if the tail weights are roughly uniform at `ε = (1 − p)/N` over `N` tail tokens, its norm is about
`ε·sqrt(N)·||w|| = (1 − p)·||w||/sqrt(N)`, which *shrinks* as the context lengthens — the junk partly
averages itself out. But the first term, `(1 − p)·μ`, does *not* shrink with `N`: it is the entire tail mass
pointed in the common direction, and it survives no matter how long the context. That common-mode term is
the real contamination — a fixed fraction `(1 − p)` of a shared value direction, poured into every token's
representation regardless of relevance.

Now here is why this is not a problem plain RoPE can already solve, which is the check that tells me whether
I need a new operator at all. Plain RoPE handed the q/k magnitudes back to the optimizer, so a head *can*
sharpen — grow the logit gap and shrink `1 − p`. But two things stop sharpening from removing the
common-mode term. First, it cannot drive `1 − p` to zero: on genuinely ambiguous contexts the
content-similarity gap between the target and the field is bounded, so the softmax has a floor it cannot
beat, and many heads must stay deliberately diffuse to hedge across several plausible tokens — a maximally
sharp head that commits to one token would be wrong too often to be worth it. Second, and more
fundamentally, the contamination `(1 − p)·μ` is *linear* in the residual tail mass. Sharpening shrinks
`1 − p`, so it shrinks this term proportionally — but it can never cancel it, because a strictly-positive,
sum-to-one map has no way to place *negative* weight on the common direction. Concentration reduces the
coefficient of a term that only subtraction can remove. So refining the same single-softmax axis — sharper
heads, better-scaled logits — is spending effort on the coefficient when the term itself is what stands in
the way, and that is the structural reason the last gain was small and the reason the next lever has to
*change* the operator rather than tune it.

So the question is whether I can change the operator so it can *subtract* mass from irrelevant positions —
because subtraction is the one thing a single positive softmax cannot do, and it is exactly what removes a
term shared between two correlated channels. Where does "cancel a common floor by subtracting two correlated
signals" already work? A differential amplifier rejects the voltage common to its two inputs and amplifies
only their difference, cancelling common-mode noise; noise-cancelling headphones subtract an estimate of the
ambient floor. The shared principle is precise: the noise is *common* to two channels, the signal *differs*
between them, so the difference keeps the signal and kills the noise. I want attention built the same way.
The construction follows directly. Form *two* softmax maps over the same context from two different
query/key projections — `A1 = softmax(Q1 K1^T / sqrt(d))`, `A2 = softmax(Q2 K2^T / sqrt(d))` — and take the
differential `A1 − lambda·A2` with a learnable scalar `lambda`, then average the values with that signed map:
`(softmax(Q1 K1^T/sqrt(d)) − lambda·softmax(Q2 K2^T/sqrt(d))) V`, with `Q = [Q1; Q2]`, `K = [K1; K2]`. Check
it against the decomposition I just wrote, because that is what tells me it targets the right term. Both maps
see the same content, so over the irrelevant tail their mass on the common direction `μ` is *correlated* —
that is the common-mode part — and `A1 − lambda·A2` drives the tail's aggregate mass toward zero:
`(1 − p1) − lambda(1 − p2)` can be made ≈ 0 even when each map individually still puts substantial mass on
the tail, so the `(tail mass)·μ` term is cancelled *without* needing either map to reach `p → 1` on the
signal. On the relevant tokens, the model has an incentive to make the first map spike where the second does
not, so the difference is large there. The resulting weights no longer sum to one and are no longer all
positive — they are *signed*, which is exactly the new power I wanted: the operator can push an irrelevant
value's contribution to zero or below instead of being stuck with the positive floor.

Before I take that as the move, let me check the other operators that could attack this same axis, because
"make the map able to zero things out" has more than one answer and I want the one that fits the *structure*
of the contamination, not just the positivity symptom. The most direct alternative is a sparse normalization
— sparsemax or entmax — which replaces the softmax with a projection that can assign *exact* zeros to
low-scoring tokens. That is tempting: it attacks positivity head-on, driving irrelevant weights to a true
zero instead of a small floor. But walk it against the common-mode decomposition and it comes up short.
Sparsemax still produces a *convex combination* — non-negative weights that sum to one — so it can zero out
an individual token but it cannot place *negative* weight, and the contamination I identified is not one loud
irrelevant token, it is the aggregate `(1 − p)·μ` smeared across the whole tail in a shared direction. To
kill that, sparsemax would have to zero out essentially the entire tail, which is exactly the over-committing
a diffuse head cannot afford; it reduces `1 − p` harder than softmax can, but it is still fighting the
coefficient, not the term. And on the engineering side it needs a sort or a bisection at every one of the 24
layers, with no fused kernel here, so it pays in both structure and speed. A second option is a hard top-k
mask — keep the k largest weights, drop the rest — which has the same convex-combination limitation plus a k
hyperparameter I would have to tune on a frozen budget and a non-differentiable selection. A third is to gate
the *output* — multiply the attention result by a learned per-token gate — but that scales the whole output
including its signal and does nothing to the internal mix, so it cannot separate the common-mode floor from
the signal at all. The differential map is the only one of the four that produces genuinely *signed* weights,
which is the exact algebraic power the common-mode cancellation requires. So it is not merely one option
among several; it is the one whose new capability matches the shape of the defect.

One check before the engineering: is the subtraction actually going to be used, or will the optimizer just
drive `lambda` to 0 and collapse this back to a single half-width softmax? At `lambda = 0` the operator is
`A1·V`, ordinary attention on a `head_dim/2`-wide head — the plain-RoPE-class model with narrower heads, no
cancellation. That is a valid configuration the optimizer *could* choose, so I need a reason it will not. The
reason is that cancelling the common-mode floor genuinely lowers loss: a cleaner average over the relevant
values predicts the next token better, and the decomposition says the floor is a real `(1 − p)·μ`
contamination, not noise that averages away, so removing it is worth real nats. As long as the gradient of
the loss with respect to `lambda` at `lambda = 0` is negative — more cancellation helps — the optimizer
moves off zero. And I stack the deck the right way with the initialization: I center `lambda` at
`lambda_init` (about 0.8, below), so training *starts* already using strong subtraction and would have to
actively learn `lambda → 0` to disable it. The burden is on *disabling* the mechanism, not on discovering it,
which is the right default when I believe the mechanism helps.

Now the engineering that makes it a fair, stable drop-in, because a naive version would double the cost and
destabilize training. Three pieces. First, `lambda`: I do not learn it as a free scalar — a free scalar is
badly conditioned and can drift the subtraction into a wild magnitude that blows the output scale. I
re-parameterize it as `lambda = exp(lambda_q1·lambda_k1) − exp(lambda_q2·lambda_k2) + lambda_init`, with four
learnable `head_dim` vectors initialized `N(0, 0.1)`. Let me check it starts where I want. With
`head_dim = 32`, each dot product `lambda_q1·lambda_k1` is a sum of 32 products of two independent
`N(0, 0.1)` scalars; each product has mean 0 and variance `0.1²·0.1² = 10^-4`, so the sum has mean 0 and
variance `32·10^-4 = 3.2×10^-3`, standard deviation ≈ 0.057. So each dot product is ≈ `0 ± 0.057`, each
exponential is ≈ `1 ± 0.057`, and their difference has standard deviation `sqrt(2)·0.057 ≈ 0.08`. At init
`lambda ≈ lambda_init ± 0.08` — the two exponentials sit near 1, roughly cancel, and the whole
reparameterization is centered on `lambda_init` with small, well-scaled signed gradients feeding the four
vectors. That is exactly the controlled start a bare scalar would not give me.

Second, the budget. I have doubled the queries and keys, so to match a vanilla head's parameters and FLOPs I
*halve the head dimension*: each logical head uses `head_dim = n_embd / n_head / 2 = 32`, with `2·n_head = 32`
query/key sub-heads of dimension 32 and `n_head = 16` value heads of dimension `2·head_dim = 64`. Check the
widths close exactly: total q/k width is `2·n_head·head_dim = 32·32 = 1024 = n_embd`, total v width is
`n_head·2·head_dim = 16·64 = 1024 = n_embd` — so the fused `c_attn` projection is the unchanged
`n_embd → 3·n_embd`, not a single parameter moves in the projection, and the doubling is absorbed entirely by
the halving. The only genuinely new parameters are the four `head_dim` lambda vectors (`4·32 = 128` per
layer) and the per-head RMSNorm on the `2·head_dim` output (`64` per layer), 192 per layer, `4608` across 24
layers — against 355M that is `1.3×10^-5` of the model, so "parameter-matched" is true to four significant
figures. Now the FLOPs, because the doubling could hide there instead of in the parameters. The `QK^T` cost
is `n_maps · T² · d_per_map`: vanilla is `16 · T² · 64`, differential is `32 · T² · 32 = 16 · T² · 64` —
identical. The value-average `AV` cost after the subtraction is `n_head` signed maps times the `2·head_dim`
value width, `16 · T² · 64`, again identical to vanilla's `16 · T² · 64`. The two big matmuls are
FLOP-matched exactly; the only genuine overheads are that the softmax is computed over `2·n_head = 32` maps
instead of 16 (double the elementwise exp, but exp is cheap next to the matmuls) and the subtraction itself
(`16·T²`, negligible). So any loss improvement is purely from the changed attention *shape*, not from added
capacity or compute — which is the honest way to claim the win.

Third, scale. The subtraction changes the operator's gain, and I have to compensate it in a way the frozen
optimizer can absorb. Each softmax row sums to 1, so `A1`'s rows sum to 1 and `lambda·A2`'s rows sum to
`lambda`, and the signed map `(A1 − lambda·A2)` has rows summing to `1 − lambda`. At init
`lambda ≈ lambda_init = 0.8`, so the rows sum to about 0.2 — which, on the common-mode direction `μ`, means
the operator passes it at gain 0.2 instead of the gain 1 a single softmax would, a 5× suppression of exactly
the floor I am targeting, so the row-sum arithmetic confirms the mechanism does what the decomposition
promised. But that `1 − lambda` row-sum also shrinks the *signal* part of the output, and it *varies*
head-to-head as `lambda` drifts during training, so I cannot leave it. I per-head normalize each head's
`2·head_dim` output — a per-head RMSNorm, the GroupNorm-across-heads discipline — which removes the variable,
lambda-dependent shrinkage and restores a unit-scale output, and then I rescale by the *fixed* constant
`(1 − lambda_init)`. Fixed, not the learned `lambda`: if I divided by the live `(1 − lambda)` the output gain
would track `lambda` as it moves and couple the residual-branch scale to the subtraction strength — a moving
target the frozen learning rate and schedule were never tuned for. Multiplying by the constant
`(1 − lambda_init)` instead gives the residual branch a fixed, predictable contribution matching what a
standard single-softmax head would deliver at this operating point, so the frozen GPT-2 Medium optimizer,
learning rate, and cosine schedule transfer unchanged — which matters here precisely because the loop is
fixed and I cannot retune.

Now the part specific to *this* edit surface, where the harness forces two real compromises I have to name
honestly. The editable region is only `CausalSelfAttention(config)`, and `config` does not carry the *layer
index*. The full method scales the subtraction with depth, `lambda_init = 0.8 − 0.6·exp(−0.3·(l − 1))`, which
needs `l` — and I do not have it: every one of the 24 blocks constructs its attention from the same `config`
with no `l`. Let me see what I am giving up by pricing the schedule out. At `l = 1` it prescribes
`0.8 − 0.6 = 0.20`; at `l = 8`, `0.8 − 0.6·exp(−2.1) = 0.73`; at `l = 24`, `0.8 − 0.6·exp(−6.9) ≈ 0.80`. So
the schedule wants *gentle* cancellation in the early layers (`lambda ≈ 0.2`, row-sum ≈ 0.8, mostly the
first map) ramping to *strong* cancellation deep (`lambda ≈ 0.8`, row-sum ≈ 0.2). I can only set one number,
so I set the deep-layer asymptote, `lambda_init = 0.8`, which is also the operating point the
reparameterization is centered on. This means the early layers will cancel far harder than the schedule would
prescribe — 0.8 where it wanted 0.2 — a genuine omission, forced by the edit surface and not a choice: if the
early layers are hurt by over-cancellation, that is the depth-schedule loss showing, not the method failing.
The second compromise: the fused SDPA path returns only the final averaged output, not the two softmax maps I
need to subtract. So I cannot use Flash here; I take the manual masked-softmax path the scaffold already
provides as the non-flash fallback, compute the `2·n_head` attention maps explicitly, reshape to
`(B, n_head, 2, T, T)`, subtract, and matmul with v. That costs the memory of the explicit `T×T` maps — the
same cost the manual fallback always had, now over 32 maps rather than 16 — and it is unavoidable, because
the differential subtraction lives *between* the softmax and the value-average, exactly where the fused
kernel hides its internals. Position stays as the strongest baseline's: RoPE on the doubled q/k sub-heads
(`use_pos_emb = False`, the split-half rotation from step 3 applied to the `head_dim`-wide sub-heads), so
this rung is precisely "the strongest position scheme, with the single softmax replaced by a differential
one." The full scaffold module is in the answer.

Let me state the bar this has to clear, against the real numbers, with no invented results — this is the
finale and there is no feedback to lean on. The rung to beat is plain RoPE: validation loss 2.2570,
WikiText-2 43.17, LAMBADA 65.81, ARC-Easy 57.32, HellaSwag 34.48, PIQA 64.42, WinoGrande 51.70. The
differential operator is parameter- and FLOP-matched, so any gain is purely from the changed attention
*shape*, not from added capacity — which is the honest way to claim it. My falsifiable expectation follows
from the decomposition: because the common-mode floor is a real `(1 − p)·μ` fraction of the attention output
on a 1024-token context and the subtraction removes it structurally rather than by concentration, validation
loss and both perplexities should fall below the RoPE numbers — LAMBADA most of all, since its long-passage
last-word prediction is exactly the retrieval-under-noise setting the cancellation targets, so the 65.81 is
the number I would watch first. Downstream should move with the LM gain, with the in-context,
retrieval-flavored tasks (ARC-Easy, HellaSwag) the likeliest to lift, since cleaner attention to the relevant
span is what those reward. The risks I would actually validate before believing the win are two. First, the
fixed `lambda_init = 0.8` may over-cancel the early layers, per the schedule values I just computed — so a
plain-RoPE-beating-but-modest result might be the depth-schedule omission showing rather than a failure of
the mechanism, and if I could pass depth I would expect a further gain. Second, the signed map can, at init,
produce a higher-variance output than a single softmax, since it is a difference of two unit-sum maps; I
would watch the first few hundred iterations for an instability that the `(1 − lambda_init)` compensation and
the per-head norm are there to prevent. And the honest fallback if validation loss does *not* drop below
2.2570: at 355M parameters, 7.1B tokens, and a 1024-token context the attention-noise floor may simply not be
the binding constraint yet — the common-mode term `(1 − p)·μ` is largest, and the cancellation most valuable,
when the context is long enough that the irrelevant tail dominates the mass, and a 1024-token window is
modest. In that case the strongest rung remains plain RoPE and the differential operator is a bet that the
floor becomes binding — and the subtraction pays off — at larger scale and longer context than this task
runs.
