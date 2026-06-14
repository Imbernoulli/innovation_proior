BAIT confirmed exactly the split I predicted, and the place it won and the place it lost together point to
what the strongest rung has to be. On **spambase** the Fisher A-optimal objective came in at 0.929 / 0.911
on the mean — above BALD's 0.905 / 0.892 and edging least confidence's 0.927 / 0.908 — and on **splice**
at 0.813 / 0.756, above BALD's 0.795 / 0.737. So filling the diversity hole with a pool-aware information
criterion genuinely helped where diversity was the binding constraint, just as the derivation said. But on
**letter** BAIT collapsed: mean 0.791 / 0.671, *below* BALD's 0.836 / 0.716 and below even random's 0.816 /
0.724, and seed 42 has no letter result at all — the run did not complete. That is the CPU adaptation
biting precisely where I said it would: 26 classes makes `d·k` largest, so the random projection to 128
dims discards the most Fisher geometry, the streaming full-pool Fisher accumulation is slowest, and the
time/memory budget can fail outright. So the lesson is sharp. The A-optimal *objective* is right —
uncertainty fused with pool-aware diversity beats per-point uncertainty — but the *machinery* to compute it
exactly (per-class rank-`k` Fisher factors, a `d·k × d·k` pool matrix, projections, pseudo-inverses,
forward/backward greedy) is too heavy to survive this harness on the hardest dataset. I want the same
fused property — long *and* diverse batches, no tradeoff knob — but obtained *cheaply*, with no matrix
inverse, no projection, nothing that blows up when the number of classes is large. That is the constraint
the final rung has to satisfy: BAIT's geometry, BALD's robustness on letter, at uncertainty-sampling cost.

Let me think about what "the model is uncertain about this point" even means for a net trained by gradient
descent, because the right currency is hiding there. The whole machine moves by gradients: an example is
informative exactly when, once I know its label, it induces a large gradient of the loss and therefore a
large parameter update. So the natural currency of informativeness is gradient magnitude. If I knew the
true label `y` of `x`, I'd compute `g_x^y = ∂ℓ_CE/∂θ` and call `‖g_x^y‖` the example's pull on the model.
Two things bug me about taking the *full*-parameter gradient: it's a giant object, and I'd have to average
it over the unknown label. So take the cheapest informative slice — the gradient with respect to just the
*last* layer's weights — and watch what it factors into. With `f(x) = softmax(W z(x))`, `z` the penultimate
embedding and `W` the final linear layer, the cross-entropy gradient in the block for class `i` is

  (g_x^y)_i = (p_i − 1[y=i]) · z(x).

Stare at this. The whole last-layer gradient embedding is an outer product: `g_x^y = (p − e_y) ⊗ z`, the
probability-residual `r = p − e_y` tensored with the penultimate embedding `z`. That decomposition is the
thing I have been chasing across three rungs. The `z` factor is *exactly* the representation space — it
carries where the example sits, its diversity and identity, the same penultimate features. The `r` factor
carries how wrong/uncertain the model is — if the model is confident and right, `p ≈ e_y` so `r ≈ 0`; if
it's unsure, `p` is spread and `r` is large. One object, and it already contains both of the things the
ladder kept separating: representation in `z` (the diversity BAIT got from the pool Fisher) and uncertainty
in `r` (what least confidence and BALD scored). I didn't have to balance them with a coefficient — they are
multiplied together blockwise inside a single vector. And it is *rank-one* per point, not the rank-`k`
Fisher BAIT had to project: this is the cheapness I need.

But the obvious snag: `g_x^y` needs the true label `y`, which is the whole point — I don't have it. The
expensive answer is to average over `y` under the model. Let me try the cheap thing and check whether it's
defensible or just lazy: hallucinate the model's own current prediction `ŷ = argmax_i p_i`, plug it in, use
`g_x = g_x^{ŷ}`. One gradient per example, no averaging. Compute the norm, because the norm is what I'll use
as the uncertainty signal:

  ‖g_x^y‖² = (Σ_i p_i² + 1 − 2 p_y) ‖z‖².

The only `y`-dependent piece is `−2 p_y`. To *minimize* `‖g_x^y‖` over `y` I maximize `p_y`, i.e.
`argmin_y ‖g_x^y‖ = argmax_y p_y = ŷ`. So the hallucinated label gives the *smallest* possible gradient
norm — meaning `‖g_x‖ = ‖g_x^{ŷ}‖` is a *lower bound* on the gradient norm the example will really induce
once I see its true label. That's not laziness, it's a guarantee: when I pick a point because its `g_x` is
large, it will produce at least that much update — the bound only ever understates, so I can't be fooled
into thinking a confident point is informative. And the magnitude as a function of confidence: if the model
is sharp, `p ≈ e_ŷ`, then `Σ p_i² + 1 − 2 p_ŷ ≈ 0` and the embedding nearly vanishes; if `p` is flat, the
quantity is large. So `‖g_x‖` is small for confident points and large for uncertain ones — the magnitude of
this one cheap hallucinated gradient *is* a conservative uncertainty score, the same uncertainty least
confidence and BALD were reading, now living in the *direction*-bearing vector that also encodes
representation. The lazy choice is the conservative-estimate choice.

Now the batch. For every pool point I have a vector `g_x` whose *length* means uncertainty and whose
*direction* means identity/representation. I want a batch of `n` points individually long (uncertain) and
collectively spread out (diverse), with no tradeoff knob — exactly BAIT's goal, but I want it without
BAIT's matrices. The natural mathematical object for "a set that is high-quality and diverse at once" is a
determinantal point process: sample a size-`n` set with probability ∝ `det(L_Y)`, the Gram matrix of the
chosen `{g_x}`. The determinant of a Gram matrix equals the squared product of the vectors' lengths times
the squared volume they span: the length part rewards big gradients (uncertainty), the volume part is large
only when the vectors point in different directions (diversity) and collapses to zero the instant two are
parallel. So `det(L_Y)` *is* a quality-and-diversity score fused, no coefficient — and it self-adjusts to
batch size: small `n`, the volume constraint is slack so length dominates and it acts like uncertainty
sampling; large `n`, spanning the volume gets hard so diversity takes over. That is the regime-adaptive
behavior I kept failing to get by hand. But sampling from a k-DPP is genuinely expensive — high-order
polynomial, MCMC mixing problems, memory blow-up at the batch sizes I care about. Wall: the criterion is
right, I can't afford to draw from it. (BAIT's whole apparatus was one way to approximate this affordably;
it turned out too heavy for letter. I need a *different*, cheaper surrogate.)

Here an old, unrelated tool snaps into place: k-means++ seeding. Its job is to initialize Lloyd's
algorithm — pick the first center, then repeatedly sample the next with probability proportional to its
*squared distance to the nearest already-chosen center* (`D²` weighting). Look at what `D²` weighting does
in *my* gradient-embedding space. After choosing some points, the next is drawn ∝ (distance to nearest
chosen)²: a point far from everything already in the batch is much more likely picked — that's diversity,
sharpened by the squaring. And magnitude is in there too once I seed carefully: a long gradient vector
tends to be far from the others, and if I seed the very first point by its *length* rather than uniformly, I
plant the batch on the most uncertain example and let `D²` spread out from there. So k-means++ seeding in
gradient space pulls toward exactly the high-magnitude, diverse set a k-DPP would — but it's a handful of
cheap passes with no determinant, no inverse, no MCMC, no projection, and no hyperparameter. It is the
affordable surrogate, and crucially it scales gracefully in the number of classes where BAIT's projection
choked: the only thing it ever needs is *distances* between gradient embeddings.

And those distances collapse because the embeddings are outer products. For `g_a = r_a ⊗ z_a` and
`g_b = r_b ⊗ z_b`, `⟨g_a, g_b⟩ = (r_a·r_b)(z_a·z_b)`, so `‖g_a − g_b‖² = ‖r_a‖²‖z_a‖² + ‖r_b‖²‖z_b‖² −
2(r_a·r_b)(z_a·z_b)`. I never form the `k·d`-dimensional vectors: I keep `z` and `r` separately, precompute
the per-point squared norms `‖z‖²` and `‖r‖²`, and compute any distance from one `z·z` and one `r·r` dot
product. The first center is `argmax ‖r‖²‖z‖²`, the most uncertain point, exactly as designed. This is the
implementation the harness actually runs, and it is the load-bearing difference from the textbook BADGE:
the scaffold does *not* call `get_grad_embedding` to materialize the `[n, d·k]` embeddings; it calls
`self.get_embedding(..., return_probs=True)` to get `z` and `p` separately, forms the residual
`r = e_ŷ − p` (sign is irrelevant — norms and the Gram matrix are even in `r`), and runs the factored
k-means++ on the `(r, z)` pair through a `_distance` helper, sampling each new center with
`scipy.stats.rv_discrete` over the `D²` distribution. Why does this beat plain uncertainty sampling in a
case I can compute? In binary logistic regression, restricted to the margin set `w·x = 0` (the genuinely
uncertain points), the hallucinated and true-label gradients differ only by a sign, which a Gram
determinant ignores — so diverse sampling of the hallucinated gradients samples the same sets as the true
ones, and sampling those gradients *diversely* is lower-variance descent on the 0-1 loss than grabbing one
cluster of near-collinear uncertain points. The diversity makes the *uncertainty* updates better; it isn't
a separate goal bolted on. That is precisely the redundant-batch waste that limited least confidence and
BALD, fixed by construction rather than by a coefficient.

So the final rung, against the literal scaffold: where BAIT streamed and projected a rank-`k` Fisher and
ran a forward/backward greedy with pseudo-inverses, BADGE here pulls `z` and `p` in one `get_embedding`
call, forms the rank-one probability-residual `r`, and runs factored k-means++ seeding — first center by
`‖r‖²‖z‖²` (most uncertain), then `D²`-weighted draws via `rv_discrete`, all distances computed from the
separate `(r, z)` factors so it never touches `k·d` space and never inverts a matrix. Same fused
uncertainty-and-diversity property as BAIT, at uncertainty-sampling cost, with no projection to choke on
letter's 26 classes. The distilled rule and the literal scaffold fill are in the answer.

Now the falsifiable expectations against the rungs already measured, which is what this endpoint has to
clear. The decisive test is **letter**, where BAIT collapsed (0.791 / 0.671, no seed-42 run) precisely
because its heavy machinery couldn't survive 26 classes: BADGE's factored seeding has no projection and no
matrix inverse, so it should *not* collapse there, and because the fused embedding captures both the
confusable-glyph uncertainty BALD scored *and* the diversity across the 26 classes that thin-budget letter
needs, I expect BADGE to be the *strongest* rung on letter — clearly above BALD's 0.836 / 0.716 and seed-42
0.893. If BADGE does not beat BALD on letter, then either the hallucinated-gradient lower bound is too loose
on this small net or k-means++ in `(r,z)` space isn't capturing the right diversity, and the fused-embedding
premise is wrong. On **spambase** I expect BADGE to roughly match BAIT and least confidence (≈0.93 / ≈0.91)
— the balanced binary boundary leaves little for diversity to add over uncertainty, so the rungs converge.
On **splice** I expect a gain over BAIT's 0.813 / 0.756. The overall bar this endpoint must clear is to be
the best *aggregate* rung — to win letter outright (where every other rung either chased noise, stayed
redundant, or collapsed) while holding spambase and splice at the front of the pack — which would make it
the only rung that fuses uncertainty and diversity cheaply enough to survive every dataset, the thing the
whole climb was reaching for.
