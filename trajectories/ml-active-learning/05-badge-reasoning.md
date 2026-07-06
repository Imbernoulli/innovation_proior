BAIT confirmed exactly the split I predicted, and the place it won and the place it lost together point to
what the strongest rung has to be. On **spambase** the Fisher A-optimal objective came in at 0.929 / 0.911
on the mean — above BALD's 0.905 / 0.892 by `0.024` in accuracy and edging even least confidence's 0.927 /
0.908 — and on **splice** at 0.813 / 0.756, above BALD's 0.795 / 0.737 by `0.018`. So filling the diversity
hole with a pool-aware information criterion genuinely helped where diversity was the binding constraint,
just as the derivation said: on the low-noise datasets, where BALD gave back least confidence's edge, the
A-optimal batch recovered it and then some. But on **letter** BAIT collapsed: mean 0.791 / 0.671, *below*
BALD's 0.836 / 0.716 by `0.045` and below even random's 0.816 / 0.724, and seed 42 has no letter result at
all — the run did not complete. That is the CPU adaptation biting precisely where I said it would: 26 classes
makes `d·k` largest (~3000), so the random projection to 128 dims discards the most Fisher geometry, the
entropy shortlist of 512 can't even cover all the confusable glyph pairs, the streaming full-pool Fisher
accumulation is slowest, and the time/memory budget can fail outright. So the lesson is sharp, and it is a
lesson about *machinery*, not about *objective*. The A-optimal objective is right — uncertainty fused with
pool-aware diversity beats per-point uncertainty, spambase and splice prove it — but the *apparatus* to
compute it exactly (per-class rank-`k` Fisher factors, a `d·k × d·k` pool matrix, projections,
pseudo-inverses, forward/backward greedy) is too heavy to survive this harness on the hardest dataset. I want
the same fused property — long *and* diverse batches, no tradeoff knob — but obtained *cheaply*, with no
matrix inverse, no projection, nothing that blows up when the number of classes is large. That is the
constraint the final rung has to satisfy: BAIT's geometry, BALD's robustness on letter, at
uncertainty-sampling cost.

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
Fisher BAIT had to project: `V_x` was a `(d·k)×k` factor; this is `(p − e_y) ⊗ z`, a single `d·k`-vector.
That collapse from rank `k` to rank one is the cheapness I need — it is exactly the `k−1` directions and the
`√p` weighting BAIT carried and letter's projection choked on, deliberately dropped.

But the obvious snag: `g_x^y` needs the true label `y`, which is the whole point — I don't have it. The
expensive answer is to average over `y` under the model. Let me try the cheap thing and check whether it's
defensible or just lazy: hallucinate the model's own current prediction `ŷ = argmax_i p_i`, plug it in, use
`g_x = g_x^{ŷ}`. One gradient per example, no averaging. Compute the norm, because the norm is what I'll use
as the uncertainty signal, and expand it in the label:

  ‖g_x^y‖² = ‖(p − e_y) ⊗ z‖² = ‖p − e_y‖² ‖z‖² = (Σ_i p_i² + 1 − 2 p_y) ‖z‖².

The only `y`-dependent piece is `−2 p_y`. To *minimize* `‖g_x^y‖` over `y` I maximize `p_y`, i.e.
`argmin_y ‖g_x^y‖ = argmax_y p_y = ŷ`. So the hallucinated label gives the *smallest* possible gradient
norm — meaning `‖g_x‖ = ‖g_x^{ŷ}‖` is a *lower bound* on the gradient norm the example will really induce
once I see its true label. That's not laziness, it's a guarantee: when I pick a point because its `g_x` is
large, it will produce at least that much update — the bound only ever understates, so I can't be fooled
into thinking a confident point is informative. Now verify the magnitude actually tracks confidence, with
numbers. A maximally unsure binary point `p = (0.5, 0.5)`: `Σ p_i² = 0.5`, and with `ŷ` the top class,
`0.5 + 1 − 2(0.5) = 0.5`, so `‖g_x‖² = 0.5 ‖z‖²`. A confident binary point `p = (0.99, 0.01)`: `Σ p_i² =
0.9802`, and `0.9802 + 1 − 2(0.99) = 0.0002`, so `‖g_x‖² = 0.0002 ‖z‖²` — the embedding has all but vanished.
A flat 3-class point `p = (0.34, 0.33, 0.33)`: `Σ p_i² = 0.334`, and `0.334 + 1 − 2(0.34) = 0.654`, so
`‖g_x‖² = 0.654 ‖z‖²`, large. So the length of this one cheap hallucinated gradient falls smoothly from ~`0.65`
of `‖z‖²` at maximal confusion to ~`0.0002` at confidence — it *is* a conservative uncertainty score, the same
uncertainty least confidence and BALD were reading, now living in the *direction*-bearing vector that also
encodes representation. The lazy choice is the conservative-estimate choice.

Now the batch. For every pool point I have a vector `g_x` whose *length* means uncertainty and whose
*direction* means identity/representation. I want a batch of `n` points individually long (uncertain) and
collectively spread out (diverse), with no tradeoff knob — exactly BAIT's goal, but I want it without
BAIT's matrices. The natural mathematical object for "a set that is high-quality and diverse at once" is a
determinantal point process: sample a size-`n` set with probability ∝ `det(L_Y)`, the Gram matrix of the
chosen `{g_x}`. Why does a determinant fuse quality and diversity? Take just two vectors `a, b` and write the
`2×2` Gram determinant: `det [[a·a, a·b],[b·a, b·b]] = ‖a‖²‖b‖² − (a·b)² = ‖a‖²‖b‖²(1 − cos²θ) =
‖a‖²‖b‖² sin²θ`, with `θ` the angle between them. Read it: the `‖a‖²‖b‖²` factor rewards *big* gradients
(uncertainty), and the `sin²θ` factor is large only when the vectors point in *different* directions
(diversity) and collapses to zero the instant they are parallel (`θ = 0`), no matter how long they are. The
general `det(L_Y)` is exactly this — squared product of lengths times squared volume spanned — so it is a
quality-and-diversity score fused, no coefficient. And it self-adjusts to batch size: for small `n` the volume
constraint is slack so length dominates and it acts like uncertainty sampling; for large `n` spanning the
volume gets hard so diversity takes over. That is the regime-adaptive behavior I kept failing to get by hand.
But sampling from a k-DPP is genuinely expensive — the normalizer needs an eigendecomposition of the full pool
kernel, MCMC mixing is fragile, and memory blows up at the batch sizes I care about. Wall: the criterion is
right, I can't afford to draw from it. BAIT's whole apparatus was one way to approximate this affordably; it
turned out too heavy for letter. I need a *different*, cheaper surrogate.

Here an old, unrelated tool snaps into place: k-means++ seeding. Its job is to initialize Lloyd's
algorithm — pick the first center, then repeatedly sample the next with probability proportional to its
*squared distance to the nearest already-chosen center* (`D²` weighting). Look at what `D²` weighting does
in *my* gradient-embedding space. After choosing some points, the next is drawn ∝ (distance to nearest
chosen)²: a point far from everything already in the batch is much more likely picked — that's diversity,
sharpened by the squaring. And magnitude is in there too once I seed carefully: a long gradient vector
tends to be far from the others (a big `‖g‖` makes `‖g_a − g_b‖` big), and if I seed the very first point by
its *length* rather than uniformly, I plant the batch on the most uncertain example and let `D²` spread out
from there. So k-means++ seeding in gradient space pulls toward exactly the high-magnitude, diverse set a k-DPP
would — it is a known cheap proxy for DPP sampling — but it's a handful of cheap passes with no determinant, no
inverse, no MCMC, no projection, and no hyperparameter. It is the affordable surrogate, and crucially it scales
gracefully in the number of classes where BAIT's projection choked: the only thing it ever needs is *distances*
between gradient embeddings, and distances are cheap.

There is a reason to trust this surrogate beyond "it feels DPP-like," and it is worth stating because it is why
I am comfortable dropping the determinant entirely. The `D²` sampling distribution is not an arbitrary heuristic
— it is the seeding whose expected objective is provably within an `O(log k)` factor of the optimal `k`-center
spread, so it comes with a guarantee that it covers the space rather than clumping. And there is a direct line to
the DPP I actually wanted: a determinant of the Gram matrix is a squared volume, and the incremental volume a
new vector adds to the span of the already-chosen ones is exactly its distance to that span — which `D²` (the
squared distance to the nearest chosen point) approximates greedily. So `D²` seeding is a greedy,
sampling-based stand-in for "grow the spanned volume," which is what `det(L_Y)` rewards. The self-adjustment to
batch size survives the swap intact: when `n` is small relative to the contested region, almost every candidate
is far from the few already chosen, so the `D²` weights are dominated by the raw gradient *lengths* I seeded on
and it behaves like uncertainty sampling; when `n` grows large, the chosen set already blankets the space, so
being *far* from it is what wins the draw and diversity dominates. No coefficient decides that trade — the batch
size and the geometry decide it, exactly as the k-DPP would have.

And those distances collapse because the embeddings are outer products — this is the step that makes the whole
thing fit in memory on letter. For `g_a = r_a ⊗ z_a` and `g_b = r_b ⊗ z_b`, the inner product of two outer
products factors: `⟨g_a, g_b⟩ = ⟨r_a ⊗ z_a, r_b ⊗ z_b⟩ = (r_a·r_b)(z_a·z_b)` — a product of a `k`-dimensional
dot and a `d`-dimensional dot, never a `d·k`-dimensional one. Check the identity on vectors small enough to expand by hand, because the whole memory saving rides on it. Take
`r_a = (1, 0)`, `z_a = (2, 0)`, so `g_a = r_a ⊗ z_a` flattens to `(2, 0, 0, 0)` with `‖g_a‖² = 4`; the factored
form gives `‖r_a‖²‖z_a‖² = 1 · 4 = 4`, matching. Take `r_b = (0, 1)`, `z_b = (0, 3)`, so `g_b` flattens to
`(0, 0, 0, 3)` with `‖g_b‖² = 9 = ‖r_b‖²‖z_b‖² = 1 · 9`. Their true inner product `⟨g_a, g_b⟩ = 0` (disjoint
support), and the factored form `(r_a·r_b)(z_a·z_b) = (0)(0) = 0` agrees; the true squared distance is
`4 + 9 = 13`, and the factored `‖r_a‖²‖z_a‖² + ‖r_b‖²‖z_b‖² − 2·0 = 13` agrees. The identity holds, so the
squared distance is `‖g_a − g_b‖² = ‖r_a‖²‖z_a‖² + ‖r_b‖²‖z_b‖² − 2(r_a·r_b)(z_a·z_b)`, and I never form the
`d·k`-vectors: I keep
`z` and `r` separately, precompute the per-point squared norms `‖z‖²` and `‖r‖²`, and compute any distance from
one `z·z` dot and one `r·r` dot. For letter that means I work with a `d`-vector and a `26`-vector per point
rather than a `~3000`-vector — the exact ambient dimension BAIT had to random-project away, here never
materialized at all. The first center is `argmax ‖r‖²‖z‖²`, the most uncertain point, exactly as the norm
computation above says. This is the implementation the harness actually runs, and it is the load-bearing
difference from a materialized gradient-embedding BADGE: the scaffold does *not* call `get_grad_embedding` to
build the `[n, d·k]` embeddings; it calls `self.get_embedding(..., return_probs=True)` to get `z` and `p`
separately, forms the residual `r = e_ŷ − p` (sign is irrelevant — the norms and the Gram matrix are even in
`r`, since `(−r)·(−r) = r·r`), and runs the factored k-means++ on the `(r, z)` pair through a `_distance`
helper, sampling each new center with `scipy.stats.rv_discrete` over the `D²` distribution and clipping any
FP-negative `dist²` at zero. Why does this beat plain uncertainty sampling in a case I can reason through? In
binary logistic regression, restricted to the margin set `w·x = 0` (the genuinely uncertain points), the
hallucinated and true-label gradients differ only by a sign, which a Gram determinant ignores — so diverse
sampling of the hallucinated gradients samples the same sets as the true ones, and sampling those gradients
*diversely* is lower-variance descent on the 0-1 loss than grabbing one cluster of near-collinear uncertain
points. The diversity makes the *uncertainty* updates better; it isn't a separate goal bolted on. Trace the fix on the
exact failure it targets: suppose two unlabeled points `a`, `b` are near-duplicates — same posterior
`p = (0.5, 0.5)` and nearly identical embeddings `z_a ≈ z_b` — sitting on the same stretch of boundary. Least
confidence scores both at `1 − 0.5 = 0.5`, ties them at the top, and buys *both*, learning one point's worth for
two labels: the redundant batch that inverted its letter mean. In `(r, z)` space these two have `g_a ≈ g_b`, so
`‖g_a − g_b‖² ≈ 0`; once k-means++ seeds on `a`, the `D²` weight for `b` is essentially zero and `b` is almost
never drawn — the batch spends its second label somewhere genuinely different. That is precisely the
redundant-batch waste that limited least confidence and BALD, fixed by construction rather than by a coefficient — and unlike BAIT it costs a few distance passes, not a stream-project-invert pipeline.

So the final rung, against the literal scaffold: where BAIT streamed and projected a rank-`k` Fisher and
ran a forward/backward greedy with pseudo-inverses, BADGE here pulls `z` and `p` in one `get_embedding`
call, forms the rank-one probability-residual `r`, and runs factored k-means++ seeding — first center by
`‖r‖²‖z‖²` (most uncertain), then `D²`-weighted draws via `rv_discrete`, all distances computed from the
separate `(r, z)` factors so it never touches `d·k` space and never inverts a matrix. Same fused
uncertainty-and-diversity property as BAIT, at uncertainty-sampling cost, with no projection to choke on
letter's 26 classes. The distilled rule and the literal scaffold fill are in the answer.

Now the falsifiable expectations against the rungs already measured, which is what this endpoint has to
clear. The decisive test is **letter**, where BAIT collapsed (0.791 / 0.671, no seed-42 run) precisely
because its heavy machinery couldn't survive 26 classes: BADGE's factored seeding has no projection and no
matrix inverse — the `d·k` space that BAIT projected and letter choked on is never even formed here — so it
should *not* collapse there, and because the fused embedding captures both the confusable-glyph uncertainty
BALD scored (in `r`) *and* the diversity across the 26 classes that thin-budget letter needs (in `z`), I
expect BADGE to be the *strongest* rung on letter — clearly above BALD's 0.836 / 0.716 and seed-42 0.893. If
BADGE does not beat BALD on letter, then either the hallucinated-gradient lower bound is too loose on this
small net (the residuals `r` too flat to separate points) or k-means++ in `(r, z)` space isn't capturing the
right diversity, and the fused-embedding premise is wrong. On **spambase** I expect BADGE to roughly match
BAIT and least confidence (≈0.93 / ≈0.91) — the balanced binary boundary leaves little for diversity to add
over uncertainty, so the rungs converge, and I would not expect BADGE to *beat* BAIT's 0.929 / 0.911 by much
if at all. On **splice** I expect a gain over BAIT's 0.813 / 0.756. Before I set the bar, let me read the
current leaderboard honestly across the four measured rungs, per dataset, because it tells me exactly which
records BADGE has to break. On letter accuracy the record is BALD's 0.8357; on spambase it is BAIT's 0.929 /
0.911; on splice it is a near tie between least confidence's 0.814 accuracy and BAIT's 0.813 / 0.756. But the
striking gap is letter *auc*: random still holds it at 0.724, and *no adaptive rung has beaten it* — least
confidence 0.668, BALD 0.716, BAIT 0.671 all sit below the blind draw. Four rungs in, the passive floor is
still the best sample-efficiency on the hardest dataset, because every adaptive rule so far either chased noise
(least confidence), left the batch redundant (BALD), or collapsed under its own machinery (BAIT). So the sharp,
falsifiable framing for this endpoint is: BADGE has to be the *first* rule to beat random's 0.724 letter auc
while also taking BALD's letter accuracy and holding BAIT's spambase/splice — no prior rung managed even the
first of those. The overall bar this endpoint must clear is therefore to be the best *aggregate* rung — to win
letter outright (where every other rung either chased noise, stayed redundant, or collapsed) while holding
spambase and splice at the front of the pack — which would make it the only rung that fuses uncertainty and
diversity cheaply enough to survive every dataset, the thing the whole climb was reaching for.
