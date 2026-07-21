Sparse-RS delivered the step change I predicted: `asr = 0.853` on `Rebuffi-R18-L2`, `0.947` on `Augustin-L2`,
`0.967` on `Engstrom-L2`, mean `~0.922`. So the random search *built for* the discrete `L0` support went from
SparseFool's `0.187` to over `0.92` — nearly fivefold in the mean — and it flattened the per-model spread:
the three numbers sit within `0.114` of each other, versus SparseFool's `0.167`-wide swing, because random
search over feasible supports does not hinge on any one model's boundary linearity. Convert the ASRs to
survivor counts, since the survivors are what a finale must attack: robust accuracy `1 - asr` leaves
`(1 - 0.853)*150 ~= 22` survivors on Rebuffi, `~8` on Augustin, `~5` on Engstrom — about `35` images across
all three, and `22` of them, nearly two-thirds, on Rebuffi alone. This is a genuinely strong baseline. But
its one structural concession is that it uses *no gradient at all*, even though the harness grants full
white-box access. Those 22 Rebuffi survivors are the hard cases where the right 24-pixel support is rare
enough that blind swapping, even with a provably efficient schedule, does not stumble onto it in 10000
queries — exactly the cases where the gradient is information Sparse-RS throws away. So the next move is not a
different search heuristic; it is to *use the gradient to choose the support*, the one lever every rung either
lacked (Pixle, OnePixel, Sparse-RS gradient-free) or used badly (JSMA greedy, SparseFool local-linear).

Before building it, would the cheaper move — more Sparse-RS queries — close those 22 survivors? The
coupon-collector bound needs `~(d-k)k(ln k+2)/(m-k)` queries and already spends `~5180` of its 10000 getting
to the `k`-among-`2k` relaxation for a typical hard image. The survivors are images where even that is not
enough — where the working support is *rarer* than the relaxation assumes, so the count of coordinates that
would work is small, `m-k` shrinks, and the bound explodes as `1/(m-k)` past 10000. Doubling to 20000 buys a
linear factor against a bound exploding for these specific images, recovering a handful at best and at real
compute cost. These are not "needs a bit longer" cases; they are "the good support is rare enough that
*sampling* will not find it" cases — the regime where *directed* search, following the gradient to the rare
support instead of sampling for it, should win. That is why the finale goes white-box rather than scaling the
random search.

The obstacle is the one that sank the white-box rungs: gradient descent does not respect a *combinatorial*
support. Take a gradient step on a dense perturbation, then project onto the `L0` ball by keeping the top-`k`
coordinates by magnitude, and two things go wrong — the same two that made JSMA brittle. The support lurches
discontinuously because the projection is a hard top-`k`, so the iterate never settles; and once the
projection zeroes a coordinate, *no gradient flows back to it*. Concretely, if `delta` is the perturbation
and I zero all but the top-`k`, then for any unselected coordinate `i` the forward pass literally does not
depend on `delta_i`, so `d loss / d delta_i = 0` *exactly* — not small, zero. The optimizer receives no
signal about whether pixel `i` should have been in the support; it can only reshuffle magnitude among the `k`
already selected. That is why naive projected-gradient `L0` attacks *over-report* robustness: the
support-selection gradient is annihilated by the projection. Sparse-RS sidesteps this by never
differentiating; the finale has to *solve* it.

The solution: stop treating the perturbation as one dense vector I project, and make the support its own
differentiable variable. Decompose `delta = p ⊙ binarize(m)` — a dense image-shaped **magnitude tensor** `p`
times the **binarized** version of a continuous **mask** `m` with one scalar per spatial pixel. The
binarization is a hard top-`k` over the mask, so exactly `k` pixels survive every forward pass and the `L0`
budget holds *by construction*, no projection of the perturbation required. Now each sub-problem has its own
gradient: the gradient on `p` says how to push the values on chosen pixels, the gradient on `m` says *which*
pixels to choose. The support is no longer an implicit byproduct of magnitude ranking; it is steered
directly. Why a mask separate from the magnitudes, rather than ranking `p` itself (what PGD0 does)? Ranking
`p` ties the support to the *current* magnitudes — a pixel enters only if its perturbation is already large,
but its perturbation only grows if it is already in the support, a chicken-and-egg the discontinuous
projection resolves arbitrarily. A dedicated mask breaks the cycle: `m` can grow for a pixel whose magnitude
is still small, *pulling* it into the support on its gradient alone, and only then does `p` optimize its
value. The attack decides *where* before *how much* — the ordering a sparse attack wants and PGD0 cannot
express.

But the hard top-`k` is still non-differentiable. Three ways across it. A Gumbel-softmax or other continuous
relaxation keeps a *soft* mask, but then the forward pass perturbs more than `k` pixels and the attack is not
`L0`-feasible until I harden at the end, reintroducing the projection lurch at the finish. A score-function /
REINFORCE estimator samples discrete supports and reweights, but it is high-variance in a 1024-dim selection
space and needs many samples per step, throwing away the query efficiency that is the whole point of going
white-box. Straight-through estimation is the right one: apply the hard top-`k` in the *forward* pass so the
attack stays exactly `k`-feasible, but in the *backward* pass let the gradient to `m` flow as if through the
*soft* mask `sigmoid(m)`. It keeps every forward pass feasible (unlike the soft relaxation) and is
low-variance and deterministic (unlike REINFORCE), at the cost of a *biased* gradient — acceptable because I
only need the mask gradient to point roughly toward better supports, not to be exact; the top-`k` re-hardens
every forward pass regardless. So every pixel, including those currently outside the support, receives a
gradient telling it whether raising its mask value would lower the loss — the exact cure for the
dead-gradient failure. Concretely, for a pixel `i` outside the support the forward `binarize(m)_i = 0`, but
under straight-through the backward treats it as `sigmoid(m)_i`, so `grad_mask_i = (grad_out * p_i)` summed
over channels is computed as if `i` were partially active; if turning it on would lower the margin that
gradient is negative, the update raises `m_i`, and after enough steps `m_i` climbs past the `k`-th-largest
mask value and pixel `i` joins the top-`k`. A pixel with zero forward contribution still accumulates mask
signal and can be pulled in — exactly what the 22 Rebuffi survivors need, a way to *discover* a rare good
pixel random swapping never sampled. There is a second, subtler routing choice for the *perturbation's*
gradient: send it through the soft mask (the "unprojected" variant, updating `p` as if all pixels were
partially active, biasing toward exploring new supports) or through the hard mask (the "projected" variant,
refining the current support faithfully). They fall into different local optima, so I alternate them across
restarts and keep the best — the white-box analogue of the diversity Sparse-RS got from its population.

Grounding this in the task's edit surface. The harness hands me a differentiable deep copy of the model,
images in `[0,1]`, the budget `pixels = 24`, and validates the `L0` count channel-wise afterward. I
parameterize `p` at full resolution `(B, 3, 32, 32)` and keep `x + p` valid *natively*: after each magnitude
step I clamp `p` into `[-eps, eps]` and then into `[-x, 1-x]`, guaranteeing `x + p in [0,1]` per pixel —
`eps = 1` because the `L0` model lets a chosen pixel take any value, so the magnitude bound is the full box
width. The clamp *order* is load-bearing: doing `[-x, 1-x]` *last* is what makes the returned image valid by
construction, so no end-clip is ever needed and no high-magnitude evidence is stripped after the fact — the
same "box inside the optimization" principle SparseFool needed, applied per-pixel, which is why sPGD does not
suffer the fooling-rate collapse an end-clipped `L0`-PGD would. The mask shape must match the harness's
counting rule: it collapses channels — a pixel is changed iff any of its three channels moved — so the
support is a property of the `(H, W)` grid. I make the mask `(B, 1, H, W)`, one scalar per spatial position,
and the top-`k` selects `k = 24` *spatial* pixels whose three channels are then free together; a per-channel
`(B, 3, H, W)` mask could pick 24 individual channels spread across more than 24 spatial pixels,
over-counting against the harness's rule and risking rejection. I sigmoid the mask before the top-`k` so the
ranking is bounded. The magnitude update is a PGD sign step `p <- p - alpha*sign(grad_p)` with
`alpha = 0.25*eps = 0.25`; the mask update is a *normalized* gradient step
`m <- m - beta*sqrt(H*W)*grad_m/||grad_m||` with `beta = 0.25`, so the mask step is `0.25*sqrt(1024) = 8`.
That `sqrt(H*W)` is not a fudge: `m` is a selection variable of dimension `H*W = 1024`, and a raw unit-norm
step would move each entry by only `~1/32`, far too small to reorder the top-`k`; scaling restores an
order-one effective step per entry so the support actually moves. The objective is the untargeted margin
`f_y - max_{r!=y} f_r`, minimized — its sign is the certificate and unlike cross-entropy it does not saturate
near success — and I keep a running best across all iterations and both restarts.

Why `n_restarts = 2` and `t = 300`? The two restarts are not random-reinit diversity in the Sparse-RS sense —
they route the perturbation gradient the two different ways, unprojected (explore) on the even restart,
projected (refine) on the odd. Those are genuinely different optimizers converging to different supports, so
running exactly one of each and keeping the better covers both basins without redundant identical runs; a
third would repeat a routing for only reinit noise. `t = 300` lets the mask reorder its top-`k` several times
over — each mask step of effective size 8 moves a handful of pixels in or out, so 300 steps let the full
support turn over multiple times, what a hard example needs to migrate from a random initial support to a
working one. And the compute has to be *lower* than Sparse-RS's for the claim to be "beats it *using* the
gradient" rather than by outspending: `t = 300` times `n_restarts = 2` is 600 forward-plus-backward passes
per image, and at roughly two-to-three forward-equivalents each that is `~1200`-`1800` forward-equivalents —
comfortably fewer than Sparse-RS's 10000 forward passes. If it needed far more compute to edge ahead, the
result would be uninteresting.

The bar is Sparse-RS at mean `0.922` — `0.853/0.947/0.967`. Merely matching those would not justify the
finale, since Sparse-RS gets them gradient-free; the claim is that *using the gradient to choose the support*
finds the hard examples random search misses in budget. So the falsifiable expectation is concrete: sPGD
should clear `0.922` mean and, most tellingly, lift the *lowest* number — Rebuffi at `0.853`, where ~22
samples survived random search — because that is exactly where a gradient-guided support search should pay
off, converting survivors that have a rare but *findable-by-gradient* support into successes, at the
comparable-or-lower compute just costed. If sPGD instead came back *below* Sparse-RS, the diagnosis would be
that on these `L2`-robust surfaces the gradient is so flattened that even a straight-through support search
cannot beat directed random search in budget, and the right move would be the ensemble: run sPGD *and*
Sparse-RS and take the best of both, which is how the strongest sparse-`L0` evaluations are actually
assembled. But the straight-through, mask-decomposed gradient is the one piece of information every rung on
this ladder left on the table, and putting it to work is the natural endpoint of the climb.
