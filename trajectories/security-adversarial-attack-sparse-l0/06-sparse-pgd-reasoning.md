Sparse-RS delivered exactly the step change I predicted: `asr = 0.853` on `Rebuffi-R18-L2`, `0.947` on
`Augustin-L2`, `0.967` on `Engstrom-L2`, mean about `0.922`. So the random search that is *built for* the
discrete `L0` support went from SparseFool's `0.187` to over `0.92` — a nearly fivefold jump in the mean —
and, as I expected, it flattened the per-model spread: the three numbers now sit within
`0.967 - 0.853 = 0.114` of each other, versus SparseFool's `0.280 - 0.113 = 0.167`-wide swing, precisely
because random search over feasible supports does not hinge on any one model's boundary linearity. Let me
convert the three ASRs into survivor counts, because the survivors are what a finale has to attack and the
raw fractions hide them. Out of the up-to-`150` correctly-classified samples per model, the robust
accuracy `1 - asr` leaves `(1 - 0.853) * 150 ~= 22` survivors on `Rebuffi`, `(1 - 0.947) * 150 ~= 8` on
`Augustin`, and `(1 - 0.967) * 150 ~= 5` on `Engstrom`. So the entire remaining attack surface is about
`22 + 8 + 5 = 35` images across all three models, and `22` of them — nearly two-thirds — are on `Rebuffi`
alone. This is the strongest baseline, and it is genuinely strong. But the very thing that makes it the
floor for a finale is also its one structural concession: it uses *no gradient at all*, even though the
harness grants full white-box access including backprop. Those `22` Rebuffi survivors are the hard cases,
the ones where the right 24-pixel support is rare enough that blind swapping, even with a provably
efficient schedule, does not stumble onto it in `10000` queries. For exactly those cases the gradient is
information Sparse-RS is throwing away. So the natural next move is not a different search heuristic — it is
to *use the gradient to choose the support*, which is the one lever every rung either lacked (Pixle,
OnePixel, Sparse-RS are gradient-free) or used badly (JSMA greedy, SparseFool local-linear).

Before I build the gradient method, I should ask whether the cheaper move — just giving Sparse-RS more
queries — would close those `22` Rebuffi survivors, because if it would, a white-box finale is
unjustified. The coupon-collector bound says the random search needs on the order of `(d-k)k(ln k+2)/(m-k)`
queries to hit the `k`-among-`2k` relaxation, and it already spends `~5180` of its `10000` budget getting
there for the typical hard image. The survivors are the images where even that is not enough — where the
working 24-pixel support is *rarer* than the relaxation assumes, so the effective `m` (the count of
coordinates that would work) is small and `m - k` shrinks, blowing the bound up toward and past `10000`.
Doubling the query budget to `20000` only buys a linear factor against a bound that is exploding as
`1/(m-k)` for these specific hard images, so it would recover a handful of the `22` at best and at real
compute cost. The survivors are not "random search needs a bit longer" cases; they are "the good support is
rare enough that *sampling* will not find it" cases. That is exactly the regime where *directed* search —
following the gradient to the rare support instead of sampling for it — should win, and it is the concrete
reason the finale goes white-box rather than simply scaling the random search.

The obstacle is the one that sank the white-box rungs in the first place, and I have to confront it head-on:
gradient descent does not respect a *combinatorial* support. If I take a gradient step on a dense
perturbation and then project onto the `L0` ball by keeping the top-`k` coordinates by magnitude, two
things go wrong, and they are the same two that made JSMA brittle. The support lurches discontinuously
because the projection is a hard top-`k`, so the iterate never settles; and once the projection zeroes a
coordinate, *no gradient flows back to it*, so the optimizer is blind to the possibility that a
currently-unselected pixel would have been a better choice. Let me be concrete about that dead gradient,
because it is the crux. If the perturbation is `delta` and I zero all but the top-`k` coordinates, then for
any unselected coordinate `i` the forward pass literally does not depend on `delta_i` — the model never
sees it — so `d loss / d delta_i = 0` identically, not small but *exactly* zero. The optimizer therefore
receives no signal whatsoever about whether pixel `i` should have been in the support; it can only reshuffle
magnitude among the `k` already-selected pixels. That is why naive projected-gradient `L0` attacks
*over-report* robustness — they cannot find the support they should, because the support-selection gradient
has been annihilated by the projection. Sparse-RS sidesteps this entirely by never differentiating; the
finale has to *solve* it, so the gradient can be put to work.

The solution is to stop treating the perturbation as one dense vector I project, and instead make the
support its own differentiable variable. Decompose the perturbation as `delta = p ⊙ binarize(m)`: a dense
**magnitude tensor** `p` (the perturbation values, image-shaped) times the **binarized** version of a
continuous **mask** `m` with one scalar per spatial pixel. The binarization is a hard top-`k` over the
mask — keep the `k` pixels with the largest mask values, zero the rest — so exactly `k` pixels survive
every forward pass and the `L0` budget holds *by construction*, no projection of the perturbation required.
Now the two sub-problems are explicit and each has its own gradient: the gradient on `p` says how to push
the values on the chosen pixels, and the gradient on `m` says *which pixels to choose*. The support is no
longer an implicit byproduct of magnitude ranking; it is a thing the optimizer steers directly.

Why a mask `m` separate from the magnitudes `p`, rather than just ranking `p` itself by magnitude (which is
what PGD0 does)? Because ranking `p` ties the support choice to the *current* magnitudes: a pixel only
enters the support if its perturbation is already large, but its perturbation only grows if it is already in
the support, a chicken-and-egg that the discontinuous projection resolves arbitrarily. A dedicated mask
breaks the cycle — `m` can grow for a pixel whose magnitude is still small, *pulling* it into the support on
the strength of its gradient alone, and only then does `p` start optimizing its value. The two variables
let the attack decide *where* before it has decided *how much*, which is exactly the ordering a sparse
attack wants and the one PGD0 cannot express.

But the hard top-`k` is still non-differentiable — the same wall. Let me weigh the ways across it, because
there are three and the choice is not arbitrary. I could use a Gumbel-softmax or other continuous
relaxation of the top-`k`, keeping a *soft* mask throughout — but then the forward pass perturbs more than
`k` pixels (every pixel is partially on), so the attack is not actually `L0`-feasible until I harden it at
the end, reintroducing the projection lurch at the finish. I could use a score-function / REINFORCE
estimator that samples discrete supports and reweights by reward — but that estimator is high-variance in a
`1024`-dimensional selection space and would need many samples per step, throwing away the query efficiency
that is the whole point of going white-box. Or I could use straight-through estimation: apply the hard
top-`k` in the *forward* pass so the attack stays exactly `k`-feasible, but in the *backward* pass let the
gradient to `m` flow as if through the *soft* mask `sigmoid(m)`, not the binarized one. Straight-through is
the right one: it keeps every forward pass feasible (unlike the soft relaxation) and it is low-variance and
deterministic (unlike REINFORCE), at the cost of a *biased* gradient — the backward pass does not match the
forward exactly. That bias is acceptable here because I only need the mask gradient to point roughly toward
better supports, not to be exact; the top-`k` re-hardens every forward pass regardless. So every pixel —
including the ones currently outside the support — receives a gradient telling it whether raising its mask
value (entering the support) would lower the loss. That is the exact cure for the dead-gradient failure:
the support can move because the unselected coordinates are no longer gradient-zero. There is a second,
subtler routing choice for the *perturbation's* gradient: send it through the soft mask (the "unprojected"
variant, which updates `p` as if all pixels were partially active, biasing toward exploration of new
supports) or through the hard mask (the "projected" variant, which refines the current support faithfully).
They fall into different local optima, so I alternate them across restarts and keep the best example either
finds — the white-box analogue of the diversity Sparse-RS got from its population, and the direct answer to
those ~22 Rebuffi survivors that one biased descent alone would miss.

Now ground this in *this* task's edit surface, because the finale is the literal fill of the same
`run_attack` contract, not a paper harness. The harness hands me a differentiable deep copy of the model
(I may backprop through it), images in `[0,1]`, the budget `pixels = 24`, and it validates the `L0` count
channel-wise after the fact. So I parameterize `p` at full image resolution `(B, C, H, W) = (B, 3, 32, 32)`
and keep `x + p` valid *natively*, not by an end-clip that would collapse a sparse attack: after each
magnitude step I clamp `p` into `[-eps, eps]` and then into `[-x, 1-x]`, which guarantees `x + p in [0,1]`
per pixel — `eps = 1` because the `L0` model lets a chosen pixel take any value in the range, so the
magnitude bound is the full width of the box. Let me check the mask shape against the harness's counting
rule, because a mismatch here would silently over-count pixels. The harness collapses channels — a spatial
pixel is "changed" iff *any* of its three channels moved — so the support is a property of the `(H, W)`
grid, not of individual channels. I therefore make the mask `(B, 1, H, W)`, one scalar per spatial
position, and the top-`k` selects `k = 24` *spatial* pixels whose all three channels are then free to move
together. If I had instead masked per channel, `(B, 3, H, W)`, the top-`24` could pick `24` individual
channels spread across more than `24` spatial pixels, over-counting against the harness's rule and risking
rejection; the `(B, 1, H, W)` mask is what matches the validated count exactly. I sigmoid the mask before
the top-`k` so the ranking is bounded and the gradients well-scaled. The magnitude update is a PGD sign
step `p <- p - alpha*sign(grad_p)` with `alpha = 0.25*eps = 0.25` (descent on the margin); the mask update
is a *normalized* gradient step `m <- m - beta*sqrt(H*W)*grad_m/||grad_m||` with `beta = 0.25`, so the mask
step size is `beta*sqrt(H*W) = 0.25*sqrt(1024) = 0.25*32 = 8`. That factor of `sqrt(H*W)` is not a fudge:
`m` is a selection variable of dimension `H*W`, and a raw unit-norm step would move each of the `1024`
mask entries by only `~1/sqrt(1024)` on average, far too small to reorder the top-`k`; scaling by
`sqrt(H*W)` restores an order-one effective step per entry so the support actually moves at a consistent
rate. The objective is the untargeted margin `f_y - max_{r!=y} f_r`, minimized — its sign is the
misclassification certificate, and unlike cross-entropy it does not saturate as the attack nears success. I
keep a running best across all iterations and both restarts, so the returned image is the strongest
`24`-sparse example ever found. The straight-through top-`k` lives in a small custom autograd function; the
full scaffold module is in the answer.

The clamp *order* is load-bearing and worth tracing, because getting it wrong reintroduces the validity
collapse that killed `L1`-DeepFool. After a magnitude step I first clamp `p` into `[-eps, eps] = [-1, 1]`,
then into `[-x, 1-x]`. The two clamps enforce different things: `[-1, 1]` caps the raw magnitude (harmless
here since `eps = 1` already spans the box, but it keeps the variable bounded across steps), and
`[-x, 1-x]` is the one that guarantees `x + p in [0, 1]` per pixel — because if `p_i >= -x_i` then
`x_i + p_i >= 0`, and if `p_i <= 1 - x_i` then `x_i + p_i <= 1`. Doing `[-x, 1-x]` *last* is what makes the
returned image valid by construction, so no end-clip is ever needed and no high-magnitude evidence is
stripped after the fact — the perturbation is *born* inside the box each step rather than clipped into it at
the end. This is the same "box inside the optimization" principle SparseFool needed, applied per-pixel to a
gradient attack, and it is why sPGD does not suffer the fooling-rate collapse an end-clipped `L0`-PGD would.

Let me trace the mask gradient once to convince myself an unselected pixel really can enter the support,
since that is the entire mechanism and I want it on paper rather than asserted. Take a pixel `i` currently
*outside* the support: in the forward pass `binarize(m)_i = 0`, so it contributes nothing and `x + delta`
does not depend on `p_i`. Under a naive projection its gradient would be zero and it would be stuck out
forever. Under straight-through, the backward pass treats the mask as the *soft* `sigmoid(m)_i`, so the
gradient `d loss / d m_i` is computed as if pixel `i` were partially active with magnitude `p_i`,
`grad_mask_i = (grad_out * p_i).sum over channels`. If turning pixel `i` on would lower the margin — if
`p_i` points in a loss-decreasing direction — that gradient is negative, the mask update `m <- m - step*grad`
*raises* `m_i`, and after enough steps `m_i` climbs past the `k`-th largest mask value and pixel `i` enters
the top-`k`, joining the support. So the support genuinely moves: a pixel with zero forward contribution
still accumulates mask signal and can be pulled in on the strength of its would-be gradient alone. That is
the dead-gradient cure working end to end, and it is exactly what the ~22 Rebuffi survivors need — a way
for the optimizer to *discover* a rare good pixel that random swapping never sampled.

Why `n_restarts = 2` and `t = 300` specifically, rather than one long run or many short ones? The two
restarts are not for random reinitialization diversity in the Sparse-RS sense — they route the
*perturbation* gradient two different ways, unprojected (explore) on the even restart and projected
(refine) on the odd one. Those two routings are genuinely different optimizers that converge to different
supports, so running exactly one of each and keeping the better result covers both the exploratory and the
refining basin without spending compute on redundant identical runs; a third restart would have to repeat
one of the two routings, buying only reinitialization noise. And `t = 300` is enough for the mask to
reorder its top-`k` several times over — each mask step of effective size `8` can move a handful of pixels
in or out, so `300` steps allow on the order of the full support to turn over multiple times, which is what
a hard example needs to migrate from a random initial support to a working one. So the `2 x 300` budget is
sized to give each of the two distinct routings enough iterations to actually relocate the support, and no
more.

Let me also account for the compute, because the finale's claim has to be that it beats Sparse-RS *using
the gradient*, not by outspending it. The configuration is `t = 300` iterations times `n_restarts = 2`
routings, so `600` forward-plus-backward passes per image. Sparse-RS spent up to `10000` forward passes per
image (forward-only, no backward). A forward-plus-backward is roughly two-to-three times the cost of a
forward, so `600` gradient steps is on the order of `1200`-`1800` forward-equivalents — comfortably *fewer*
than Sparse-RS's `10000`. That matters for the falsifiable claim: if sPGD beats the strongest baseline, it
does so at a *lower* query-equivalent budget, which is the signature of actually using the gradient
information rather than brute-forcing more evaluations. If it needed far more compute to edge ahead, the
result would be uninteresting.

What bar must this clear, and what would I validate? The strongest baseline is Sparse-RS at mean ASR
`0.922` — `0.853 / 0.947 / 0.967` on Rebuffi / Augustin / Engstrom. A finale that merely matched those
numbers would not justify itself, because Sparse-RS already gets them gradient-free; the claim has to be
that *using the gradient to choose the support* finds the hard examples random search misses in budget. The
falsifiable expectation is therefore concrete: sPGD should clear `0.922` mean and, most tellingly, lift the
*lowest* number — `Rebuffi-R18-L2` at `0.853`, the model where `~22` samples survived random search —
because that is exactly where a gradient-guided support search should pay off, converting survivors that
have a rare but *findable-by-gradient* 24-pixel support into successes. I would validate three things
against the Sparse-RS feedback: (1) mean ASR strictly above `0.922`; (2) the per-model minimum strictly
above Rebuffi's `0.853`, since closing the hardest model is the whole point and the `22` Rebuffi survivors
are the bulk of the remaining surface; (3) that the gain comes from the gradient and not from extra
iterations — i.e. it holds at the comparable-or-lower compute I just costed against the 10000-query random
search. If sPGD came back *below* Sparse-RS, the diagnosis would be that on these particular `L2`-robust
surfaces the gradient is so flattened that even a straight-through support search cannot beat directed
random search in budget, and the right move would be the ensemble: run sPGD *and* Sparse-RS and take the
best of both, which is how the strongest reported sparse-`L0` evaluations are actually assembled. But the
straight-through, mask-decomposed gradient is the one piece of information every rung on this ladder left on
the table, and putting it to work is the natural endpoint of the climb.

The causal chain in one breath: Sparse-RS's `0.922` is the strongest baseline but uses *no gradient*,
leaving ~22 Rebuffi survivors that random search cannot find a support for in budget → use the gradient to
choose the support, which naive `L0`-PGD cannot do because the hard top-`k` projection lurches and zeroes
the gradient on unselected pixels → decompose `delta = p ⊙ binarize(m)` so the support is its own
differentiable variable and the budget holds by construction → cross the non-differentiable top-`k` with
straight-through estimation (hard forward, soft-mask backward) so unselected pixels still get signal, and
alternate the projected/unprojected perturbation-gradient routing across restarts for diversity → ground it
as the literal `run_attack` fill with native `[-x,1-x]` validity clamps, a `(B,1,H,W)` sigmoid mask, a
margin objective, and best-tracking → expecting mean ASR above `0.922` and, decisively, the Rebuffi minimum
above `0.853`, because gradient-guided support search is built to find exactly the hard examples that
gradient-free random search misses.
