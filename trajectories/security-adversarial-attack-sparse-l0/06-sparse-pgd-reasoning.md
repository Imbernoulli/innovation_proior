Sparse-RS delivered exactly the step change I predicted: `asr = 0.853` on `Rebuffi-R18-L2`, `0.947` on
`Augustin-L2`, `0.967` on `Engstrom-L2`, mean about `0.922`. So the random search that is *built for* the
discrete `L0` support went from SparseFool's `0.187` to over `0.92` — and, as I expected, it flattened the
per-model spread (the three numbers sit within `0.11` of each other, versus SparseFool's `0.17`-wide swing
from `0.113` to `0.280`), precisely because random search over feasible supports does not hinge on any one
model's boundary linearity. This is the strongest baseline, and it is genuinely strong. But the very thing
that makes it the floor for a finale is also its one structural concession: it uses *no gradient at all*,
even though the harness grants full white-box access including backprop. Look at where it still leaves
robust accuracy on the table — `Rebuffi-R18-L2` at `0.853` means roughly 22 of 150 correctly-classified
samples *survived* 10000 queries of directed random search. Those survivors are the hard cases, the ones
where the right 24-pixel support is rare enough that blind swapping, even with a provably efficient
schedule, does not stumble onto it in the budget. For exactly those cases the gradient is information
Sparse-RS is throwing away. So the natural next move is not a different search heuristic — it is to *use
the gradient to choose the support*, which is the one lever every rung either lacked (Pixle, OnePixel,
Sparse-RS are gradient-free) or used badly (JSMA greedy, SparseFool local-linear).

The obstacle is the one that sank the white-box rungs in the first place, and I have to confront it head-on:
gradient descent does not respect a *combinatorial* support. If I take a gradient step on a dense
perturbation and then project onto the `L0` ball by keeping the top-`k` coordinates by magnitude, two
things go wrong, and they are the same two that made JSMA brittle. The support lurches discontinuously
because the projection is a hard top-`k`, so the iterate never settles; and once the projection zeroes a
coordinate, *no gradient flows back to it*, so the optimizer is blind to the possibility that a
currently-unselected pixel would have been a better choice. That is why naive projected-gradient `L0`
attacks *over-report* robustness — they cannot find the support they should. Sparse-RS sidesteps this
entirely by never differentiating; the finale has to *solve* it, so the gradient can be put to work.

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

But the hard top-`k` is still non-differentiable — the same wall. The device that crosses it is
straight-through estimation: apply the hard top-`k` in the *forward* pass so the attack stays feasible, but
in the *backward* pass let the gradient to `m` flow as if through the *soft* mask `sigmoid(m)`, not the
binarized one. Then every pixel — including the ones currently outside the support — receives a gradient
telling it whether raising its mask value (entering the support) would lower the loss. That is the exact
cure for the dead-gradient failure: the support can move because the unselected coordinates are no longer
gradient-zero. There is a second, subtler routing choice for the *perturbation's* gradient: send it through
the soft mask (the "unprojected" variant, which updates `p` as if all pixels were partially active, biasing
toward exploration of new supports) or through the hard mask (the "projected" variant, which refines the
current support faithfully). They fall into different local optima, so I alternate them across restarts and
keep the best example either finds — the white-box analogue of the diversity Sparse-RS got from its
population, and the direct answer to those ~22 Rebuffi survivors that one biased descent alone would miss.

Now ground this in *this* task's edit surface, because the finale is the literal fill of the same
`run_attack` contract, not a paper harness. The harness hands me a differentiable deep copy of the model
(I may backprop through it), images in `[0,1]`, the budget `pixels = 24`, and it validates the `L0` count
channel-wise after the fact. So I parameterize `p` at full image resolution and keep `x + p` valid
*natively*, not by an end-clip that would collapse a sparse attack: after each magnitude step I clamp `p`
into `[-eps, eps]` and then into `[-x, 1-x]`, which guarantees `x + p in [0,1]` per pixel — `eps = 1`
because the `L0` model lets a chosen pixel take any value in the range. The mask `m` is `(B, 1, H, W)`, one
scalar per spatial position (matching the harness's per-pixel, channel-collapsed `L0` count); I sigmoid it
before the top-`k` so the ranking is bounded and the gradients well-scaled. The magnitude update is a PGD
sign step `p <- p - alpha*sign(grad_p)` with `alpha = 0.25*eps` (descent on the margin); the mask update is
a *normalized* gradient step `m <- m - beta*sqrt(H*W)*grad_m/||grad_m||` with `beta = 0.25`, because `m` is
a selection variable, not a bounded perturbation, and normalizing keeps the support moving at a consistent
rate. The objective is the untargeted margin `f_y - max_{r!=y} f_r`, minimized — its sign is the
misclassification certificate, and unlike cross-entropy it does not saturate as the attack nears success. I
keep a running best across all iterations and both restarts, so the returned image is the strongest
`24`-sparse example ever found. The straight-through top-`k` lives in a small custom autograd function; the
full scaffold module is in the answer.

What bar must this clear, and what would I validate? The strongest baseline is Sparse-RS at mean ASR
`0.922` — `0.853 / 0.947 / 0.967` on Rebuffi / Augustin / Engstrom. A finale that merely matched those
numbers would not justify itself, because Sparse-RS already gets them gradient-free; the claim has to be
that *using the gradient to choose the support* finds the hard examples random search misses in budget. The
falsifiable expectation is therefore concrete: sPGD should clear `0.922` mean and, most tellingly, lift the
*lowest* number — `Rebuffi-R18-L2` at `0.853`, the model where ~22 samples survived random search — because
that is exactly where a gradient-guided support search should pay off, converting survivors that have a rare
but *findable-by-gradient* 24-pixel support into successes. I would validate three things against the
Sparse-RS feedback: (1) mean ASR strictly above `0.922`; (2) the per-model minimum strictly above Rebuffi's
`0.853`, since closing the hardest model is the whole point; (3) that the gain comes from the gradient and
not from extra iterations — i.e. it holds at a comparable compute budget to the 10000-query random search.
If sPGD came back *below* Sparse-RS, the diagnosis would be that on these particular `L2`-robust surfaces
the gradient is so flattened that even a straight-through support search cannot beat directed random search
in budget, and the right move would be the published ensemble: run sPGD *and* Sparse-RS and take the best
of both, which is how the strongest reported sparse-`L0` evaluations are actually assembled. But the
straight-through, mask-decomposed gradient is the one piece of information every rung on this ladder left on
the table, and putting it to work is the natural endpoint of the climb.

The causal chain in one breath: Sparse-RS's `0.922` is the strongest baseline but uses *no gradient*,
leaving ~22 Rebuffi survivors that random search cannot find a support for in budget → use the gradient to
choose the support, which naive `L0`-PGD cannot do because the hard top-`k` projection lurches and zeroes
the gradient on unselected pixels → decompose `delta = p ⊙ binarize(m)` so the support is its own
differentiable variable and the budget holds by construction → cross the non-differentiable top-`k` with
straight-through estimation (hard forward, soft-mask backward) so unselected pixels still get signal, and
alternate the projected/unprojected perturbation-gradient routing across restarts for diversity → ground it
as the literal `run_attack` fill with native `[-x,1-x]` validity clamps, a sigmoid mask, a margin
objective, and best-tracking → expecting mean ASR above `0.922` and, decisively, the Rebuffi minimum above
`0.853`, because gradient-guided support search is built to find exactly the hard examples that gradient-
free random search misses.
