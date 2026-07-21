The flat profile came back at exactly `2.000000`, identically at `N = 50`, `600`, and `30000` — the refinement
invariance I expected — and the harness reproduced the published record `1.5028628969` on the record sequence,
so the evaluator agrees with the literature at both ends of the range. The machine is trustworthy and the flat
value is a genuine ceiling. The obstruction it leaves is the whole content of this rung: the flat vector is a
stationary point of `R` where the peak term and the mass-normalization term cancel, so its gradient is
identically zero, and it is a stationary point of *maximum* type — a hilltop. A pure gradient method dropped
here does not move at all, which rules out "just run gradient descent from flat" before I waste effort on it.

But zero gradient does not mean trapped, and getting this right is what selects the search. A single-coordinate
perturbation moves `R` only at second order, and that move is *downward* (at `N = 4`, `R − 2 = −2ε^2/16`,
negative for interior or end). So flat is a hilltop with downhill in every direction; leaving it is easy. The
difficulty is what happens *after* I step off: `a*a` is bilinear and the `max` over nodes is non-smooth, so the
landscape below the summit is riddled with genuine local minima. A greedy descender rolls off the hilltop and
parks at the first poor basin. The requirement on the search is therefore not "help me leave flat" but "help me
not get stuck at the first basin below flat" — and that is exactly what simulated annealing provides. Greedy
hill-descent I have ruled out; a population method like CMA-ES would work on `50` dimensions but spends its
budget on a few expensive covariance updates when the objective is a cheap FFT and the natural move is a simple
coordinatewise rescale — I would rather spend the same budget on tens of thousands of cheap proposals. Annealing
is the minimal thing that does the one job: propose a kick to one height, accept if `R` drops, accept anyway
with Metropolis probability `exp(−ΔR/T)` if it rises, cooling `T` geometrically. Hot early so the walk wanders
across the many basins below the summit without committing; cold late so it settles into the deepest it found.

Scale comes first because it constrains everything after. A high-dimensional search lets a blind local search
wander a long time before finding the narrow good shapes, so the move is to find the *shape* at low resolution
and defer the fine grid. The rung-1 feedback gives a concrete landing near `N = 50`: with
`R = [2N/(2N−1)]·(peak/mean)` the prefactor is `1.0101` at `N = 50`, within a percent of `1`, so the reported
number tracks the underlying `peak/mean` rather than being inflated as at tiny `N`; and `50` heights (only `99`
autoconvolution nodes) is short enough that tens of thousands of proposals revisit every coordinate hundreds of
times. Long enough to hold real internal structure, short enough to canvas thoroughly.

Now the perturbation, and this comes straight from the geometry. `R` is scale-free — multiplying every height
by a constant leaves it unchanged — so the meaningful moves are *relative*. The good profiles will also span a
wide dynamic range (I suspect heavier toward the ends, lighter in the middle, the arrangement that starves the
central self-overlap), so a single additive Gaussian kick would be far too coarse for the small heights and too
timid for the large. I make the kick *multiplicative*: `a_j ← max(0, a_j·(1 + s·ξ))` with `ξ` standard normal,
clipping negatives back to non-negative. That lets a tall value and a thin one move on comparable relative terms
— the right invariance for a scale-free objective — and I shrink `s` alongside `T` (`s ≈ 0.3·(T/T_0) + 0.01`)
so late steps make fine adjustments to a settled shape. The multiplicative form has a second effect I welcome:
a proposal that drives a height through zero is clipped to exactly `0`, and `0·(1 + s·ξ) = 0` forever after, so
zero is an *absorbing* state — a one-way ratchet toward sparsity. It can prune a piece out of the profile but
never resurrect it, which is exactly the mechanism that would carve an end-weighted, partly empty shape out of a
full one. I read the number of pruned heights in the returned profile as a direct measure of how sparse the
coarse optimum wants to be.

`R` sits in a bounded range above `1.28` and its single-kick changes are small and well-scaled, so I anneal
directly on `R` — no log, no rescale — with `T` of order a few hundredths cooled geometrically. Starting
`T_0 = 0.05` and cooling `×0.9995` per proposal, after `40000` proposals `T ≈ 0.05·e^{−20} ≈ 10^{−10}`,
genuinely frozen. Early a proposal that worsens `R` by a typical `0.01` is accepted with probability
`e^{−0.01/0.05} ≈ 0.82`, so the walk explores; once `T` has fallen to a few thousandths the same proposal is
accepted `~3%` of the time, so the walk has become a descent. Forty thousand proposals is the count that spans
free exploration to hard freeze while still visiting each of the `50` coordinates roughly `800` times. I hedge
on seeds, since I do not know which way the optimal shape leans: independent restarts from the flat vector (so
the anneal can discover the direction from scratch), a monotone ramp, a smooth single hump, and a Gaussian bump,
keeping the best profile any restart reaches — covering both the symmetric and the lopsided hypotheses.

Annealing one height at a time tends to leave the profile rough, jittering around the basin floor, so after the
anneal settles a shape I want a gradient polish. The obstruction is that `R` involves a `max` over nodes,
non-differentiable at the peak, and its naive gradient is exactly zero at the symmetric point. I replace the
hard `max` by the smooth log-sum-exp `L_β = (1/β) log Σ_k exp(β b_k)`, whose gradient with respect to `b` is
the softmax weights `w_k ∝ exp(β b_k)`. Chaining through the self-convolution, each node's gradient with respect
to a height is `∂b_k/∂a_j = 2 a_{k−j}` (differentiating `b_k = Σ_n a_n a_{k−n}` picks up two equal terms), so
`∂L_β/∂a_j = Σ_k w_k · 2 a_{k−j} = 2 (w ⋆ a)_j`, one more FFT. The mass normalization contributes the rest by
the quotient rule: with `s = Σa`, `∂R/∂a = (2N/s^2) ∂L_β/∂a − (2R/s) 1`. Everything is `O(N log N)`. I anneal
`β` *upward* over the polish — soft first, where the surrogate is a broad smooth guide, sharp last (`~20 →
~5000`), where `L_β` faithfully tracks the true peak — run Adam, since its per-coordinate adaptive scaling suits
the wide dynamic range, project to non-negative after each step, inject a small multiplicative kick periodically
(`~0.02` every `1500`) to escape shallow traps, and track the best *true* `R` ever seen so the number I keep is
real. Starting `β` soft matters: at large `β` from a rough start the softmax collapses onto whatever single
node is tallest and I inherit the one-node-at-a-time myopia I am trying to avoid; softer lets the gradient feel
the whole cluster of tall nodes and press them down together.

The ordering — anneal first, then polish — is forced by the zero gradient. Run the softmax-Adam polish from the
flat vector and it sits still: the softmax gradient there is the same identically-zero object, because the
symmetry that cancels the peak against the mass does not care whether the max is hard or softened. The polish
needs a symmetry-broken start, and the anneal supplies one. The two stages are complementary: annealing to leave
the summit and cross the shallow basins, gradient descent to reach the floor of the good basin it found.

I should be explicit about the coupling that limits how far either stage can go, because it seeds the next
rung. By the sum rule `Σ_k b_k = (Σa)^2` is conserved, so lowering the current peak node raises the total on the
others; press only the single tallest down and the second-tallest becomes the new peak — pressing a balloon. The
right object is the *whole* top cluster of near-equal nodes, lowered together under fixed mass. The softmax with
moderate `β` blurs across that cluster and so beats a hard subgradient, but it is still an averaging device, not
a genuine solver of the coupled minimax; at a coarse `50`-node autoconvolution the cluster is small enough that
this weakness stays hidden, but it will bind harder once the grid is fine and the top cluster grows.

One thing to watch is whether the optimizer drives heights to *zero* and concentrates mass toward the ends —
signal, not noise, the specific end-weighted structure I expect from a minimization that punishes central
self-overlap. The peak/mean reading sharpens it: at `N = 50` the prefactor is `~1`, so any real drop in `R` is a
proportional drop in `peak/mean`; a result near `1.55` would mean the autoconvolution top has been flattened
from twice its mean down toward `1.5×`, the top spread across several nodes rather than piled into one. All of
this is affordable at this scale — an FFT at `N = 50` is microseconds, so `40000` proposals across `4` seeds
plus a `6000`-step polish is seconds — which is why I can be profligate with candidates here, a luxury I lose
the moment I lift to hundreds or thousands of pieces.

I expect annealing alone to do most of the crossing, a large first drop off the ceiling into roughly the `1.7`
neighbourhood, leaving the profile rough; then the softmax-Adam polish should carve it the rest of the way into
the low `1.5`s. That decomposition is itself a check: if the polish added almost nothing, my claim that the
random walk stops short of the basin floor would be false and I would conclude the anneal was already
converging. I do *not* expect to reach the `600`-piece AlphaEvolve `1.5053` or the record `1.5028628969`; those
come from far finer grids, and `50` pieces with only `99` nodes caps how finely the autoconvolution top can be
flattened. So this rung should prove the principle — breaking the flat symmetry buys a large drop — and then
stall, limited not by the search idea but by coarse resolution and by a single-peak optimizer that cannot
flatten a whole plateau at once. That double stall is the opening for the next rung: lift the optimized shape
onto a finer grid and bring an optimizer that attacks the entire flat top of near-peak nodes simultaneously.
