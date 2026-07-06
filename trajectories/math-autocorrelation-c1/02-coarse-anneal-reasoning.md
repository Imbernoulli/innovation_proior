The flat profile came back at exactly `2.000000`, and identically so at `N = 50`, `600`, and `30000` — the
harness confirmed the refinement invariance I expected, and it reproduced the published record `1.5028628969`
on the record sequence, so the evaluator agrees with the literature at both ends of the range. That settles that
the machine is trustworthy and that the flat value is a genuine ceiling, not a numerical accident. The
obstruction it leaves is the whole content of this rung: with every piece identical the autoconvolution is
locked to a triangle whose single central node is the peak, and there is no direction to descend in. I want to
be precise about *why* there is no direction, because the precise reason dictates which search I reach for. At
the flat vector the argmax of `a*a` is the central node, and its gradient with respect to each height is the
same constant across all coordinates; when I form the derivative of `R = 2N·max_k(a*a)_k/(Σa)^2`, the term from
the peak and the term from the mass normalization cancel exactly, leaving the gradient identically zero. So flat
is a stationary point of `R`. A pure gradient method dropped here does not move at all — not slowly, but never,
because it has a zero to follow. That rules out the most obvious first idea, "just run gradient descent from the
flat vector," before I waste any effort on it.

But "zero gradient" does not mean "trapped," and I have to get this right or I will design the wrong search. A
single-coordinate perturbation `a_j → 1 + ε` moves `R` only at second order, and — checking a small case by
hand — that second-order move is *downward*: at `N = 4`, bumping any one height gives `R − 2 = −2ε^2/16 +
O(ε^3)`, negative for either an interior or an end coordinate. So the flat profile is not a local minimum I have
to tunnel out of; it is a local *maximum*, a hilltop, with downhill in essentially every direction, just with a
flat (zero-gradient) summit. That means a search taking finite steps can leave it easily — the difficulty is not
escaping flat. The difficulty is what happens *after* I step off. The objective is badly non-convex: `a*a` is
bilinear in the heights and the `max` over nodes is non-smooth, so the landscape below the hilltop is riddled
with genuine local minima. A greedy descender that only accepts improving moves will roll off the summit and
then park itself at the first such minimum, which for a problem this irregular is very unlikely to be a good one.
So the real requirement on the search is not "help me leave flat" but "help me not get stuck at the first poor
basin below flat." That reframing is what selects the method.

That is exactly what simulated annealing provides, and it is why I prefer it here over the alternatives. Greedy
hill-descent I have already ruled out — it stops at the first local minimum. A population method like CMA-ES
would work in principle on `50` dimensions, but it adapts a full covariance matrix and spends its budget on
relatively few, relatively expensive updates; when the objective is as cheap as an FFT and the natural move is a
simple coordinatewise rescaling, I would rather spend the same budget on tens of thousands of cheap proposals
that canvas the space than on a few hundred covariance updates. Annealing is the minimal thing that does the one
job I actually need: propose a perturbation to one height, accept it outright if it lowers `R`, and accept it
*anyway* with Metropolis probability `exp(−ΔR/T)` if it raises `R`, cooling the temperature `T` geometrically
over the run. Hot early, so the search wanders freely across the many basins below the summit and does not
commit to the first one; cold late, so it settles into whichever basin it has found to be deepest. The whole bet
is that with enough wandering it discovers a profile whose autoconvolution has its peak pushed down relative to
the mass, well below `2`.

Scale comes first among the remaining decisions, because it constrains everything after it. I am tempted to go
straight to a long height vector, but that is a trap at this stage: a high-dimensional search space lets a blind
local search wander for a very long time before it finds anything, and the good shapes are narrow. The smart
move is to find the *shape* at low resolution first, where the vector is short enough to explore thoroughly, and
defer the fine grid. The rung-1 feedback also gives me a concrete reason to land near `N = 50`: the score is
`R = [2N/(2N−1)]·(peak/mean)`, and the prefactor `2N/(2N−1)` is `1.0101` at `N = 50` — within a percent of `1`
— so the reported number tracks the underlying `peak/mean` faithfully rather than being inflated the way it
would be at very small `N`. And `50` heights is short: its autoconvolution has only `99` nodes, so a search that
evaluates tens of thousands of candidates revisits every coordinate hundreds of times and genuinely canvasses
the space. So `N` around `50` is short enough to explore, long enough that the autoconvolution already has real
internal structure to exploit and that I can read off which direction the gains come from.

Now the perturbation, and this choice comes straight from the geometry of the objective. `R` is scale-free:
multiplying every height by a common constant leaves it unchanged, so the meaningful moves are *relative*, not
absolute. The good profiles will also span a wide dynamic range — I suspect something heavier toward the ends
and lighter in the middle, since that is the arrangement that starves the central self-overlap the flat function
maximizes — so a single additive Gaussian kick of fixed size would be far too coarse for the small heights and
far too timid for the large ones. I make the kick *multiplicative in scale*: perturb a randomly chosen height by
an amount proportional to its own magnitude, `a_j ← max(0, a_j·(1 + s·ξ))` with `ξ` standard normal, clipping
any negative result back to non-negative so the candidate stays legal. That lets a tall value and a thin one
move on comparable relative terms — the right invariance for a scale-free objective — and I shrink the kick
scale `s` alongside `T` (`s ≈ 0.3·(T/T_0) + 0.01`) so that late in the run the search makes fine adjustments to
a settled shape rather than large jumps. The multiplicative form has a second effect I welcome: because a
proposal that drives a height through zero is clipped to exactly `0`, and `0·(1 + s·ξ) = 0` for any subsequent
kick, zero is an *absorbing* state under the anneal. A height that is once starved to zero stays there. That
makes the anneal a one-way ratchet toward sparsity — it can prune a piece out of the profile but a
multiplicative kick can never resurrect it — which is exactly the mechanism that would carve an end-weighted,
partly empty shape out of a full one, rather than merely reweighting a fully-supported vector. I read the number
of pruned heights in the returned profile as a direct measurement of how sparse the coarse optimum wants to be.

What to anneal on needs a moment's thought but resolves cleanly. `R` sits in a bounded range above the analytic
floor `1.28`, and its changes under a single multiplicative kick are small and well-scaled, so I do not need to
take a log or rescale the objective; I can anneal directly on `R` with a temperature of order a few hundredths
cooled geometrically, and the Metropolis acceptance behaves sanely — a kick that worsens `R` by, say, `0.01`
against `T = 0.05` is accepted about `82%` of the time early and almost never once `T` has cooled to a few
thousandths, which is exactly the hot-then-cold behaviour I want. The cooling schedule follows from those numbers. I want the temperature to start hot enough that early uphill
moves are routinely accepted and end cold enough that the last thousands of proposals are effectively greedy, so
the search freezes into its best basin. Starting at `T_0 = 0.05` and cooling geometrically by `×0.9995` per
proposal, after `40000` proposals the temperature is `0.05·0.9995^{40000} ≈ 0.05·e^{−20} ≈ 10^{−10}` —
genuinely frozen. Early on, a proposal that worsens `R` by a typical `0.01` is accepted with probability
`e^{−0.01/0.05} ≈ 0.82`, so the walk really does explore; by the time `T` has fallen to a few thousandths the
same proposal is accepted with probability `e^{−0.01/0.003} ≈ 0.036`, essentially never, so the walk has become
a descent. Forty thousand proposals per restart is the number that lets this geometric cool span the full range
from free exploration to hard freeze while still visiting each of the `50` coordinates roughly `800` times.

The seeds are where I should hedge, because I
do not yet know which way the optimal shape leans. I run several independent restarts from different
initializations — the flat vector itself, so the anneal can discover the descent direction from scratch; a
monotone ramp; a smooth single hump; and a Gaussian bump — and keep the best profile any restart ever reaches.
The ramp and the bump are educated guesses: if the gains come from breaking the left-right symmetry of the flat
indicator, an asymmetric or concentrated seed should land in the good basin faster than starting flat. I am
covering both the symmetric and the lopsided hypotheses rather than betting on one.

There is a real risk that annealing alone, perturbing one height at a time, leaves the profile rough and stops
short of the local optimum its basin contains — the multiplicative random walk gets the shape approximately
right but jitters around the true minimum. So after the anneal settles a shape I want a gradient polish to slide
it to the bottom of that basin. The obstruction is that `R` involves a `max` over the autoconvolution nodes,
which is not differentiable at the peak, and I already saw that a naive gradient is exactly zero at the most
symmetric point. I handle the non-smoothness the standard way: replace the hard `max` by the smooth
log-sum-exp `L_β = (1/β)·log Σ_k exp(β·b_k)`, whose gradient with respect to the node vector `b` is precisely
the softmax weights `w_k ∝ exp(β·b_k)`. Then I chain through the self-convolution. Each node's gradient with
respect to a height is `∂b_k/∂a_j = 2·a_{k−j}` — differentiating `b_k = Σ_n a_n a_{k−n}` picks up two equal
terms — so by the chain rule `∂L_β/∂a_j = Σ_k w_k·2·a_{k−j} = 2·(w ⋆ a)_j`, a correlation of the softmax
weights against the heights, which is one more FFT. The mass normalization contributes the second piece by the
quotient rule: with `s = Σa` and writing the surrogate ratio in place of `R`, `∂R/∂a = (2N/s^2)·∂L_β/∂a −
(2R/s)·1`. Everything is `O(N log N)`, so the polish is cheap. I anneal `β` *upward* over the polish — starting
soft, where the log-sum-exp is a smooth broad surrogate whose gradient points toward better shapes generally,
and ending sharp, where `L_β` faithfully tracks the true peak — run Adam on this gradient, since its
per-coordinate adaptive scaling suits the wide dynamic range of the heights, project to non-negative after each
step, inject a small multiplicative kick periodically to escape shallow traps, and — this matters — track the
best *true* `R` ever seen, never the surrogate's value, so the number I keep is real.

The ordering — anneal first, then polish — is not incidental, and the zero-gradient fact is exactly why. If I
tried to run the softmax-Adam polish from the flat vector directly, it would sit still: the softmax gradient at
the flat profile is the same identically-zero object as the hard subgradient, because the symmetry that makes
the peak term cancel the mass term does not care whether the max is hard or softened. So the gradient polish
*needs* a symmetry-broken starting point to have any slope to descend, and the anneal is precisely what supplies
one — it breaks the symmetry with its uphill-tolerant random walk and hands the polish a rough asymmetric shape
that is already off the hilltop. The two stages are complementary by construction: annealing to leave the
zero-gradient summit and escape the shallow basins, gradient descent to reach the floor of the good basin the
annealing found. I also anneal `β` from soft (`~20`) to sharp (`~5000`) rather than jumping straight to a large
`β`, because at large `β` from a rough start the softmax collapses onto whatever single node happens to be
tallest and I inherit the same one-node-at-a-time myopia I am trying to avoid; starting soft lets the gradient
feel the whole cluster of tall nodes and press them down together before it sharpens. The periodic small
multiplicative kick (`~0.02` every `1500` steps) is a cheap hedge against Adam settling into a shallow trap of
its own — it jostles the shape, and because I only ever keep the best true `R`, a kick that fails to help costs
nothing.

I should be explicit about the coupling that limits how far either stage can go, because it is the seed of the
next rung's whole design. By the sum rule the total overlap `Σ_k b_k = (Σa)^2` is conserved, so lowering the
current peak node necessarily raises the total on the other nodes; if I press only the single tallest node down,
the second-tallest becomes the new peak and `R` barely moves — it is like pressing one spot on a balloon. The
right object is therefore the *whole* top cluster of near-equal nodes, lowered together subject to the mass
being fixed. The softmax with a moderate `β` blurs across that cluster and so does better than a hard
subgradient, but it is still an averaging device, not a genuine solver of the coupled problem of holding the
whole top cluster down at once, and at a coarse `50`-node autoconvolution the cluster is small enough that this
weakness stays hidden. I expect it to bind harder once the
grid is fine and the top cluster grows to many near-tight nodes — but that is a problem for the resolution I do
not yet have.

One thing I want to watch as this runs is whether the optimizer drives some heights to *zero* and concentrates
mass toward the ends. That would be signal, not noise: it would mean the best coarse profile is genuinely
asymmetric and partly sparse — a specific structure, not an even spread — which is exactly what I expect from a
minimization that punishes central self-overlap and rewards pushing mass to the boundary. The peak/mean reading
sharpens the prediction: the flat profile has `peak/mean ≈ 2`, and any real drop in `R` at `N = 50` (where the
prefactor is ~`1`) is a proportional drop in `peak/mean`, so a result near `1.55` would mean the autoconvolution
top has been flattened from twice its mean down toward `1.5×` its mean — the top spread across several nodes
rather than piled into one.

The budget arithmetic confirms this is all affordable at this scale, which is the other reason to stay small.
An FFT autoconvolution at `N = 50` is microseconds, so `40000` proposals across `4` seeds is `160000`
evaluations — a fraction of a second of real convolution work — and the `6000`-step Adam polish adds two FFTs
per step, another few thousand transforms. The whole pipeline is seconds, not minutes. That cheapness is what
lets me run four restarts and a long polish here without thinking about cost, and it is a luxury I will lose the
moment I lift to hundreds or thousands of pieces, where the per-evaluation cost climbs and the number of
proposals I can afford collapses. So this rung is also the last place I get to be profligate with the number of
candidates; the coarse grid buys both a small search space and a cheap objective at once.

One design alternative I considered and folded in rather than discarded is plain basin-hopping — many random
restarts of pure local descent, keep the best. That is almost what I am doing with the several seeds, but
without the annealed uphill acceptance each restart would be a greedy descent that stops at the first minimum
below its seed, so I would be sampling first-minima rather than searching within each basin. Annealing inside
each restart is the strict improvement: it lets a single run cross ridges between shallow basins before
committing, so four annealed restarts explore far more than four greedy ones. I keep the multi-seed structure
for its coverage of the symmetric-versus-lopsided question and get the within-basin escape for free from the
Metropolis rule.

The prediction is testable in a way I can commit to before I run it, and the two stages should be separable in
the metric. I expect annealing alone to do most of the crossing — from the ceiling at `2.0` down into the
neighbourhood of `1.7`, a large first drop — but to leave the profile rough, jittering around its basin rather
than sitting at the bottom. Then I expect the softmax-Adam polish to carve that rough shape the rest of the way,
into the low `1.5`s. If the polish instead added almost nothing on top of the anneal, that would falsify my
claim that the multiplicative random walk stops short of the basin floor, and I would have to conclude the
anneal was already converging — so the decomposition is itself a check on my picture of the two stages, not just
a pipeline. On the returned metric columns, that reads as `pieces_N = 50` and a `ratio` somewhere in the low
`1.5`s, with the sparsity of the returned vector — how many of the `50` heights are driven to (near-)zero —
telling me directly whether the optimum is the asymmetric, partly sparse, end-weighted shape I am predicting or
something more evenly spread that would surprise me.

So what do I expect the harness to report? The annealing-plus-polish should clear the flat-triangle ceiling
easily — that is the entire reason to accept uphill moves and slide down the basin — and drop well into the low
`1.5`s, the band where coarse constructions for this constant live, on a genuinely asymmetric, partly sparse
profile with mass toward the ends. I do *not* expect to reach the `1.5053` of the `600`-piece AlphaEvolve
construction, let alone the `1.5028628969` record; those are the product of careful optimization on far finer
grids, and `50` pieces is a coarse grid that caps how low the peak can be pushed — with only `99` nodes there is
a hard limit on how finely the autoconvolution top can be flattened. So I expect this rung to prove the
principle — that breaking the flat symmetry buys a large drop, from `2.0` down past `1.55` — and then to stall,
limited not by the search idea but by the coarse resolution. There is also a second limit I can already feel
even if I cannot fully act on it here: when I push the peak down I am not moving one node, I am trying to hold a
whole cluster of near-equal top nodes down together, and a gradient (even the softmax one) presses on the single
current argmax while a neighbour in the cluster pops up to replace it. Annealing and the softmax polish blur
past this at `50` pieces, but it is a structural weakness that will bind harder as the grid refines. That double
stall — coarse resolution, and a single-peak optimizer that cannot flatten a whole plateau at once — is the
opening for the next rung: lift this optimized coarse shape onto a much finer grid and bring an optimizer that
attacks the entire flat top of near-peak nodes simultaneously, rather than one node at a time.
