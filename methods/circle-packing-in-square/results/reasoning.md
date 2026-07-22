OK, let me think this through from scratch. I want to pack `n` equal circles into the unit
square as fat as possible. Circles are awkward objects to optimize over — overlap is a quadratic
constraint, the radius is shared, and the wall offset depends on the radius too. So before I
touch any optimizer, let me see whether I can get rid of the circles entirely.

A circle of radius `r` at centre `s` needs `s` to be at least `r` from every wall and every two
centres to be at least `2r` apart. Fix the centres and ask: what is the largest `r` they admit?
It's set by whichever is tighter — the closest pair of centres (gives `r ≤ ½·min_{i<j} ‖s_i−s_j‖`)
or the closest centre-to-wall distance. The wall part is a nuisance, but I can absorb it: a circle
in the unit square with centre `r` from the walls is the same as a *point* in a shrunken square of
side `1−2r`. So if I rescale the shrunken square back to unit size, the circle problem becomes:
put `n` points in a unit square, push the minimum pairwise distance up. Let me nail the conversion.
Centres live in a square of side `1−2r`; rescale by `1/(1−2r)` so they live in the unit square,
and the smallest center-spacing `2r` becomes `2r/(1−2r) =: d`. Invert: `d(1−2r)=2r`, so
`d = 2r/(1−2r)` and `r = d/(2(1+d))`. Since `r` is monotone increasing in `d`, the point-spreading
problem — maximize `d_n = min_{i<j} ‖s_i−s_j‖` over `n` points in the unit square — should be the
same problem as maximizing the radius. Before I commit to that I want to check the conversion on a
case I already know. Take `n = 2`: two points spread maximally in the unit square sit at opposite
corners, so `d_2 = √2`. Then `r = √2/(2(1+√2)) = 0.29289…`, which I can simplify by rationalizing
to `1/(2+√2)` — and `1/(2+√2) = 0.29289…` too. That is exactly the textbook `r_2` for two circles
in a square, so the formula maps the point answer back onto the circle answer correctly. One more:
`n = 5`, whose exact value is `d_5 = √2/2` (four corners plus centre). Then
`r = (√2/2)/(2(1+√2/2)) = 0.20711…`, and the published `r_5` is `1/(2+2√2) = 0.20711…` — same
number. The conversion holds on both, so from here on I forget the circles. I have `n` points in a
box and I want them as spread out as possible.

Now, the objective. `d_n = min_{i<j} ‖s_i − s_j‖`. Stare at this for a second. It is a *minimum
over pairs*. At any given configuration only one pair, or a few tied pairs, actually achieve that
minimum; all the other distances are bystanders. If I perturb a point, the active pair can switch,
and the function has a kink there — it is piecewise-smooth with a subgradient at the active
constraint, not a gradient. That kink is worth taking seriously, because it predicts trouble for
any optimizer that assumes a clean gradient. de Groot, Peikert and Würtz did exactly the direct
thing — variables are the centres, maximize the admissible radius, drive it with simplex and BFGS
— and they couldn't reach the optimum for `n = 14, 15, 17`. A smooth quasi-Newton method builds a
quadratic model from gradients, but here the function has no clean gradient at the very points that
matter (the contacts); it stalls in the kinks. Tellingly, their stochastic Langevin variant —
gradient flow plus injected noise — did *better*, which fits the diagnosis: the noise lets it slip
out of the kinks a plain gradient method gets stuck in. So whatever I do, I want something that
smooths over the kinks and shakes free of local optima.

The other tempting escape is brute force: put a fine grid in the square, try point subsets, maybe
enumerate candidate contact graphs. That is exactly the wrong direction. The search space is
continuous in `2n` variables, so a grid gets exponentially expensive before it is even accurate;
the best packings often move by tiny symmetry-breaking offsets, so the optimum is not sitting on
my grid; and the contact graph is the thing I am trying to discover, not something I can cheaply
guess in advance. Brute force spends the budget on a discretization artifact and still leaves me
with the original geometric question. I need a smooth continuous surrogate, not an enumeration.

Let me think about what the "right" smoothing is. The pain is the hard `min`. Is there a soft
version of `min` that becomes the hard one in a limit, and is differentiable along the way? The
classic `L^p`/power-mean trick is the candidate. For positive numbers `a_k`,
`(Σ_k a_k^p)^{1/p} → max_k a_k` as `p → +∞`, and `→ min_k a_k` as `p → −∞`. The negative side is
the one I need. Let me pin down *why* it converges to the min, because I'll lean on it. If
`a_* = min_k a_k` and there are `N` terms, then for `p < 0`, dividing through by `a_*^p` (the
largest of the `a_k^p`, since `p<0` inverts the ordering), `a_*^p ≤ Σ_k a_k^p ≤ N a_*^p`; raising
to `1/p` reverses the inequalities (that map is decreasing for `p<0`), so
`a_* ≥ (Σ_k a_k^p)^{1/p} ≥ a_* N^{1/p}`, and `N^{1/p} → 1`. So the soft-min is squeezed onto `a_*`.
I don't fully trust an inequality chain I just wrote until I've seen it move, so let me put numbers
on it: take `a = (1.0, 1.5, 2.3)`, where `min = 1.0` and `N = 3`. At `p = −2` the soft-min is
`0.782`, and the lower bound `a_* N^{1/p} = 1·3^{−1/2} = 0.577` — so `0.577 ≤ 0.782 ≤ 1.0`, the
squeeze holds and it's still loose. At `p = −10` the soft-min is `0.998` with lower bound `0.896`;
at `p = −50` it reads `1.0000000` to seven places, lower bound `0.978`. The bound brackets the
value at every step and the value climbs monotonically to `1.0` exactly as the chain says. Good —
the surrogate really does become the hard min. Concretely,
`min_{i<j} ‖s_i−s_j‖ = lim_{p→−∞} ( Σ_{i<j} ‖s_i−s_j‖^p )^{1/p}.`
The sign matters. For fixed negative `p`, the map `z ↦ z^{1/p}` is decreasing, so maximizing the
soft minimum is the same as minimizing `Σ ‖s_i−s_j‖^p`. If I keep Euclidean distances, this is an
inverse-power repulsion. For the optimizer, though, I want the algebra in squared distances. Write
`q_ij = (x_i−x_j)² + (y_i−y_j)² = ‖s_i−s_j‖²`. Choosing `p = −2m` with `m > 0` gives
`‖s_i−s_j‖^p = q_ij^{-m}`. So the positive sharpness parameter I optimize with is not `−p`; it is
`−p/2`, because the square has been folded into the exponent. Multiplying every term by the fixed
positive constant `λ^m` cannot change the minimizer for a stage, so the energy I should minimize is
`E = Σ_{i<j} (λ / q_ij)^m`.
Each term is huge when a pair is close and tiny when it's far. As `m` grows, the whole sum becomes
dominated by the single closest pair, so pushing `E` down increasingly pushes the *closest* pair
apart; in the limit, that is exactly the max-min objective for `d_n`. And as a bonus the physical
reading is immediate: this is a steep repulsion potential, `1/ρ^{2m}` in the Euclidean distance
`ρ`. The points are mutually
repelling particles and I'm letting them relax to a low-energy configuration. That's the same
instinct Clare and Kepert, and Kottwitz, used for spreading points on a sphere — minimize a
repulsive energy rather than the raw min-distance. They used it as a fixed surrogate, though, and a
fixed soft energy isn't the same as the max-min: for moderate `m` it cares about *all* the
distances, not just the smallest, so its minimizer trades a slightly-too-close pair for many
comfortably-far ones. The max-min only cares about the worst pair. So `m` can't stay moderate; I'll
come back to that.

Before optimizing I still have the box constraint `0 ≤ x_i, y_i ≤ 1`. I could add penalty walls or
project, but penalties reintroduce kinks at the boundary and projection makes points *stick* to
the wall in a way that fights the optimizer. There's a slicker move: change variables so the box
constraint is automatic. If I write `x_i = sin(x̃_i)` with `x̃_i` ranging freely over `R`, then
`x_i` is *always* in `[−1, 1]` no matter what `x̃_i` does — and `[−1,1]` is just a box of side 2,
which is the unit square up to a harmless rescale (the conversion to `r` doesn't care about the
absolute side). Same for `y_i = sin(ỹ_i)`. So in the `(x̃, ỹ)` variables the problem is **fully
unconstrained** — `2n` real numbers, no walls to enforce, the sine bends any real value back into
the box. The constraint hasn't been penalized, it's been *parameterized away*. The price is a
chain-rule factor `∂x_i/∂x̃_i = cos(x̃_i)`, which is mild and only vanishes at the box edges where
a point is hard against a wall anyway.

Now the `λ`. With large `m`, the raw power `(λ/d²)^m` is catastrophic numerically: if `λ/d²` is far
above 1, raising it to a large exponent overflows; if far below 1, it underflows to 0 and the
gradient dies. The base needs to sit right around 1 for the dominant terms. The dominant term is
the closest pair, whose `d²` is the current minimum squared distance. So after each optimization
stage I set **`λ = d²_min`**, the square of the current shortest distance. Then the worst term
starts at `1`, all other terms start below `1`, and the comfortable pairs contribute negligibly.
During the next smooth minimization I hold this `λ` fixed; it is a numerical conditioner, not
another coordinate of the objective I'm differentiating.

Even with that centering, a trial step inside a line search can temporarily make a pair closer than
the stage-start minimum, and then a raw power sum can still overflow. I do not actually need the
raw value of `E`; I only need an objective with the same minimizers. Since `log` is increasing,
minimizing `E` and minimizing `log E = log Σ exp(m(log λ − log d_ij²))` are equivalent for fixed
`m` and fixed `λ`. Computing that with `logsumexp` removes the overflow path entirely. It also
gives the right scaled gradient: write `T_ij = (λ/d_ij²)^m`, `Z = Σ_{a<b} T_ab`, and
`w_ij = T_ij/Z` for each unordered pair. Then
`∂log Z = (1/Z)∂Z`, so
`∂logE/∂x_i = Σ_{j≠i} (−m·w_ij / d_ij²)·2(x_i − x_j)`,
and the same with `y`. Finally chain through the sine:
`∂logE/∂x̃_i = (∂logE/∂x_i)·cos(x̃_i)`. The repulsion reading is still exactly the same, except
the pair forces are normalized by the total energy: close pairs get almost all the weight, distant
pairs get almost none, and the exponent controls how hard that soft selection becomes.

So I can run conjugate gradients or a Newton method to a local minimum of this energy for a fixed
`m`, using `log E` in floating point so the arithmetic stays finite.
If `m` is small, the energy is gentle and almost convex-ish — easy to minimize, few bad local
minima — but it's a poor proxy: it rewards spreading
*all* the points evenly, not maximizing the single worst gap, so its minimizer is a smeared-out
configuration that isn't the max-min optimum. If `m` is huge, the energy is a faithful proxy for
`min`-distance — only the closest pair matters — but the landscape is now savagely non-convex,
nearly as kinky as the original `min`, with the same swarm of local optima, and an optimizer
dropped into it from a random start will jam in the first basin it finds.

So I want the faithfulness of large `m` and the tractability of small `m`. I can't have both at one
value — but I don't have to use one value. **Anneal `m`.** Start at a small `m` (say in the range
10–100), where the energy is smooth and forgiving, and minimize it; this drags a random cloud of
points into a sensibly spread configuration that's already in roughly the right neighbourhood.
Then *double* `m`, recompute `λ`, and re-minimize, warm-started from where I just was. Each
doubling sharpens the energy a little — the focus narrows from "all distances" toward "only the
smallest distances" — and because I start each stage from the previous stage's solution, I'm
tracking the minimizer continuously as the objective morphs from the easy soft version toward the
true max-min. This is a homotopy / continuation: deform a problem I can solve into the problem I
want, following the solution along the way. In a high-precision production search I can keep
doubling toward `m ≈ 10⁶` once the contact pattern is nearly stable, and in stubborn cases the same
continuation can be pushed as far as about `10⁵⁰`. In a compact double-precision implementation I
stop much earlier, at `1280` by default, because the log objective is already sharp enough for the
small sanity checks and pushing farther mostly makes the local optimizer stiffer. At that point the
smooth surrogate is close enough to the hard `min` for this compact double-precision
implementation, and I have a candidate local maximum of `d_n`.

"Local" is the operative word, and it's the second wall. Even with annealing, a single run lands in
*a* good basin, not necessarily the best one — the landscape of rigid packings is genuinely
multimodal (and I know from the structure of these problems that the best arrangement is often a
*slightly broken* symmetric one, so clean symmetric basins are exactly the seductive local optima
to be wary of). The cure is brute multiplicity at the *start*, not the end: **many random restarts**
— at least ~50 independent runs per `n`, each from a fresh random cloud — and keep the best. The
annealing makes each run reliable; the restarts cover the basins.

There's also a specific, recurring failure mode worth a targeted patch. Sometimes a run converges
with a circle pressed against a wall that *could* slide inward without touching anyone — it's a
local optimum of `E` only because the boundary parameterization happens to pin it there, not
because the packing is rigid. When I see a point hard against a wall whose interior neighbourhood is
empty, I try a tiny inward move and accept it only if the current minimum distance does not get
smaller. Then I re-minimize at the same large `m`. That keeps the repair honest: it cannot improve
the score by allowing an overlap, and it only breaks a boundary pin that was not part of the real
contact structure.

Now, the configuration coming out of the optimizer is good to maybe 6–8 digits, but the answer I
want — `d_n` — needs far higher precision if two candidates are close or if the structure is going
to be used in a proof. The optimizer won't give me that; floating-point energy minimization
saturates. But the *structure* it reveals is much more stable than the last few coordinate digits.
So the last move is to read off the structure and re-solve it. Sort all pairwise distances into
increasing order and look for the spot where they jump — below the jump are the "contact" distances
(pairs that are touching, all essentially equal to `d_n`), above it are the genuine gaps. Likewise,
points sitting within a tiny threshold of a wall are taken to be *on* the wall. Each contact
(`d_ij² = d²` for a touching pair, or `x_i = 0`/`= 1` for a wall point) is one equation; together
they form a nonlinear system in the coordinates and `d`. Solve that system by Newton–Raphson in
arbitrary precision, and the contacts become numerically consistent to far more digits (the
overlaps/gaps can be driven below `~10⁻³³`, leaving essentially no doubt the structure is real). In
special cases the system can be reduced, by elimination, to a single univariate polynomial whose
appropriate root is the diameter; for large irregular packings the practical output is the
high-precision numerical solution. The detection is automatic given two thresholds (how close a
distance must be to the shortest to count as a contact, how close a point must be to a wall to
count as on it); only in a few awkward cases — where gaps are so narrow the system is over- or
under-determined — do I have to add or drop an equation by hand until I get a solvable system
describing a genuinely non-overlapping packing. In a compact SciPy version I cannot pretend that
`least_squares` is a proof engine, so I add a guard: after polishing, recompute the actual minimum
pairwise distance and accept the polished coordinates only if the contact residual is small, the
solved contact distance is finite and agrees with the actual minimum, and the actual minimum has not
dropped. If the detected contact system was wrong, the code falls back to the unpolished optimizer
output.

Before I trust any of this on hard `n`, I should run the whole pipeline on cases whose answer I
already know and see what comes back. `n = 5` first: with 20 restarts the solver returns
`d_5 = 0.70710678`, and `√2/2 = 0.70710678` — they agree to about `6×10⁻¹¹`, so the annealing plus
polish reproduces the exact pigeonhole value essentially to machine precision. `n = 9` is the 3×3
grid: nine points at `{0, ½, 1}²` have nearest spacing exactly `½`, so `d_9 = 0.5`, and the solver
returns `0.50000000` on the nose. (I'd first mis-remembered `d_9` as `√2/4 ≈ 0.354`; computing the
grid spacing directly — the nearest neighbours are half a side apart, not a half-diagonal — set me
straight, and the code agrees with the corrected `0.5`.) `n = 6` is the interesting one: the solver
gives `d_6 = 0.60092521`, converting to `r_6 = 0.18768`, against the published `r_6 = 0.18776` — so
the raw optimizer lands about four digits in, not ten. That four-vs-ten-digit gap is exactly the
reason the contact polish exists: for the lattice-clean cases the optimizer already nails it, but
for an irregular packing like `n = 6` the floating-point energy saturates a few digits short, and
only re-solving the contact equations recovers the rest. The checks do two jobs at once — they show
the surrogate-plus-polish actually recovers known optima, and they show me where its unaided
accuracy runs out.

Let me also sanity-check this against the physical alternative I know works, to make sure I trust
the energy route. The billiard picture — Graham and Lubachevsky's growing-disk molecular dynamics,
the Lubachevsky–Stillinger simulation — starts the points as particles with random velocities and a
common radius growing linearly in time, runs event-driven elastic collisions, and stops when the
assembly jams. That's a beautiful, genuinely different way to reach a rigid packing, and it's
principled physics. But notice what it shares with my method and what it doesn't. It shares the
multimodality cure-by-randomness: different initial velocities jam at different radii, so it too
needs many restarts. What it lacks is the continuation knob — there's no smooth dial turning an
easy problem into the hard one; you just run dynamics and take what jams, and "rattlers" (loose
particles) can leave the packing under-determined. The energy method gives me a single
differentiable objective I can ride from soft to sharp, an exact gradient, and a clean handoff to a
contact-equation polish. That's why I'll build on the energy route. The billiard simulation is the
cross-check: when both land on the same rigid configuration, I believe it.

The variables are `u = (x̃, ỹ) ∈ R^{2n}`; `to_box` is the sine map; the objective and its gradient
are the log inverse-power energy and the normalized force I derived with `λ` fixed during a stage;
`solve` does the restart loop, and inside each restart the `m`-annealing loop resets
`λ = d²_min`; `maybe_release_wall_points` performs the controlled inward nudge; and
`polish_contacts` reads off the contact structure, solves the resulting equations numerically, and
validates the result against the actual minimum distance. The first value returned by `solve` is
`d_n` in the unit square, not the circle radius; `radius_from_unit_distance` performs the final
conversion to `r_n`.

The causal chain, start to finish: packing fat circles is, after absorbing the wall offset and a
rescale (`r = d/(2(1+d))`), nothing but spreading points to maximize their minimum pairwise
distance; that minimum is a non-smooth `min`-over-pairs that defeats smooth optimizers, so I
replace it by the power-mean soft-min and minimize the inverse-power repulsion energy
`Σ(λ/q_ij)^m` with `q_ij = ‖s_i−s_j‖²`, which a large positive `m` makes faithful to the true
minimum; I parameterize the
box away with `x=sin(x̃)` so the search is unconstrained, condition the arithmetic with
`λ=d²_min` while holding `λ` fixed inside each gradient calculation, and use the exact gradient to
run a real optimizer; I get faithfulness *and* tractability by annealing `m` from soft to sharp as a
homotopy, warm-starting each stage; I beat the multimodality with many random restarts and a
controlled inward nudge for non-rigid wall pins; and I sharpen the reported `d_n` by detecting the
contact structure, with the compact code accepting only non-worsening contact polish and the full
search using high-precision Newton refinement.
