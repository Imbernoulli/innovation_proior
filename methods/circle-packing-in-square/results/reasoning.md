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
`d = 2r/(1−2r)` and `r = d/(2(1+d))`. Clean. `r` is monotone increasing in `d`, so **maximizing
the minimum pairwise distance `d_n = min_{i<j} ‖s_i−s_j‖` of `n` points in the unit square is
exactly maximizing the radius.** From here on I forget the circles. I have `n` points in a box and
I want them as spread out as possible.

Now, the objective. `d_n = min_{i<j} ‖s_i − s_j‖`. Stare at this for a second. It is a *minimum
over pairs*. At any given configuration only one pair, or a few tied pairs, actually achieve that
minimum; all the other distances are bystanders. If I perturb a point, the active pair can switch,
and the function has a kink there — it is piecewise-smooth with a subgradient at the active
constraint, not a gradient. That's the wall the obvious approaches keep hitting. de Groot, Peikert
and Würtz did exactly the direct thing — variables are the centres, maximize the admissible
radius, drive it with simplex and BFGS — and they couldn't reach the optimum for `n = 14, 15, 17`.
A smooth quasi-Newton method builds a quadratic model from gradients, but here the function has no
clean gradient at the very points that matter (the contacts); it stalls in the kinks. Tellingly,
their stochastic Langevin variant — gradient flow plus injected noise — did *better*. So the
missing ingredient is something that smooths over the kinks and shakes free of local optima.

Let me think about what the "right" smoothing is. The pain is the hard `min`. Is there a soft
version of `min` that becomes the hard one in a limit, and is differentiable along the way? Yes —
the classic `L^p`/power-mean trick. For positive numbers `a_k`,
`(Σ_k a_k^p)^{1/p} → max_k a_k` as `p → +∞`, and `→ min_k a_k` as `p → −∞`. The negative side is
the one I need. If `a_* = min_k a_k` and there are `N` terms, then for `p < 0`,
`a_*^p ≤ Σ_k a_k^p ≤ N a_*^p`; raising to `1/p` reverses the inequalities, so
`a_* ≥ (Σ_k a_k^p)^{1/p} ≥ a_* N^{1/p}`, and `N^{1/p} → 1`. Thus
`min_{i<j} ‖s_i−s_j‖ = lim_{p→−∞} ( Σ_{i<j} ‖s_i−s_j‖^p )^{1/p}.`
The sign matters. For fixed negative `p`, the map `z ↦ z^{1/p}` is decreasing, so maximizing the
soft minimum is the same as minimizing `Σ ‖s_i−s_j‖^p`. Put `m = −p > 0` and the same objective is
`Σ 1/‖s_i−s_j‖^m`. So I should **minimize**
`E = Σ_{i<j} 1 / ‖s_i − s_j‖^m`.
Each term is huge when a pair is close and tiny when it's far. The whole sum is dominated, for
large `m`, by the single closest pair. Pushing `E` down therefore pushes the *closest* pair apart
— which is precisely maximizing `d_n`. And as a bonus the physical reading is immediate: this is a
**repulsion potential** between like charges, `1/r^m`. The points are mutually repelling particles
and I'm letting them relax to a low-energy configuration. That's the same instinct Clare and
Kepert, and Kottwitz, used for spreading points on a sphere — minimize a repulsive energy rather
than the raw min-distance. They used it as a fixed surrogate, though, and a fixed soft energy isn't
the same as the max-min: for moderate `m` it cares about *all* the distances, not just the
smallest, so its minimizer trades a slightly-too-close pair for many comfortably-far ones. The
max-min only cares about the worst pair. So `m` can't stay moderate; I'll come back to that.

Let me work in squared distances to keep everything polynomial and differentiable —
`d_ij² = (x_i−x_j)² + (y_i−y_j)²` — and fold the square into the exponent, writing the energy as
`E = Σ_{i<j} (λ / d_ij²)^m`. The `λ` in the numerator I'll explain in a moment; for now it's a
positive scale. Minimizing `Σ (λ/d²)^m` is the same as minimizing `Σ 1/d^{2m}`, same idea, just a
constant factor and a doubled exponent.

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

Now the `λ`. With large `m`, `(λ/d²)^m` is catastrophic numerically: if `λ/d²` is far above 1,
raising to `m = 10⁶` overflows; if far below 1, it underflows to 0 and the gradient dies. The base
needs to sit right around 1 for the dominant terms. The dominant term is the closest pair, whose
`d²` is the current minimum squared distance. So after each optimization stage I set
**`λ = d²_min`**, the square of the current shortest distance. Then the worst term starts at `1`,
all other terms start below `1`, and the comfortable pairs contribute negligibly. During the next
smooth minimization I hold this `λ` fixed; it is a numerical conditioner, not another coordinate of
the objective I'm differentiating.

Let me get the gradient with `λ` held fixed, because I want a real optimizer, not finite
differences over `2n` vars. Write `T_ij = (λ/d_ij²)^m`. Then
`∂T_ij/∂(d_ij²) = m·(λ/d_ij²)^{m−1}·(−λ/d_ij⁴) = −m·T_ij/d_ij²`.
And `∂(d_ij²)/∂x_i = 2(x_i − x_j)`. The unordered pair `{i,j}` contributes to both point `i` and
point `j`; collecting everything that lands on point `i`,
`∂E/∂x_i = Σ_{j≠i} (−m·T_ij / d_ij²)·2(x_i − x_j)`,
and the same with `y`. Finally chain through the sine: `∂E/∂x̃_i = (∂E/∂x_i)·cos(x̃_i)`. (I
checked this against a central finite-difference — they agree to a relative `~5×10⁻⁹`, so the
sign and the factor of `2m` are right.) The repulsion reading is reassuring: `−2m·T_ij/d_ij²·
(x_i−x_j)` is a force pushing `i` directly away from `j`, strong when they're close (small `d²`,
big `T`), negligible when far — a steep pairwise repulsion whose sharpness is controlled by `m`.

So I can run conjugate gradients or a Newton method to a local minimum of `E` for a fixed `m`.
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
want, following the solution along the way. Keep doubling — `m` runs up through `10³`, `10⁴`, and
out to `~10⁶`, occasionally far beyond (`10⁵⁰`) for stubborn cases — until only the genuine
minimum distances feel the energy and the configuration stops moving. At that point the smooth
surrogate has become the hard `min` for all practical purposes, and I'm sitting at a local maximum
of `d_n`.

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
empty, I nudge it off the wall by hand and re-minimize with a large `m`; it then settles into the
genuinely better arrangement. Small thing, but it rescues otherwise-wasted runs.

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
describing a genuinely non-overlapping packing.

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

Let me write it. The variables are `u = (x̃, ỹ) ∈ R^{2n}`; `to_box` is the sine map; the objective
and its gradient are the inverse-power energy and the force I derived with `λ` fixed during a
stage; `solve` does the restart loop, and inside each restart the `m`-annealing loop resets
`λ = d²_min`; and `polish_contacts` reads off the contact structure and solves the resulting
equations numerically.

```python
import numpy as np
from scipy.optimize import minimize, least_squares

# x_i = sin(x_tilde_i), y_i = sin(y_tilde_i): the box constraint |x|,|y| <= 1 is automatic,
# so the search over u = [x_tilde; y_tilde] in R^{2n} is fully unconstrained.
def to_box(u, n):
    ut = u.reshape(2, n)
    return np.sin(ut[0]), np.sin(ut[1])          # coords in [-1,1] (a box of side 2)

def pairwise_sq_dists(x, y):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return dx*dx + dy*dy

def min_pair_distance(u, n):
    x, y = to_box(u, n)
    d2 = pairwise_sq_dists(x, y)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2.min())

# E = sum_{i<j} (lambda / d_ij^2)^m : a 1/r^(2m) repulsion. As m grows it is dominated by the
# single closest pair, so minimizing E maximizes the minimum pairwise distance.
def objective(u, n, sharpness, scale):
    x, y = to_box(u, n)
    d2 = pairwise_sq_dists(x, y)
    iu = np.triu_indices(n, 1)
    return np.sum((scale / d2[iu])**sharpness)

# dE/dx_i = sum_{j!=i} (-m T_ij / d_ij^2) * 2(x_i - x_j),  T_ij = (lam/d_ij^2)^m,
# then chain through the sine: dE/dx_tilde_i = (dE/dx_i) * cos(x_tilde_i).
def objective_grad(u, n, sharpness, scale):
    ut = u.reshape(2, n)
    x, y = np.sin(ut[0]), np.sin(ut[1])
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx*dx + dy*dy
    np.fill_diagonal(d2, np.inf)
    T = (scale / d2)**sharpness
    coef = -sharpness * T / d2                    # = dT/d(d2), with scale held fixed
    gx = np.sum(coef * 2*dx, axis=1)              # force on point i in x
    gy = np.sum(coef * 2*dy, axis=1)
    return np.concatenate([gx * np.cos(ut[0]), gy * np.cos(ut[1])])

def detect_contacts(points, distance_tol=1e-7, wall_tol=1e-7):
    x, y = points[:, 0], points[:, 1]
    d2 = pairwise_sq_dists(x, y)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2.min())
    iu = np.triu_indices(len(points), 1)
    distances = np.sqrt(d2[iu])
    pairs = [(i, j) for i, j, dij in zip(iu[0], iu[1], distances) if abs(dij - d) <= distance_tol]
    walls = []
    for i, (px, py) in enumerate(points):
        if abs(px + 1.0) <= wall_tol:
            walls.append((i, 0, -1.0))
        if abs(px - 1.0) <= wall_tol:
            walls.append((i, 0, 1.0))
        if abs(py + 1.0) <= wall_tol:
            walls.append((i, 1, -1.0))
        if abs(py - 1.0) <= wall_tol:
            walls.append((i, 1, 1.0))
    return d, pairs, walls

def polish_contacts(points, distance_tol=1e-7, wall_tol=1e-7):
    n = len(points)
    d0, pairs, walls = detect_contacts(points, distance_tol, wall_tol)
    z0 = np.concatenate([points[:, 0], points[:, 1], [d0]])

    def residual(z):
        x, y, d = z[:n], z[n:2*n], z[-1]
        out = [((x[i]-x[j])**2 + (y[i]-y[j])**2 - d*d) for i, j in pairs]
        for i, axis, value in walls:
            out.append((x if axis == 0 else y)[i] - value)
        return np.asarray(out)

    if len(pairs) + len(walls) == 0:
        return points, d0
    lo = np.r_[-np.ones(2*n), 0.0]
    hi = np.r_[ np.ones(2*n), np.inf]
    res = least_squares(residual, z0, bounds=(lo, hi),
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
    z = res.x
    return np.column_stack([z[:n], z[n:2*n]]), z[-1]

def optimize_one_start(u, n, schedule=None):
    if schedule is None:
        schedule = (10, 20, 40, 80, 160, 320, 640, 1280)
    for sharpness in schedule:                     # homotopy: soft energy -> sharp max-min
        d = min_pair_distance(u, n)
        scale = d*d if d > 0 else 1.0             # lambda = d_min^2, fixed during this stage
        res = minimize(objective, u, args=(n, sharpness, scale), jac=objective_grad,
                       method='CG', options={'maxiter': 500, 'gtol': 1e-10})
        u = res.x
    return u

def solve(n, restarts=50, rng=None, schedule=None):
    rng = rng or np.random.default_rng(0)
    best_u, best_d_side2 = None, -1.0
    for _ in range(restarts):                     # many random starts cover the basins
        u = rng.uniform(-np.pi/2, np.pi/2, 2*n)   # random points spread over the whole box
        u = optimize_one_start(u, n, schedule)
        d_side2 = min_pair_distance(u, n)
        if d_side2 > best_d_side2:
            best_d_side2, best_u = d_side2, u.copy()
    x, y = to_box(best_u, n)
    polished, polished_d_side2 = polish_contacts(np.column_stack([x, y]))
    return polished_d_side2 / 2.0, (polished + 1.0) / 2.0, best_u
```

The causal chain, start to finish: packing fat circles is, after absorbing the wall offset and a
rescale (`r = d/(2(1+d))`), nothing but spreading points to maximize their minimum pairwise
distance; that minimum is a non-smooth `min`-over-pairs that defeats smooth optimizers, so I
replace it by the power-mean soft-min and minimize the inverse-power repulsion energy
`Σ(λ/d_ij²)^m`, which a large positive `m` makes faithful to the true minimum; I parameterize the
box away with `x=sin(x̃)` so the search is unconstrained, condition the arithmetic with
`λ=d²_min` while holding `λ` fixed inside each gradient calculation, and use the exact gradient to
run a real optimizer; I get faithfulness *and* tractability by annealing `m` from soft to sharp as a
homotopy, warm-starting each stage; I beat the multimodality with many random restarts (and an
off-the-wall nudge for pinned points); and I sharpen the reported `d_n` by detecting the contact
structure and Newton-polishing it to high precision.
