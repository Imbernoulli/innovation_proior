## Research question

Given an integer `n`, place `n` equal, non-overlapping circles inside the unit square so that
their common radius `r_n` is as large as possible. Solving this for every `n` would answer the
everyday question "how many bottles of a given size fit in a square crate, and how should they be
arranged" — divide the bottle radius by the crate side, compare against the decreasing series
`r_n`, and the largest `n` whose `r_n` still exceeds the quotient is the answer.

There is a cleaner, equivalent way to state the same problem that strips away the circles. A
circle of radius `r` centred at a point must keep its centre at distance `≥ r` from every wall
and at distance `≥ 2r` from every other centre. If the centre square has side `1 − 2r`, rescaling
that centre square to side `1` turns the nearest centre spacing `2r` into
`d = 2r/(1 − 2r)`. Solving `d(1 − 2r) = 2r` gives
`r = d/(2(1 + d))`, hence `r_n = d_n / (2(d_n + 1))`. Equivalently, unit-radius circles fit in a
square of side `L = 2 + 2/d_n`. Maximizing `d_n = min_{i<j} ‖s_i − s_j‖` therefore maximizes
`r_n`, so we may forget the circles and the walls' radius offset and just spread points. This is
the formulation everything below works in.

The objective `d_n` is a **min over all pairs** — a function of the `2n` coordinates that is
piecewise smooth, with kinks wherever the pair achieving the minimum switches. At any
configuration only a handful of pairs are at the minimum distance, and the feasible region is the
box `[0,1]^{2n}`. The broad task is: place `n` points in the unit square so as to maximize the
smallest pairwise distance, and report the best arrangement found together with its `d_n`.

## Background

Packing circles into shapes is an old subject, opened by Fejes Tóth's *Lagerungen* (1972). For
the square, the small cases were settled by hand: `n ≤ 9` in the 1960s (Graham, Meir, Schaer),
with `n ≤ 5` elementary and `6 ≤ n ≤ 9` requiring real work. The flavour of the easy proofs is
the pigeonhole argument: for `n = 5`, cut the unit square into four sub-squares; some sub-square
holds two of the five points, so `d_5` cannot exceed a sub-square diagonal `√2/2`, and the
obvious configuration (four corners plus centre) attains it, so `d_5 = √2/2`. Beyond `n = 9`,
only sporadic cases (`n = 14, 16, 25, 36`) were proved by hand over the following decades.

Two facts about the structure of optima shape intuition. First, for `n = k²` the naive
`k × k` square-lattice arrangement is the obvious candidate, and it is known to be optimal for
`n = 1, 4, 9, 16, 25, 36`; for larger square numbers denser non-lattice packings exist (from
`n = 64` upward). Second, the best arrangements are frequently **almost but not exactly
symmetric**: a configuration obtained by slightly *breaking* an obvious symmetry can beat the
symmetric one.

Because exact proofs run out so quickly, the practical goal becomes **lower-bounding `d_n`**: find
the best arrangement you can by computer, and report its `d_n` as a candidate (a packing that
exists but is not proven optimal). The infinite-`n` limit is governed by the hexagonal lattice
(packing density `π/√12 ≈ 0.9069`), which all finite square packings stay below; the densities of
the best finite packings creep upward toward it as `n` grows. The working picture is a piecewise
objective, a constraint box, a multimodal landscape, exact values only for modest `n`, and best
configurations that tend to be slightly-broken-symmetric.

## Baselines

**Direct radius maximization (de Groot, Peikert, Würtz, 1990–92).** Treat the `n` centres as the
variables and maximize the largest non-overlapping radius they admit — i.e. maximize the
piecewise function `min` over inter-point and point-to-wall distances. They drove this with the
simplex (Nelder–Mead polytope) algorithm and the quasi-Newton BFGS method, and also tried a
stochastic Langevin-equation variant that adds noise to a gradient flow.

**Grid search and contact-graph enumeration.** A literal brute-force attack discretizes the
square and tests many `n`-tuples of grid points, or enumerates possible graphs of touching
circle-circle and circle-wall contacts and solves each graph as a system of contact equations.
The coordinates are continuous, so the grid resolution controls how finely a candidate packing
can be located, and the contact graph (which pairs touch, which circles are loose) must be posited
before each graph's equations are solved.

**Energy minimization on the sphere (Clare–Kepert 1986; Kottwitz 1991).** A parallel community was
spreading points on a *sphere* (closest packing of equal circles on a sphere, spherical codes).
They did not optimize the min-distance directly. Instead they regarded the points as mutually
repelling charges and **minimized a total potential energy** of repulsion, `Σ_{i<j} φ(d_ij)` for a
decreasing `φ`, with a quasi-Newton method (Fletcher–Powell–Davidon) or gradient-plus-Newton.
This replaces the kinky max-min function with a smooth surrogate that is everywhere
differentiable. Having converged to an approximate configuration, they would **identify which
pairs are at minimum distance**, write the contact conditions as a system of equations, and solve
it by Newton–Raphson to sharpen the result. The minimizer of `Σ φ(d_ij)` balances all the
pairwise distances together.

**Billiard / molecular-dynamics simulation (Graham–Lubachevsky 1995–96; Lubachevsky–Stillinger
1990).** A physically-motivated alternative. Put `n` point particles in the box with random
positions and random velocities; give them a common radius that **grows linearly in time**; run an
event-driven simulation of perfectly elastic collisions (particle–particle and particle–wall),
updating velocities at each collision; as the disks swell they collide more, their motion is
increasingly hemmed in, and the system approaches a **jammed** state where the common radius can
grow no further. The jammed radius is a candidate `r_n`. The simulation is event-driven (compute
the next collision time analytically for the growing disks, jump to it) rather than time-stepped,
so it is fast — over a million collisions per CPU-hour on early-1990s hardware. "Rattlers" are
loose particles that never jam (they rattle in a cage of jammed neighbours). Different initial
positions and velocities jam at different radii, so the method is run from many random starts.

## Evaluation settings

The natural yardstick is the value `d_n` (or `r_n = d_n/(2(d_n+1))`) achieved, compared against:
the exact values where they exist (`n ≤ 20`, plus special structured cases such as `25` and `36`);
the strongest previously published candidate arrangements for larger `n`; the square-lattice
value for `n = k²`; and the hexagonal-lattice density as an asymptotic ceiling. Secondary
descriptors of a candidate packing are its **density** (`n·π·r_n²`, since the square has area 1),
the **number of contacts** (circle–circle and circle–wall pairs at the minimum distance, a proxy
for rigidity), the count of **loose / rattler** circles, and the **order of the symmetry group**.
The protocol is: for each `n`, run many independent searches from random starts and keep the best;
for a promising configuration, detect its contact structure and re-solve to high precision so the
reported `d_n` is trustworthy.

## Code framework

The available ingredients are a numerical-optimization library with smooth minimizers and
least-squares/root solvers, stable log-sum primitives, plus array operations for computing all
pairwise distances among `n` points. The missing pieces are the box map and inverse map, the
objective, its gradient, the single-start search, boundary handling, and the contact polish:

```python
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.special import logsumexp

def to_box(u, n):
    # TODO: map optimizer variables in R^{2n} into square coordinates
    pass

def points_to_u(points):
    # TODO: map square coordinates back to optimizer variables
    pass

def pairwise_sq_dists(x, y):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return dx*dx + dy*dy

def min_pair_distance_from_points(points):
    if len(points) < 2:
        return np.inf
    d2 = pairwise_sq_dists(points[:, 0], points[:, 1])
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2.min())

def min_pair_distance(u, n):
    x, y = to_box(u, n)
    return min_pair_distance_from_points(np.column_stack([x, y]))

def radius_from_unit_distance(d_unit):
    if np.isinf(d_unit):
        return 0.5
    return d_unit / (2.0 * (1.0 + d_unit))

def objective(u, n, sharpness, scale):
    # TODO: the scalar objective the optimizer drives at this stage
    pass

def objective_grad(u, n, sharpness, scale):
    # TODO: analytic gradient of the objective, chained through the box map
    pass

def maybe_release_wall_points(u, n, wall_tol=1e-6, inward_step=1e-3, keep_tol=1e-10):
    # TODO: repair a non-rigid boundary point and report whether anything changed
    pass

def detect_contacts(points, distance_tol=1e-7, wall_tol=1e-7):
    # TODO: identify pair contacts and wall contacts from an approximate packing
    pass

def polish_contacts(points, distance_tol=1e-7, wall_tol=1e-7):
    # TODO: solve the contact equations numerically
    pass

def optimize_one_start(u, n, schedule=None):
    # TODO: turn one random start into a locally optimized configuration
    pass

def solve(n, restarts=50, rng=None, schedule=None):
    if n == 1:
        # TODO: return the one-circle configuration
        pass
    rng = rng or np.random.default_rng(0)
    best = None
    for _ in range(restarts):
        u = rng.uniform(-np.pi/2, np.pi/2, 2*n)
        u = optimize_one_start(u, n, schedule)
        # TODO: score the configuration by d_n and keep the best one
    return best
```
