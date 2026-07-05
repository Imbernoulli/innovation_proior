# Solar-Farm Inverter Design Sweep (budgeted multi-objective hypervolume)

An engineering team is commissioning a fleet of grid-tie inverters for a solar farm.
Each candidate **design configuration** (switching / DC-link / thermal / control
settings) is a decision vector `x` with `n` coordinates, each in `[0, 1]`. A
deterministic cost surrogate turns any configuration into `M` competing costs to
**minimize**:

- `f1` = capital / bill-of-materials cost
- `f2` = conversion power loss (heat)
- `f3` = harmonic distortion / THD (only when `M = 3`)

The surrogate is a **biased (DTLZ4-style) frontier surface**:

```
g(x)     = sum_{i = n_pos .. n-1} (x_i - 0.5)^2          # "distance" cost
theta_j  = (x_j ** alpha) * (pi/2)   for j = 0 .. n_pos-1 # biased "position" angles
```

For `M = 3` (`n_pos = 2`):

```
f1 = (1 + g) * cos(theta_0) * cos(theta_1)
f2 = (1 + g) * cos(theta_0) * sin(theta_1)
f3 = (1 + g) * sin(theta_0)
```

For `M = 2` (`n_pos = 1`):

```
f1 = (1 + g) * cos(theta_0)
f2 = (1 + g) * sin(theta_0)
```

- The first `n_pos` coordinates steer the **trade-off direction** along the cost
  surface, but they enter through the **biased power map** `theta_j = (x_j**alpha)*(pi/2)`.
  When `alpha > 1` this map is strongly non-linear, so a uniform grid in decision
  space lands as a **clustered, uneven** spread on the cost frontier. To spread
  evenly you must invert the bias (`x_j = target**(1/alpha)`).
- The remaining `n - n_pos` coordinates are **distance variables** (thermal / control
  margins); their cost `g` is minimized (= 0) exactly when each equals `0.5`. When
  `g = 0` the configuration is Pareto-optimal and lands on the **unit-radius cost
  frontier** `sum_i f_i^2 = 1` in the positive orthant.

You may only afford to prototype **`budget`** configurations. Propose that batch.

## Objective (maximize)

The batch is scored by its **exact dominated hypervolume** with respect to a fixed
worst-tolerable reference cost `ref = [R, ..., R]` (length `M`): the volume of cost
space dominated by at least one prototyped configuration and bounded above by `ref`
(minimization convention). A large hypervolume requires BOTH pushing configurations
onto the cost frontier (set distance variables near `0.5`) AND spreading them across
complementary trade-offs. This is open-ended: for a fixed budget, the optimal
placement of points on the curved, bias-warped frontier is a hard continuous global
optimization problem, and uniform grids, bias-inverted angle-even spreads, and local
hypervolume ascent all give different volumes.

## Input (public instance, one JSON object on stdin)

```json
{
  "surrogate": "dtlz_biased",
  "M": 3,                    // number of objectives (2 or 3)
  "n": 5,                    // decision-vector dimension
  "n_pos": 2,                // number of trade-off (position) variables = first coords (= M-1)
  "alpha": 2.0,              // bias exponent of the position power map
  "budget": 20,              // max configurations you may prototype
  "ref": [1.12, 1.12, 1.12], // reference (worst-tolerable) cost point, length M
  "note": "..."
}
```

## Output (one JSON object on stdout)

```json
{"points": [[x_1, ..., x_n], ...]}
```

- `points` is a list of **at least 1 and at most `budget`** configurations.
- Each configuration is a list of exactly `n` numbers, each in `[0, 1]`.
- Any out-of-range coordinate, wrong length, missing `points`, empty batch, more than
  `budget` configurations, or non-finite value makes the whole answer **infeasible
  (score 0)**.

## Scoring

Let `obj` be the exact `M`-dimensional dominated hypervolume of your batch's cost
vectors w.r.t. `ref`, and let `b` be the hypervolume of the single midpoint
configuration `x = [0.5]*n` (one Pareto-optimal but un-spread prototype). Your
per-instance score is

```
r = min(1, 0.1 * obj / b)
```

so a batch that only reproduces the single midpoint point scores `~0.1`, and a batch
whose spread earns `k` times the single-point volume scores `min(1, 0.1*k)`. The final
score is the mean of `r` over all instances (a fixed, seeded set of 12 instances mixing
`M = 2` and `M = 3` fronts and varying bias exponents, budgets, distance-variable
counts, and references, including harder held-out instances). Scoring is fully
deterministic.

## Isolation

Your program is run in an isolated subprocess: it reads only the public instance from
stdin and writes only the answer to stdout. It cannot see the surrogate evaluation, the
hypervolume computation, or any evaluator state.
