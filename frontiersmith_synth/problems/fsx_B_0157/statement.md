# Bakery Supply-Chain Recipe Trials (budgeted multi-objective hypervolume)

An artisan bakery is redesigning a product line. Each candidate **configuration**
(a recipe + sourcing + logistics setting) is a decision vector `x` with `n` coordinates,
each in `[0, 1]`. A deterministic 3-objective cost surrogate turns any configuration into
three costs that the bakery wants to **minimize**:

- `f1` = ingredient / production cost
- `f2` = delivery lead-time / staleness
- `f3` = spoilage / waste

The surrogate is a **shifted-radius DTLZ2 surface**:

```
g(x)  = sum_{i = n_pos .. n-1} (x_i - 0.5)^2          # "distance" cost
theta_j = x_j * (pi/2)   for j = 0 .. n_pos-1         # "position" (trade-off) angles
f1 = (1 + g) * cos(theta_0) * cos(theta_1)
f2 = (1 + g) * cos(theta_0) * sin(theta_1)
f3 = (1 + g) * sin(theta_0)
```

- The first `n_pos` coordinates steer the **trade-off direction** along the cost surface.
- The remaining `n - n_pos` coordinates are **distance variables**; their cost `g` is
  minimized (= 0) exactly when each equals `0.5`. When `g = 0` the configuration is
  Pareto-optimal and lands on the **unit-radius cost frontier**
  `f1^2 + f2^2 + f3^2 = 1` in the positive octant.

You may only afford to trial **`budget`** configurations. Propose that batch.

## Objective (maximize)

The batch is scored by its **exact dominated hypervolume** with respect to a fixed
worst-tolerable reference cost `ref = [R, R, R]`: the volume of cost space that is
dominated by at least one trialled configuration and bounded above by `ref`
(minimization convention). A large hypervolume requires BOTH pushing configurations onto
the cost frontier (set distance variables near `0.5`) AND spreading them so they cover
complementary trade-offs. This is open-ended: for a fixed budget, the optimal placement
of points on the curved frontier is a hard continuous global-optimization problem, and
uniform grids, angle-even spreads, and local hypervolume ascent all give different
volumes.

## Input (public instance, one JSON object on stdin)

```json
{
  "surrogate": "dtlz2",
  "M": 3,                    // number of objectives (always 3)
  "n": 6,                    // decision-vector dimension
  "n_pos": 2,                // number of trade-off (position) variables = first coords
  "budget": 20,              // max configurations you may trial
  "ref": [1.12, 1.12, 1.12], // reference (worst-tolerable) cost point
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

Let `obj` be the exact 3D dominated hypervolume of your batch's cost vectors w.r.t.
`ref`, and let `b` be the hypervolume of the single midpoint configuration
`x = [0.5]*n` (one Pareto-optimal but un-spread trial). Your per-instance score is

```
r = min(1, 0.1 * obj / b)
```

so a batch that only reproduces the single midpoint point scores `~0.1`, and a batch
whose spread earns `k` times the single-point volume scores `min(1, 0.1*k)`. The final
score is the mean of `r` over all instances (a fixed, seeded set of 12 instances with
varying distance-variable counts, budgets, and references, including harder held-out
instances). Scoring is fully deterministic.

## Isolation

Your program is run in an isolated subprocess: it reads only the public instance from
stdin and writes only the answer to stdout. It cannot see the surrogate evaluation, the
hypervolume computation, or any evaluator state.
