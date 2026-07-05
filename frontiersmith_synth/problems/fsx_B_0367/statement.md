# Orbital Debris Cleanup Mission Portfolio (budgeted 4-objective hypervolume)

An orbital-debris-removal operator is planning a cleanup campaign. Each candidate
**mission profile** (a capture/rendezvous + deorbit maneuver setting) is a decision
vector `x` with `n` coordinates, each in `[0, 1]`. A deterministic **4-objective** cost
surrogate turns any profile into four campaign costs the operator wants to **minimize**:

- `f1` = propellant / delta-v budget
- `f2` = time-to-deorbit (mission duration)
- `f3` = collision / operational risk
- `f4` = residual debris cross-section left uncleared

The surrogate is a **shifted-radius DTLZ2 hypersurface** with `M = 4` objectives:

```
g(x)     = sum_{i = n_pos .. n-1} (x_i - 0.5)^2           # "distance" cost
theta_j  = x_j * (pi/2)   for j = 0 .. n_pos-1            # "position" (trade-off) angles
f1 = (1 + g) * cos(theta_0) * cos(theta_1) * cos(theta_2)
f2 = (1 + g) * cos(theta_0) * cos(theta_1) * sin(theta_2)
f3 = (1 + g) * cos(theta_0) * sin(theta_1)
f4 = (1 + g) * sin(theta_0)
```

- The first `n_pos` coordinates (`n_pos = 3`) steer the **trade-off direction** across
  the cost hypersurface.
- The remaining `n - n_pos` coordinates are **distance variables**; their cost `g` is
  minimized (= 0) exactly when each equals `0.5`. When `g = 0` the profile is
  Pareto-optimal and lands on the **unit-radius cost frontier**
  `f1^2 + f2^2 + f3^2 + f4^2 = 1` in the positive orthant.

You may only afford to fly **`budget`** mission profiles. Propose that portfolio.

## Objective (maximize)

The portfolio is scored by its **exact dominated 4D hypervolume** with respect to a
fixed worst-tolerable reference cost `ref = [R, R, R, R]`: the volume of 4-objective
cost space that is dominated by at least one flown profile and bounded above by `ref`
(minimization convention). A large hypervolume requires BOTH pushing profiles onto the
cost frontier (set distance variables near `0.5`) AND spreading them so they cover
complementary trade-offs across the curved 3D frontier. This is open-ended: for a fixed
budget, the optimal placement of points on the frontier is a hard continuous
global-optimization problem, and uniform grids, angle-even spreads, and local
hypervolume ascent all yield different volumes.

## Input (public instance, one JSON object on stdin)

```json
{
  "surrogate": "dtlz2",
  "M": 4,                          // number of objectives (always 4)
  "n": 6,                          // decision-vector dimension
  "n_pos": 3,                      // number of trade-off (position) variables = first coords
  "budget": 16,                    // max mission profiles you may fly
  "ref": [1.10, 1.10, 1.10, 1.10], // reference (worst-tolerable) cost point
  "note": "..."
}
```

## Output (one JSON object on stdout)

```json
{"points": [[x_1, ..., x_n], ...]}
```

- `points` is a list of **at least 1 and at most `budget`** mission profiles.
- Each profile is a list of exactly `n` numbers, each in `[0, 1]`.
- Any out-of-range coordinate, wrong length, missing `points`, empty portfolio, more
  than `budget` profiles, or non-finite value makes the whole answer **infeasible
  (score 0)**.

## Scoring

Let `obj` be the exact 4D dominated hypervolume of your portfolio's cost vectors w.r.t.
`ref`, and let `b` be the hypervolume of the single midpoint profile `x = [0.5]*n` (one
Pareto-optimal but un-spread mission). Your per-instance score is

```
r = min(1, 0.1 * obj / b)
```

so a portfolio that only reproduces the single midpoint point scores `~0.1`, and a
portfolio whose spread earns `k` times the single-point volume scores `min(1, 0.1*k)`.
The final score is the mean of `r` over all instances (a fixed, seeded set of 12
instances with varying distance-variable counts, budgets, and references, including
harder held-out instances). Scoring is fully deterministic.

## Isolation

Your program is run in an isolated subprocess: it reads only the public instance from
stdin and writes only the answer to stdout. It cannot see the surrogate evaluation, the
hypervolume computation, or any evaluator state.
