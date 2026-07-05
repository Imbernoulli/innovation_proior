# Harbor Container Port: Budgeted Berth-Plan Trials

## Story

A container terminal tunes its operations on a **digital twin** of the port. Each
candidate operating plan is a decision vector `x` in `[0,1]^n` that sets berth timing,
quay-crane scheduling and yard-stacking parameters. Running the twin on a plan returns
three **costs to minimize**:

- `f1` = vessel turnaround time
- `f2` = quay-crane energy draw
- `f3` = landside yard congestion

These objectives conflict: no single plan minimizes all three. The twin implements a
deterministic **re-centred DTLZ2** surrogate. The first `n_pos = 2` coordinates (the
*position* variables) steer the trade-off **direction** along the cost surface; the
remaining `n - n_pos` coordinates (the *distance* variables) control how close the plan
sits to the ideal cost frontier. The frontier is reached exactly when **every distance
variable equals `center`** (a per-instance calibrated operating point, not always 0.5),
at which point `(f1,f2,f3)` lands on the positive-octant unit sphere
`f1^2 + f2^2 + f3^2 = 1`.

You may only afford to trial **`budget`** plans on the twin. Choose a batch that maps out
the best possible set of trade-offs.

## Objective (MAXIMIZE)

The quality of a batch is its **exact dominated hypervolume** of the resulting cost
vectors with respect to a fixed, **asymmetric** reference (worst-tolerable) cost point
`ref = [ref1, ref2, ref3]`: the volume of cost space dominated by at least one trialled
plan and bounded above by `ref`. High hypervolume requires BOTH pushing plans onto the
cost frontier AND spreading them to cover complementary trade-offs — and, because `ref`
is skewed per objective, the ideal spread is instance-specific.

## Public instance JSON (stdin)

```json
{
  "surrogate": "dtlz2_recentred",
  "M": 3,
  "n": 5,
  "n_pos": 2,
  "budget": 16,
  "ref": [1.10, 1.15, 1.08],
  "center": 0.5,
  "note": "..."
}
```

- `M` — number of objectives (always 3).
- `n` — decision-vector length; `n_pos` position vars then `n - n_pos` distance vars.
- `budget` — maximum number of plans you may return.
- `ref` — asymmetric reference cost point (upper bound of the hypervolume box).
- `center` — value each distance variable must equal to reach the cost frontier.

## Answer JSON (stdout)

```json
{"points": [[x_1, ..., x_n], [x_1, ..., x_n], ...]}
```

- A list of **1 to `budget`** decision vectors, each of length `n`.
- Every coordinate must lie in `[0, 1]`. Non-numeric, `NaN`, `inf`, wrong-length, or
  out-of-range vectors, or more than `budget` of them, make the whole answer infeasible
  (score 0).

## Scoring

The evaluator computes the surrogate and the exact 3D hypervolume itself. Let `obj` be
your batch's dominated hypervolume and `b` the hypervolume of the single domain-midpoint
plan `x = [0.5]*n` (the trivial construction). Your normalized score on an instance is

```
r = min(1, 0.1 * obj / b)
```

so reproducing only the single midpoint point scores exactly `0.1`, and a well-spread
batch scores higher (capped at 1.0). Infeasible or malformed answers score 0. The final
`Ratio` is the mean of `r` over all (public and held-out) instances.

## Notes

- Scoring is fully deterministic — no wall-clock, GPU, or randomness.
- Your program is run in an isolated subprocess and only ever sees the public instance;
  the surrogate, reference, and scorer stay in the evaluator process.
