# Kepler Invariant Discovery

## Problem
A test particle moves in a plane under the gravity of a central mass with an
**unknown** gravitational parameter `mu`. Its state is `(x, y, vx, vy)`. You are
given noisy state samples drawn from several **training orbits**; the samples that
share a `traj_id` all lie on the same physical orbit.

Discover a **conserved quantity**: a closed-form expression
`C(x1, x2, x3, x4)` (with `x1=x, x2=y, x3=vx, x4=vy`) that stays (nearly) constant
*along* each orbit, while still differing *between* orbits. You never see `mu`, the
orbital elements, or the grading orbits — you must infer the invariant from data.

## Input (stdin)
```
n_traj n_per_traj test_id
traj_id  x1  x2  x3  x4          (repeated n_traj*n_per_traj times)
...
```
Rows sharing a `traj_id` lie on the same orbit. Values are floats with measurement
noise (noise grows with `test_id`).

## Output (stdout)
A single line: one closed-form expression in the variables `x1,x2,x3,x4`.
Allowed operators: `+ - * / ** %` and functions
`exp, log, sin, cos, sqrt, tanh, atan, abs`. Numeric constants are allowed.
No other names, calls, imports, or attributes. The result must be finite on every
evaluation point.

## Feasibility
The expression must parse under the whitelist above, use only the stated
variables/functions, and evaluate to a finite real number on every grading point.
A **constant** expression (zero variance across all points) is degenerate and
scores 0.

## Objective (maximize)
Grading uses a fresh, **larger** set of held-out orbits (an extrapolation region,
regenerated deterministically inside the grader). Group the held-out points by
orbit. With `SS_between` the between-orbit variance and `SS_total` the pooled
variance of your expression's values, the conservation quality is the correlation
ratio
```
eta2 = SS_between / SS_total   in [0, 1].
```
`eta2 -> 1` means the quantity is essentially constant within each orbit yet varies
across orbits — i.e. a genuine first integral of the motion. Measurement noise
keeps even the true invariants strictly below 1.

## Scoring
```
F   = eta2_you      / (1 + LAMBDA * complexity_you)
B   = eta2_baseline / (1 + LAMBDA * complexity_baseline)   # baseline = x3**2 + x4**2
Ratio = min(1000, 100 * F / B) / 1000
```
with `LAMBDA = 0.006`. Reproducing the partially-conserved baseline `v^2` scores
about `0.1`; a genuine invariant scores several times higher; the noise floor
caps the best achievable score below `1.0`. Any feasibility violation, non-finite
value, or degenerate constant scores `0.0`.

## Constraints
- `1 <= test_id <= 10` (difficulty ladder: more noise for larger ids).
- Expression length `<= 200000` bytes, a single non-empty line.
- Deterministic scoring: the grader re-derives the ground truth and the held-out
  orbits solely from `test_id`.

## Example (illustrative FORM only — not the hidden law)
If your line were
```
sin(x1) + x3*x4 / (1 + abs(x2))
```
the grader would evaluate it on every held-out orbit, compute `eta2`, apply the
complexity penalty, and normalize against the baseline. This example expression is
just to show the output *format*; it is **not** the conserved quantity you are
looking for — you must discover that from the data.
