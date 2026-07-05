# Saddle Optimizer: Duality-Gap Descent

## Task
Design the **update rule** of a first-order optimizer for a smooth convex–concave
**saddle problem**

```
min_x  max_y   L(x,y) = 1/2 x^T P x  +  x^T B y  -  1/2 y^T Q y  +  a^T x  -  c^T y
```

with `P = P^T > 0` (strongly convex in `x`) and `Q = Q^T > 0` (strongly concave in
`y`). The problem has a unique saddle point `(x*, y*)` where both partial gradients
vanish. Plain gradient descent–ascent (GDA) tends to **oscillate or diverge** here:
the bilinear coupling `B` rotates the joint gradient field, so a naive optimizer
spirals outward. Taming that rotation — with optimism, extrapolation, asymmetric
per-player step sizes, and (negative) momentum — is exactly what you must do.

You do **not** run the optimizer yourself. The evaluator executes a fixed-form
update for `T` steps using **four constant coefficients that you supply**, then
measures how small a duality gap your coefficients achieve.

### The fixed-form update the evaluator runs for you
From the given start `x_0, y_0` (with gradient memory seeded `gx_{-1}=gx_0`,
`gy_{-1}=gy_0`), for `t = 0, 1, …, T-1`:

```
gx_t = P x_t + B y_t + a           # dL/dx  (descend on x)
gy_t = B^T x_t - Q y_t - c         # dL/dy  (ascend on y)
dx   = (1 + theta) gx_t - theta gx_{t-1}      # optimistic extrapolation
dy   = (1 + theta) gy_t - theta gy_{t-1}
x_{t+1} = x_t - eta_x * dx + alpha * (x_t - x_{t-1})   # + heavy-ball momentum
y_{t+1} = y_t + eta_y * dy + alpha * (y_t - y_{t-1})
```

Special cases: `theta=0, alpha=0` → plain GDA; `theta=1` → optimistic/extra-gradient
flavor; `alpha<0` → negative momentum (damps rotation); `eta_x != eta_y` →
asymmetric steps for ill-conditioning. The coefficients are **constant over all `T`
steps**, so the dynamics are a fixed linear recurrence: convergence is at best
**geometric** (`gap ~ rho^T`) and never exact in a finite budget.

## Public instance (stdin, one JSON object)
```json
{
  "name": "saddle101",
  "n": 6,
  "P": [[...], ...],   "Q": [[...], ...],   "B": [[...], ...],
  "a": [...], "c": [...],
  "x0": [...], "y0": [...],
  "T": 55
}
```
`P`, `Q`, `B` are `n x n` float matrices; `a`, `c`, `x0`, `y0` are length-`n`
vectors; `T` is the number of update steps.

## Answer (stdout, one JSON object)
```json
{"eta_x": 0.03, "eta_y": 0.02, "theta": 0.75, "alpha": -0.3}
```
A **valid** answer is a JSON object whose four fields are finite real numbers with
```
0 < eta_x, eta_y <= 100 ,   0 <= theta <= 10 ,   -1 < alpha < 1 .
```
Any missing key, wrong type, non-finite value, out-of-range value, non-JSON output,
crash, or timeout scores **0.0** on that instance.

## Objective — minimize the final duality gap
After running your coefficients through the fixed-form update for `T` steps, the
evaluator computes the exact **duality gap**

```
gap(x_T, y_T) = [ max_y L(x_T, y) ] - [ min_x L(x, y_T) ]  >= 0
```

via the closed-form inner optima `y*(x) = Q^{-1}(B^T x - c)` and
`x*(y) = -P^{-1}(B y + a)`. **Smaller gap is better.**

## Scoring (deterministic)
Per instance the gap is normalized in log-space against a weak reference (plain GDA
at a default step, anchored to `~0.1`) and an unreachable ideal gap:

```
Lb = log10( gap of default GDA baseline )
Li = log10( GAP_IDEAL )                       # below any reachable gap
Lc = log10( gap of your coefficients )
r  = clamp( 0.1 + 0.9 * (Lb - Lc) / max(Lb - Li, 1.0),  0, 1 )
```

Matching the default baseline scores `~0.1`; reducing the gap by orders of
magnitude climbs toward (but does not reach) `1.0`; diverging scores below `0.1`
(clamped at `0`). The final score is the mean of `r` over 12 instances (8 base +
4 harder held-out with larger `n` and stronger coupling). Because the coefficients
must be constant, even the best possible choice leaves a positive gap — there is
always headroom.

## Notes
- The candidate runs in an isolated sandbox and sees only the public instance; the
  baseline gap, the ideal target, and the duality-gap computation live in the
  evaluator. You may freely simulate the update yourself (the model matrices are
  public) to tune your four coefficients.
- Everything is deterministic; there is no wall-clock component to the score.
