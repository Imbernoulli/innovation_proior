# One Dimensionless Knob

A calibration bench measures a positive response `y` as four positive knobs
`x1, x2, x3, x4` are varied. Each knob carries a fixed integer **grading**
across 3 abstract calibration axes (given to you as a 3x4 integer matrix
`U`, one column per knob). The bench is built so that `y` itself is
**grading-neutral**: any valid model for `y` must, taken in logs, have its
knob-exponents form a vector in the **null space of `U`**. Because `U` has 3
independent rows over 4 knobs, that null space is exactly **1-dimensional**
— there is a unique (up to overall scale) integer exponent vector `b` with
`U @ b = 0`. Consequently the bench's hidden law has the shape

```
y = C * Pi ** p,      Pi = x1**b1 * x2**b2 * x3**b3 * x4**b4
```

for the single dimensionless group `Pi` fixed by `U`, and unknown constants
`p` (exponent) and `C` (amplitude) that you must fit from data.

## Input (stdin)
- Line 1: `n testId`.
- Next 3 lines: the grading matrix `U`, 4 integers each (one row per axis).
- Next `n` lines: `x1 x2 x3 x4 y`, one noisy measurement each (floats,
  `x_i > 0`, `y > 0`).

The campaign sweeps each knob independently over a **narrow** multiplicative
band around its own center — no single raw knob, and no combination, is
explored over a wide range during training.

## Output (stdout)
One line: a closed-form Python expression for `y` in the variables `x1`,
`x2`, `x3`, `x4`. Allowed: `+ - * / **`, unary `-`, numeric constants, and
the functions `sqrt log exp absv`. Example (illustrative **form only — NOT
the hidden law**): `1.2*sqrt(x2) - 0.3*log(x4) + x1`. No other names are
accepted.

## Scoring (deterministic, minimization)
Your expression is evaluated on a **held-out grid**, regenerated inside the
grader, where each of the four knobs is pushed far outside its training
band. Let `p_i` be your prediction and `t_i` the true (noisy) response at
held-out point `i`:

```
metric   = mean_i  min(1, |p_i - t_i| / (|p_i| + |t_i|))     # bounded rel. error
O        = metric * (1 + LAMBDA * nodes)                     # nodes = expr size
baseline = the same metric for the constant predictor geomean(train y)
Ratio    = min(1000, 100 * baseline / O) / 1000
```

Lower held-out error gives a higher `Ratio` (capped at `1.0`). A constant
predictor scores about `0.1`. `LAMBDA` is a small parsimony weight. Non-finite
or complex-valued predictions score `0`.

## Why the obvious fit is a trap
The obvious move is to fit a generic power law on the raw knobs — least
squares on `log y ~ c0 + a1*log(x1) + ... + a4*log(x4)` — ignoring the
grading matrix as flavor text. This matches the narrow training band fine:
with a small, near-orthogonal sweep in each raw knob, four free exponents
track the data about as well as one. But the true law depends on `y`
through only ONE combination of the knobs; the other three log-knob
directions have **zero** true effect, yet the free fit's four exponents
still absorb sampling noise along all of them. Held out, each knob is
pushed independently — mostly a mix of genuine motion along the true
dimensionless direction *and* large motion orthogonal to it with no effect
on `y` at all — and the noise picked up in those orthogonal directions gets
amplified by that large, task-irrelevant motion. The recipe never checks
whether the grading matrix says some of its four "signals" are pure noise.

The insight is to run the dimensional analysis **first**: solve `U @ b = 0`
exactly (an integer null-space computation, no data needed for the
*direction*), form the single group `Pi`, and fit only the remaining scalar
`p`. That single-parameter fit only ever asks "how far did `Pi` move", so it
is blind to the wasted orthogonal motion that sinks the free fit, and the
same closed form keeps working however far the knobs are extrapolated.

## Constraints
- Time limit 5 s, memory 512 MB; `n` is a few dozen rows.
- Held-out noise leaves irreducible error, so even the correct law does not
  reach `Ratio = 1.0` — there is room above the reference solutions.
