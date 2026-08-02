# Band Gap of a Compound Nobody Made

A materials-discovery lab hands you a logbook: a host semiconductor doped
with a small VISIBLE family of trace elements. For each element they swept
the doping fraction `x` across a narrow, low-concentration window and
recorded the resulting band gap `y` (with sensor noise). Each element also
comes with two elemental descriptors relative to the host: its
electronegativity mismatch `dEN` and its covalent-radius mismatch `dR`
(both fixed per element, independent of `x`).

Your job: fit a closed-form predictor `y = f(x, dEN, dR)`. It will be graded
on a HELD-OUT set that pushes into territory your data never showed you:
doping fractions beyond the visible window, and — critically — brand-new
dopant elements whose electronegativity/radius mismatch fall well outside
anything in your logbook. A curve that only memorises the shape of the
visible composition sweep, and ignores *which* element produced it,
extrapolates badly the moment the chemistry changes.

## Input (stdin)

```
testId n
idx_0 x_0 dEN_0 dR_0 y_0
idx_1 x_1 dEN_1 dR_1 y_1
...
```

`n` training rows follow the header. `idx` is an arbitrary integer
identifying which visible dopant element produced that row (several rows
share the same `idx`, one per swept `x`). `dEN`, `dR`, `y` are floats.

## Output (stdout)

Print ONE line: a Python-syntax arithmetic expression in the variables
`x`, `dEN`, `dR`. Allowed: `+ - * /`, parentheses, `**` (exponent must be a
literal constant with magnitude ≤ 6), numeric constants (magnitude ≤ 1e6),
and the unary functions `sin cos tanh abs exp log sqrt`. At most 90
expression nodes total. No other names, calls, or syntax.

**Illustrative FORM only — NOT the hidden law:**
```
sin(2 * x) + 0.5 * abs(dEN - dR) - exp(-x)
```
This just shows the allowed syntax; the real relationship between `y` and
`(x, dEN, dR)` has a different shape and you must discover it from the data.

## Feasibility

The output must parse under the grammar above and reference only known
names. During grading it is evaluated on every held-out row; if evaluation
raises an error or ever produces a non-finite value, or the expression
violates any syntax/name/magnitude/size rule, the submission scores `0`.

## Objective (maximise)

Let `RMSE` be the root-mean-squared error of your expression against the
true held-out band gaps, and `nodes` the size of your expression. The
grader builds its own internal baseline: an ordinary-least-squares straight
line `y = a + b*x` fit to YOUR training rows (ignoring the descriptors),
evaluated on the same held-out set with `RMSE_base`. It forms

```
F = 1 / ((RMSE + 0.03) * (1 + 0.01 * max(0, nodes - 40)))
B = 1 / (RMSE_base + 0.03)
Ratio = min(1000, 100 * F / B) / 1000
```

Reproducing the straight-line baseline gives `Ratio ≈ 0.1`. Lower held-out
error raises the score; an oversized expression is mildly taxed.

## Why the visible sweep is a trap

Within the visible family, the composition trend (including its real
curvature) is easy to nail — a plain `x`/`x²` fit already looks excellent,
because every visible dopant's chemistry-specific contribution is small. But
that contribution does not stay small: it grows sharply for chemistry far
from what you were shown, and a fit with zero dependence on `dEN`/`dR` has
no way to see it coming. The signature of a mismatch-driven mechanism is
residuals of a memoryless composition-only fit that correlate with the
dopant's descriptors, not with `x` alone.

## Constraints

`n` is at most a few hundred rows. Time limit 5 s, memory 512 MB. All
scoring is fully deterministic — the grader's held-out split and hidden law
are fixed functions of `testId`.
