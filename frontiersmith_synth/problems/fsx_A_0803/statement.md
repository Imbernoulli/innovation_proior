# Collapse to the Curie Point

A ferromagnetic sample is measured near its **hidden Curie temperature** `Tc`.
At temperature `T` and applied field `h > 0` the (noisy) order-parameter
magnitude `m` obeys one hidden scaling law:

```
m(T, h) = A * |Tc - T|^beta * F( h / |Tc - T|^phi )
```

where `beta` and `phi` are fixed critical exponents, `A` is an amplitude, and
`F` is a fixed scaling function. Because `beta` is not an integer, `|Tc-T|^beta`
is **non-analytic** at `T = Tc`: no polynomial, exponential, or other smooth
function of `T` agrees with it on both sides of the transition.

## Input (stdin)
- Line 1: two integers `n` and a case id.
- Next `n` lines: `T h m`, one measurement each (floats).

The recorded campaign is a **single facility on one side of the transition**:
every training row has `T < Tc`. The field `h` is swept widely, but the
*distance to the transition* `Tc - T` only varies over a moderate band — the
campaign never gets close to `Tc` and never crosses it.

## Output (stdout)
One line: a closed-form Python expression for `m` in the variables `T`, `h`.
Allowed: `+ - * / **`, unary `-`, numeric constants, and the functions
`sqrt log exp sig tanh absv`. Example (illustrative **form only — NOT the
hidden law**): `0.4*tanh(h) + 0.05*T**2`. No other names are accepted.

## Scoring (deterministic, minimization)
Your expression is evaluated on a **held-out grid**, regenerated inside the
grader, where most points sit **much closer to `Tc`** than any training point
ever was, split evenly across **both sides of the transition** (including
`T > Tc`, never seen in training). Let `p_i` be your prediction and `t_i` the
true (noisy) magnetization at held-out point `i`:

```
metric   = mean_i  min(1, |p_i - t_i| / (|p_i| + |t_i|))     # bounded rel. error
O        = metric * (1 + LAMBDA * nodes)                     # nodes = expr size
baseline = the same metric for the constant predictor geomean(train m)
Ratio    = min(1000, 100 * baseline / O) / 1000
```

Lower held-out error gives a higher `Ratio` (capped at `1.0`). A constant
predictor scores about `0.1`. `LAMBDA` is a small parsimony weight, so an
overgrown expression is penalized. Non-finite or complex-valued predictions
score `0`.

## Why the obvious fit is a trap
A generic smooth regression — fit `m` as a polynomial in `T` (with a
log-field term thrown in) by ordinary least squares — matches the training
band nicely, because the band is narrow and far from the transition. But
polynomials have no cusp: they cannot reproduce the `|Tc-T|^beta` singularity
as `T` approaches `Tc`, and they do **not fold** onto `|T-Tc|` when `T`
crosses to the other side. Held out close to the transition, and on the
untrained side, this recipe is far off.

The insight is to stop fitting curves and instead **search for the change of
variables that collapses them**. Plot `m / |Tc-T|^beta` against the combined
variable `x = h / |Tc-T|^phi` for a *trial* `(Tc, phi)`: for the WRONG trial
the different field-sweeps at different distances-to-`Tc` scatter as separate
curves; for the RIGHT `(Tc, phi, beta)` they all fall onto **one** curve
`F(x)`. Locating that collapse identifies `Tc` and the exponents directly —
and because the collapsed form is exact (not a local polynomial patch), the
same formula extrapolates correctly to distances no training point reached,
and to the far side of the transition.

## Constraints
- Time limit 5 s, memory 512 MB; `n` up to a few hundred rows.
- Held-out noise leaves irreducible error, so even the correct law does not
  reach `Ratio = 1.0` — there is room above the reference solutions.
