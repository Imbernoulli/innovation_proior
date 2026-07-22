# The Silo Master's Secret Gauge

An old grain silo master kept a private logbook of every hopper he ever
commissioned: for a given **aperture width** `D` (cm), **grain diameter** `d`
(cm), and grain **bulk density** `rho` (g/cm^3), he wrote down the measured
**discharge rate** `Q` (g/s) at which grain poured out. He never wrote down
the formula he used to size new hoppers — only these readings, all taken on
**small test hoppers**, aperture `D` between 6 and 20 cm.

His apprentices are now asked to size a new **large** hopper, aperture `D`
between 40 and 120 cm, for grain sizes the old logbook never tested. Your job:
recover a closed-form gauge `Q(D, d, rho)` from the small-hopper log that
still holds at large-hopper scale.

Three physical facts are known to hold, all shaping `Q` at once, in every
reading:

1. **Density enters exactly linearly.** Discharge rate is mass per volume
   times volume flow rate, so doubling `rho` (packing the same grain shape
   denser) exactly doubles `Q` — this is dimensional necessity, not
   something to curve-fit.
2. **The aperture has a hidden offset.** Grains near the rim of the opening
   cannot reach its center, so the flow only "sees" an *effective* aperture,
   narrower than the raw `D` by some multiple of the grain diameter `d`.
3. **The effective aperture obeys a power law.** Once you use the *effective*
   aperture (not the raw one), discharge rate scales as that effective width
   raised to a fixed real exponent — a genuine physical exponent, not
   necessarily a round number, and not the same on every hopper (it depends
   on the cone angle and grain shape, which differ silo to silo).

## Input (stdin)
```
t  n
D_0  d_0  rho_0  Q_0
D_1  d_1  rho_1  Q_1
...
D_(n-1)  d_(n-1)  rho_(n-1)  Q_(n-1)
```
`t` is the test id; `n` small-hopper log rows follow (floats). The held-out
grading grid is a **different, larger** set of hoppers and grain sizes for
the same silo master's formula; it is NOT given to you.

## Output (stdout)
One line: a closed-form expression for `Q` in variables `D`, `d`, `rho`.
Allowed: `+ - * /`, unary `-`, parentheses, numeric constants, and the
functions `absv(a)` (absolute value) and `powv(a, b)` (computes `a` to the
power `b`; `a` must evaluate to a strictly positive number, otherwise the
submission is rejected). No other names are accepted — in particular `**`
is NOT allowed; use `powv`.

**Illustrative FORM only — NOT the hidden gauge:**
`powv(D, 2) + absv(d - rho)`
This only shows the syntax; the real gauge's offset and exponent are
different and must be discovered from the data.

## Scoring (deterministic, maximisation)
Your expression is evaluated on a held-out grid of large hoppers and unseen
grain sizes, regenerated deterministically inside the grader from the same
test id — you never see it. Let `p_i` be your prediction and `t_i` the true
(noisy) discharge at held-out point `i`:
```
F     = mean_i (log(p_i) - log(t_i))^2 * (1 + LAMBDA * nodes)   # nodes = expr size
B     = mean_i (log(qbar) - log(t_i))^2 * (1 + LAMBDA * 1)      # qbar = geometric
                                                                  # mean of YOUR
                                                                  # OWN training Q
Ratio = min(CAP, 0.1 * (B / F) ** GAMMA)
```
with small fixed `LAMBDA, GAMMA > 0` and a hard cap `CAP < 1`. Predicting the
flat training geometric mean everywhere gives `B/F == 1` (`Ratio == 0.1`);
recovering the true growth law drives `F` down and raises the ratio. The
sub-linear `GAMMA` keeps a merely-right-shaped gauge from saturating even
though `B/F` can span a wide range once the shape is close. Held-out sensor
noise keeps even a correctly-shaped gauge below the cap. Non-finite,
non-positive, or complex-valued predictions score `0`.

## Why the obvious fit is a trap
The log has three numeric predictor columns and one numeric target, so the
obvious move is a single global **multiplicative** power law,
`Q = k0 * D^a * d^b * rho^c`, fit by log-log least squares over all three at
once. It recovers `rho`'s exponent fine and tracks the small-hopper log's
scale reasonably. But a multiplicative form can never express an *additive*
offset between `D` and `d` — so on the large-hopper grid, where the same
absolute offset is a much smaller fraction of `D`, this recipe's exponents
(biased by the training band's offset curvature) predict the wrong growth
rate. Only hypothesizing the offset `D - k*d` and checking which `k`
re-linearizes `log(Q/rho)` against `log(D - k*d)` recovers a gauge that
generalizes to large-hopper scale.

## Constraints
Time limit 5 s, memory 512 MB. `n` = 200. Scoring is fully deterministic.
