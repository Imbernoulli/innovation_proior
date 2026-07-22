# The Notch in the Scrolls

## Problem

A conquered province's tax code has been lost, but its payroll scrolls survive. Every
worker of wage rate `w` chose hours `h >= 0` to maximize

```
U(w,h) = w*h - T(w*h) - h^2/2
```

i.e. take-home pay minus tax minus a quadratic disutility of labor. The lost code taxed
pretax income `z = w*h` through exactly one bracket boundary `z0`: below it a flat
marginal rate `tau_lo` applied; at `z0` a one-time surcharge `dT` was assessed (a
"notch" — crossing the line costs you `dT` outright, not just a higher rate on the
margin), and above it a marginal rate `tau_hi` applied. `tau_lo`, `z0` and `tau_hi` are
lost; `dT` survives as a recovered fragment.

The surviving scroll batch records `N` workers whose wage happened to fall in the
archived window `[W_LO, W_HI]` (uniformly distributed there). That window is known to
run exactly up to the wage at which a worker first finds it worthwhile to push into the
upper bracket — so **no** archived worker is observed inside the upper bracket, yet the
full pile-up of workers bunching at the notch boundary IS archived. Reported hours carry
small idiosyncratic noise, and in the harder scrolls a few entries are mis-transcribed.

Your task: reconstruct the tax code `T(z)` well enough that it correctly predicts hours
for a newer, higher-earning cohort whose wages were never recorded — precisely the
regime where the upper bracket binds.

## Input (stdin)

```
testId N W_LO W_HI dT
w_1 h_1
w_2 h_2
...
w_N h_N
```
All values are floats (`testId`, `N` integers). `1 <= N <= 6000`.

## Output (stdout)

One line: a Python-style arithmetic expression for `T(z)` in the single variable `z`.
Allowed: `+ - * /`, unary `-`, parentheses, numeric literals, calls `min(a,b)`,
`max(a,b)`, `abs(a)`, and single comparisons `z<C`, `z<=C`, `z>C`, `z>=C`, `z==C` (usable
as 0/1 indicators, e.g. `50*(z>=800)`). At most 300 characters, at most 60 AST nodes.

*Illustrative FORM only — not the hidden schedule and not from any real test case:*
`0.15*min(z,500)+0.35*max(z-500,0)+50*(z>=500)`

## Feasibility

Your `T` must satisfy `T(0) ~= 0` and must never DECREASE as income rises (checked
densely everywhere your expression can possibly place a jump, not just on a coarse
grid — there is no way to hide a shrinking-liability pocket by making it narrow).
Away from a point where your expression jumps, the implied marginal rate (a
finite-difference slope) must stay below `0.98`; an upward jump exactly at one of
your own literal thresholds (a genuine notch) is exempt from that ceiling — that is
the whole point of the family. Any violation, a parse/syntax error, a disallowed
token, or a non-finite value anywhere probed scores `Ratio: 0.0`.

## Objective (minimize)

The checker regenerates a held-out cohort of higher-earning workers (their wages are
never shown to you) and computes, for the TRUE hidden schedule and for YOUR submitted
`T`, each worker's best-response hours via the same generic best-response solver
(numerical maximization of `U(w,h)` over `h>=0`). Let `F` be the mean squared error
between your predicted hours and the (noisy) true held-out hours, lightly inflated by
expression size (`F *= 1 + 0.0015*nodes`). Let `B` be the mean squared error of the
checker's own naive baseline: a single flat rate fitted from `sum(hours)/sum(wages)`
using only the LOWER HALF of your training data by wage (a lazy read of the archive
that never even glances at the top of the window), applied to the held-out cohort.
The score is

```
Ratio = min(1000, 100*B/F) / 1000
```

A flat-rate guess reproduces the baseline (`~0.1`). Fitting hours-vs-wage directly, no
matter how well it fits the archive, carries zero information about the never-observed
upper bracket. Idiosyncratic labor-supply noise in the held-out cohort keeps even a
correctly reconstructed schedule below the ceiling.

## Constraints

`500 <= N <= 6000`, `1 <= testId <= 10`, time limit 5s, memory 512MB.

## Example (worked score, illustrative only)

Suppose the true held-out MSE of your submission is `F_mse = 6.5`, size penalty raises
it to `F = 6.53`, and the baseline's held-out MSE is `B = 40.0`. Then
`Ratio = min(1000, 100*40/6.53)/1000 = min(1000, 612.6)/1000 = 0.613` — a solid but
not perfect reconstruction (numbers illustrative only; irreducible noise in the
held-out cohort means even the best achievable schedule leaves some residual error).
