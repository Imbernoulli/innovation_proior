# Onset-Matched Alpha Spending

## Problem

You are designing the interim-monitoring plan for a clinical trial that can
enroll up to `N_max` patients. Patients are recruited one at a time, indexed
`j = 1..N_max`. You must choose **up to `K_max` interim looks**: cumulative
enrollment sizes `1 <= n_1 < n_2 < ... < n_K <= N_max` (`K <= K_max`), and at
each look an **efficacy boundary** `z_eff_i` and a **futility boundary**
`z_fut_i` (`z_fut_i <= z_eff_i`).

Patient `j` contributes an independent standardized signal `s_j ~ N(mu_j, 1)`.
Under the null hypothesis `mu_j = 0` for all `j`. Under the alternative the
treatment effect **ramps up**: `mu_j = delta * min(1, j / (onset_frac *
N_max))` (immediately `= delta` if `onset_frac = 0`) -- i.e. the true effect
size is only fully present once `onset_frac` of enrollment has passed
(a delayed/emerging treatment effect). At look `i` the test statistic is
`Z_i = (sum_{j<=n_i} s_j) / sqrt(n_i)`.

**Stopping rule.** At look `i < K`: if `Z_i >= z_eff_i` stop and *reject*
(declare efficacy, using `n_i` patients); else if `Z_i <= z_fut_i` stop and
declare *futility* (also `n_i` patients); else continue. At the final look
`K`: reject if `Z_K >= z_eff_K`, otherwise fail to reject -- `n_K` patients
are used either way (`z_fut_K` is only checked for the ordering constraint
above, it has no other effect).

## Feasibility (hard gates -- any violation scores 0)

The checker draws `M=4000` independent length-`N_max` cohorts under the null
and `M=4000` under the alternative (seeded deterministically by `rng_seed`
from the instance -- reproduce the same recipe yourself if you want to
calibrate against it: `numpy.random.default_rng(rng_seed).standard_normal((M,
N_max))` for the null cohorts). Your plan must satisfy, on these ensembles:

1. **Family-wise type-I error**: `P(reject | null) <= alpha_total` (plus a
   small finite-sample margin, roughly `alpha_total + 0.01`). Testing
   repeatedly at the unadjusted nominal threshold at every look blows this
   cap by a wide margin -- it must be spent deliberately across looks.
2. **Power**: `P(reject | alternative) >= power_floor`.
3. Structural validity: `n_i` strictly increasing integers in `[1, N_max]`,
   `K` in `[1, K_max]`, all thresholds finite and in `[-50, 50]`, and
   **`n_K = N_max` exactly** -- the final scheduled look must enroll the
   full `N_max` (you cannot pre-commit to a smaller trial ceiling; the only
   way to use fewer patients is to genuinely stop *early*, at an interim
   look, by actually crossing one of its boundaries).

## Objective (maximize)

The expected **enrollment-cost saved** versus always enrolling everyone:

```
F = cost_per_patient * (N_max - E[n_used])
E[n_used] = 0.5 * E_alternative[n_used] + 0.5 * E_null[n_used]
```

(`n_used` is however many patients were enrolled before your plan stopped, in
each simulated cohort; the 0.5/0.5 weighting is a fixed prior that the
treatment is equally likely to work or not.) There is no free lunch: stopping
earlier only saves cost if the boundaries you chose still clear the power
floor above -- and where the treatment effect only shows up late, boundaries
that spend the budget uniformly across all `K_max` looks waste it on
uninformative early looks and lose power, forcing you toward the (feasible
but cheap) fixed-sample plan or below the floor entirely.

## Input (stdin)

```
N_max K_max
alpha_total power_floor
cost_per_patient delta onset_frac
rng_seed
```

## Output (stdout)

```
K
n_1 z_eff_1 z_fut_1
...
n_K z_eff_K z_fut_K
```

## Scoring

The checker also builds its own simple reference plan `B`: one interim look
at `N_max/2` spending 5% of `alpha_total` and a final look at `N_max`
spending the remaining 95% (safe by the Bonferroni union bound, no
calibration needed) -- a non-adaptive plan that ignores the onset timing.
Your feasible score is `Ratio = min(1, 0.1 * F / B)`, so matching this
reference scores `0.1`; a plan that saves `10x` more expected enrollment
cost caps at `1.0`. (Illustrative only: this reference is deliberately weak.)
