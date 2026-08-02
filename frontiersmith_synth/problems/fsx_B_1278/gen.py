#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE instance of the adaptive-trial monitoring-plan
problem to stdout.

Instance tokens (whitespace separated, any layout):
    N_max K_max alpha_total power_floor cost_per_patient delta onset_frac rng_seed

  N_max          maximum total patients that may be enrolled
  K_max          maximum number of interim analyses (looks) the plan may use
  alpha_total    required family-wise one-sided type-I error cap
  power_floor    required minimum power under the alternative (hard constraint)
  cost_per_patient  cost charged per enrolled patient
  delta          alternative-hypothesis per-patient standardized effect size
  onset_frac     fraction of N_max over which the treatment effect ramps up
                 linearly from 0 to `delta` (0 => effect present from patient 1)
  rng_seed       seed for the checker's (and the solver's own, if it chooses to
                 replicate it) deterministic Monte-Carlo feasibility ensemble

Difficulty ladder (testId 1..10): grows in N_max/K_max; testId 5,7,8,9,10 are
"delayed-onset" instances (onset_frac >= 0.3, several >= 0.8) engineered so a
monitoring plan that is blind to the onset timing loses power below the floor.
`delta` is precomputed offline (per testId) so that an ONSET-AWARE plan can
reach ~85% power -- the values are fixed constants of the ladder, not derived
at runtime, so the instance stream is trivially reproducible.
"""
import sys

# (N_max, K_max, onset_frac, delta, cost_per_patient)
_CASES = [
    (300,  3, 0.00, 0.178830, 1.0),
    (500,  4, 0.10, 0.145560, 1.2),
    (700,  4, 0.20, 0.126920, 0.8),
    (900,  5, 0.00, 0.107110, 1.5),
    (1100, 5, 0.30, 0.105790, 1.0),
    (1300, 6, 0.00, 0.088350, 2.0),
    (900,  5, 0.80, 0.168370, 1.0),
    (1100, 5, 0.85, 0.161180, 1.3),
    (1300, 6, 0.85, 0.149060, 1.0),
    (1800, 3, 0.80, 0.115000, 1.1),
]

ALPHA_TOTAL = 0.025
POWER_FLOOR = 0.75


def main():
    t = int(sys.argv[1])
    n_max, k_max, onset_frac, delta, cost = _CASES[t - 1]
    rng_seed = t
    print(n_max, k_max)
    print(ALPHA_TOTAL, POWER_FLOOR)
    print(cost, delta, onset_frac)
    print(rng_seed)


if __name__ == "__main__":
    main()
