#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE censored-tenure TRAIN sample to stdout.

A subscription business logs customers over a VISIBLE observation window of
length T_obs.  Each customer i belongs to a cohort described by a continuous
loyalty-program covariate x_i in [0,1] (drawn from six weighted buckets).  Each
customer has a TRUE (hidden) tenure T_true_i drawn from a Weibull hazard whose
SHAPE kappa(x) and SCALE lambda(x) both depend on the cohort covariate x --
different cohorts churn at genuinely different RATES and in different SHAPES
(some cohorts have decreasing hazard / long sticky tails, others have rising
hazard and churn in a burst around a typical tenure).

Customers who have not yet churned when the window closes are RIGHT-CENSORED:
we only learn that they were still active at T_obs, not their true tenure.
STDOUT prints ONLY the observable log: a header "N T_obs testId" then N rows
"x  observed_tenure  censored" (censored=1 means "still active at T_obs, true
exit unknown"; censored=0 means the true exit was observed).  The hidden
kappa(x)/lambda(x) law, and every held-out horizon, are NEVER printed here --
they live only inside verify.py.
"""
import sys, math, random

# One row per testId: (N, T_obs, kappa0, kappa1, lambda0, lambda1, bucket_weights[6])
# kappa(x) = kappa0 + kappa1*x        (hazard SHAPE: <1 decreasing/long-tail, >1 rising)
# lambda(x) = lambda0 * exp(lambda1*x) (hazard SCALE)
# buckets are x in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
PARAMS = {
    1:  (400,  40, 1.40,  0.40, 25.0,  0.10, (0.20, 0.20, 0.20, 0.16, 0.14, 0.10)),
    2:  (500,  25, 0.55,  0.15, 30.0,  0.20, (0.30, 0.25, 0.20, 0.15, 0.07, 0.03)),
    3:  (350,  60, 1.10, -0.50, 20.0, -0.15, (0.15, 0.15, 0.15, 0.15, 0.20, 0.20)),
    4:  (800,  20, 0.50,  0.00, 45.0,  0.00, (0.166667,) * 6),
    5:  (300,  22, 1.80, -0.90, 15.0,  0.30, (0.10, 0.10, 0.15, 0.20, 0.20, 0.25)),
    6:  (600,  35, 0.60,  0.60, 22.0, -0.25, (0.05, 0.10, 0.15, 0.20, 0.25, 0.25)),
    7:  (250,  18, 0.45,  0.05, 50.0,  0.05, (0.166667,) * 6),
    8:  (1200, 28, 1.60, -1.00, 18.0,  0.35, (0.30, 0.05, 0.05, 0.05, 0.05, 0.50)),
    9:  (700,  30, 0.70,  0.90, 28.0, -0.40, (0.35, 0.25, 0.15, 0.10, 0.10, 0.05)),
    10: (1500, 22, 0.40,  1.40, 35.0,  0.15, (0.50, 0.02, 0.02, 0.02, 0.02, 0.42)),
}
BUCKETS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def draw_customer(rng, kappa0, kappa1, lam0, lam1, T_obs, weights):
    x = rng.choices(BUCKETS, weights=weights, k=1)[0]
    kappa = kappa0 + kappa1 * x
    lam = lam0 * math.exp(lam1 * x)
    u = max(rng.random(), 1e-12)
    t_true = lam * ((-math.log(u)) ** (1.0 / kappa))
    if t_true >= T_obs:
        return x, T_obs, 1
    return x, t_true, 0


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t not in PARAMS:
        t = ((t - 1) % len(PARAMS)) + 1
    N, T_obs, kappa0, kappa1, lam0, lam1, weights = PARAMS[t]
    rng = random.Random(730021 + t * 104729)

    lines = ["%d %d %d" % (N, T_obs, t)]
    for _ in range(N):
        x, obs, cens = draw_customer(rng, kappa0, kappa1, lam0, lam1, T_obs, weights)
        lines.append("%.1f %.4f %d" % (x, obs, cens))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
