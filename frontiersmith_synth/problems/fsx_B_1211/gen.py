#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN table to stdout.

Ice-shelf calving forecast.  A shelf segment has thickness H(t) = H0 - gamma*t
(slow linear thinning) and a rift whose depth D(t) grows because of ongoing
stress accumulation from the local strain rate:

  phase 1 (intact buttressing):   dD/dt = c0                    for D/H < phi
  phase 2 (buttressing lost):     dD/dt = c0 * (1 + BETA)        for D/H >= phi

i.e. once the crevasse-to-thickness RATIO D(t)/H(t) reaches the segment's
feedback-onset ratio `phi`, the neighbouring shelf can no longer brace the
rift, back-stress is lost, and the rift deepens BETA+1 times faster for the
rest of its life.  BETA is a fixed structural gain, identical for every
segment on every shelf.  Calving happens the instant D(t)/H(t) reaches a
CRITICAL ratio `kappa`.  kappa is a property of THIS particular shelf (fixed
across every segment / every held-out configuration in this test id) and is
NEVER printed -- it must be inferred from the training rows.

Each testId fixes a DIFFERENT hidden shelf (different kappa, different RNG).
The TRAIN table logged here comes from segments whose feedback ratio `phi`
sits just BELOW kappa: buttressing loss only ever engages in the closing
moments before calving (if at all), so the observed calving horizon is long
and almost the whole trajectory looks like plain phase-1 thinning.  The
HELD-OUT table (regenerated only inside verify.py) comes from segments whose
`phi` sits a large, fixed margin above their OWN starting ratio: buttressing
loss engages well before calving, so a substantial share of the true calving
horizon runs at the accelerated phase-2 rate -- a regime training segments
essentially never visit.  A model that only fit the phase-1 rate has never
seen that regime.

STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train rows
"H0 D0 c0 gamma phi T". kappa, BETA, and the RNG seed are NOT printed.
"""
import sys, random

BETA = 3.0
N_TRAIN = 60
SIGMA_TRAIN = 0.05


def shelf_kappa(t):
    """Hidden per-shelf critical ratio (lives in gen AND verify, never printed)."""
    rng = random.Random(500000 + t * 9176111)
    return rng.uniform(0.75, 0.92)


def true_calve_time(H0, D0, c0, gamma, phi, kappa, beta=BETA):
    """Exact two-phase closed form for the calving time."""
    t1 = (phi * H0 - D0) / (c0 + phi * gamma)
    D1 = D0 + c0 * t1
    H1 = H0 - gamma * t1
    tprime = (kappa * H1 - D1) / (c0 * (1.0 + beta) + kappa * gamma)
    return t1 + tprime


def gen_rows(t, n, rng, kappa, train, sigma):
    rows = []
    for _ in range(n):
        H0 = rng.uniform(150.0, 500.0)
        gamma = rng.uniform(2.0, 6.0)
        c0 = rng.uniform(3.0, 10.0)
        r0 = rng.uniform(0.05, 0.25)
        D0 = r0 * H0
        if train:
            phi = kappa - rng.uniform(0.01, 0.04)
        else:
            phi = r0 + rng.uniform(0.30, 0.45)
        Ttrue = true_calve_time(H0, D0, c0, gamma, phi, kappa)
        Tobs = Ttrue * (2.718281828459045 ** rng.gauss(0.0, sigma))
        rows.append((H0, D0, c0, gamma, phi, Tobs))
    return rows


def gen_train(t):
    kappa = shelf_kappa(t)
    rng = random.Random(111 + t * 13)
    return gen_rows(t, N_TRAIN, rng, kappa, train=True, sigma=SIGMA_TRAIN)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = gen_train(t)
    out = ["%d %d" % (len(rows), t)]
    for H0, D0, c0, gamma, phi, T in rows:
        out.append("%.6f %.6f %.6f %.6f %.6f %.6f" % (H0, D0, c0, gamma, phi, T))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
