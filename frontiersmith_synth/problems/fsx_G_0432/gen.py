#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training sample to stdout.

Oceanographic tidal-gauge harmonic recovery.  A coastal tide gauge records the
sea-surface height h(t) (metres, relative to a fixed datum) at hourly intervals.
The true water level is governed by a hidden, FOURIER-SPARSE periodic law: a
mean sea level plus a SMALL number of tidal constituents

        h(t) = mu + sum_k  A_k * cos( omega_k * t + phi_k )

The angular frequencies omega_k sit in (but are perturbed OFF) the astronomical
tidal band (semidiurnal ~12 h, diurnal ~24 h); the amplitudes A_k, phases phi_k,
mean level mu, and the exact frequencies are all HIDDEN and differ per gauge.
Every reading carries instrument noise.

The solver only ever sees a SHORT recent window of hourly readings (the training
split emitted here).  The grader regenerates a FUTURE window (later times, an
extrapolation region) inside itself and never prints it.  The task is to recover
a compact closed-form law in the single variable `t` (hours) that predicts the
future tide, not one that merely memorises the noisy window.

Difficulty ladder (testId 1..10): shorter training window, more constituents,
and heavier instrument noise.  STDOUT prints ONLY a header "<n_train> <test_id>"
then n_train rows "t h".  The hidden law and its seed are NEVER printed.
"""
import sys, math, random

TWO_PI = 2.0 * math.pi

# Well-separated tidal-band periods (hours): a shallow-water overtide, a
# semidiurnal, a diurnal-ish, and two longer constituents.  They are spread far
# enough apart in frequency to be resolvable inside even the shortest (60 h)
# observation window -- unlike the tightly clustered real M2/S2 pair, which
# would require a two-week record to separate.
BASE_PERIODS = [6.0, 9.0, 13.0, 19.0, 31.0]


def n_constituents(t):
    return 2 if t <= 3 else (3 if t <= 6 else 4)


def constituents(t):
    """Hidden Fourier-sparse tidal law for test id t.  Deterministic."""
    rng = random.Random(0xC0FFEE + t * 7919)
    K = n_constituents(t)
    mu = rng.uniform(0.5, 2.0)
    chosen = rng.sample(BASE_PERIODS, K)
    cons = []
    for P0 in chosen:
        P = P0 * rng.uniform(0.97, 1.03)          # off-grid perturbation
        omega = TWO_PI / P
        A = rng.uniform(0.3, 1.5)
        phi = rng.uniform(0.0, TWO_PI)
        cons.append((omega, A, phi))
    return mu, cons


def height(tval, mu, cons):
    h = mu
    for omega, A, phi in cons:
        h += A * math.cos(omega * tval + phi)
    return h


def train_window(t):
    """(n_train, L_train) for test id t."""
    L = 168 - (t - 1) * 11          # 168h (7d) down to 60h (2.5d)
    return L + 1, L                 # hourly sampling incl. endpoints


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n, L = train_window(t)
    mu, cons = constituents(t)
    sigma = 0.03 + (t - 1) * 0.012
    rng = random.Random(500 + t * 104729)
    out = ["%d %d" % (n, t)]
    for i in range(n):
        tv = float(i)               # hourly, t = 0 .. L
        h = height(tv, mu, cons) + rng.gauss(0.0, sigma)
        out.append("%r %r" % (tv, h))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
