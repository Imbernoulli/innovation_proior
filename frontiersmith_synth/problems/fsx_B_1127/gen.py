#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN trace to stdout.

Silly-putty rheology: a linear viscoelastic material has an unknown memory
kernel G(u) (the "relaxation modulus" at lag u).  Under the Boltzmann
superposition principle, the stress at time t from ANY strain history is a
convolution of G against the strain-rate history:

    sigma(t) = integral_0^t  G(t - s) * dgamma/ds(s)  ds

Each testId fixes a DIFFERENT hidden kernel (never printed).  The solver only
sees STEP-strain relaxation experiments: a strain step of magnitude gamma0 is
applied at time 0 to a quiescent sample, and the resulting stress
sigma(t) = gamma0 * G(t) is logged at several lags spanning about two decades
of time.  A few step magnitudes are given (superposition means they should
all reveal the SAME underlying G once you divide out gamma0).

STDOUT prints ONLY a header "<n_rows> <test_id>" then n_rows rows
"<gamma0> <t> <sigma>".  The hidden kernel parameters and RNG seed are never
printed -- they live only inside gen.py/verify.py.
"""
import sys
import random
import math


def truth(t):
    """Hidden generative parameters for this test id (duplicated verbatim in
    verify.py; never printed to the solver)."""
    rng = random.Random(20260726 + 97 * t)
    alpha_true = round(rng.uniform(0.15, 0.85), 4)
    A_true = round(rng.uniform(0.6, 4.0), 4)
    window_choices = [(0.1, 10.0), (1.0, 100.0), (0.05, 5.0), (2.0, 200.0), (0.5, 50.0)]
    t_min, t_max = window_choices[t % len(window_choices)]
    noise_amp = 0.01 + 0.0025 * (t - 1)
    # small deterministic "discrete-scale" correction so the kernel is not an
    # EXACT pure power law -- keeps even the best power-law fit from reaching
    # zero held-out error (headroom), while a wrong functional family (sums
    # of exponentials) is hurt far more by it.
    delta = round(rng.uniform(0.125, 0.165), 4)
    plog = round(rng.uniform(0.9, 1.6), 4)
    phase = round(rng.uniform(0.0, 2 * math.pi), 4)
    return alpha_true, A_true, t_min, t_max, noise_amp, delta, plog, phase


def kernel_true(u, alpha, A, delta, plog, phase):
    """G(u) = A * u^-alpha * (1 + delta * sin(wlog * ln(u) + phase)); u > 0."""
    wlog = 2.0 * math.pi / (plog * math.log(10.0))
    base = A * (u ** (-alpha))
    corr = 1.0 + delta * math.sin(wlog * math.log(u) + phase)
    return base * corr


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    alpha_true, A_true, t_min, t_max, noise_amp, delta, plog, phase = truth(t)

    rng = random.Random(31337 + 911 * t)
    n_times = 12
    times = [t_min * (t_max / t_min) ** (i / (n_times - 1)) for i in range(n_times)]
    magnitudes = [0.5, 1.0, 1.5]

    rows = []
    for g0 in magnitudes:
        for tt in times:
            sig_true = g0 * kernel_true(tt, alpha_true, A_true, delta, plog, phase)
            noise = rng.uniform(-noise_amp, noise_amp)
            sig_obs = sig_true * (1.0 + noise)
            rows.append((g0, tt, sig_obs))

    out = ["%d %d" % (len(rows), t)]
    for g0, tt, sig in rows:
        out.append("%.6f %.8f %.8f" % (g0, tt, sig))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
