#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy multi-day TRAIN trace of GNSS multipath
position error to stdout.

Physical picture (see statement.md): the receiver's excess position error is
the SUM of

  (a) an "orbital-repeat multipath" component: the constellation ground track
      -- and hence which satellites are up and at what elevation/azimuth --
      repeats with a period P1 close to, but NOT exactly, one solar day
      (86400 s).  Because the reflection geometry sweeps through several
      distinct fade/boost configurations each repeat cycle, this component is
      the FUNDAMENTAL + 2nd + 3rd harmonic of P1 (not a single sinusoid);

  (b) a smaller "solar/thermal" component at EXACTLY one solar day (86400 s,
      a public physical constant, not hidden);

  (c) i.i.d. sensor noise.

Each test id fixes a DIFFERENT hidden P1, amplitudes and phases.  The solver
only ever sees a TRAIN window of a few CONSECUTIVE days starting at t=0.  The
held-out grading horizon lives weeks-to-months later and is regenerated only
inside verify.py -- it is never printed here, and neither is P1, any
amplitude/phase, or the noise seed.

STDOUT: header "<n_train> <test_id>", then n_train rows "<t_seconds> <error>".
"""
import sys, random, math

DT = 300.0              # 5-minute sampling
SOLAR_PERIOD = 86400.0  # exact solar day -- PUBLIC constant, also known to solvers


def hidden_params(t):
    """Hidden per-instance GNSS multipath law (lives in gen AND verify, never printed)."""
    rng = random.Random(910177 + t * 104729)
    gap = rng.uniform(120.0, 420.0)          # true repeat period is `gap` s SHORTER than 24h
    P1 = SOLAR_PERIOD - gap
    A0 = rng.uniform(0.12, 0.42)
    phi0 = rng.uniform(0.0, 2 * math.pi)
    harmonics = []
    for lo, hi in [(0.55, 1.05), (0.18, 0.45), (0.05, 0.18)]:
        A = rng.uniform(lo, hi)
        phi = rng.uniform(0.0, 2 * math.pi)
        harmonics.append((A, phi))
    return P1, (A0, phi0), harmonics


def true_signal(tsec, P1, solar, harmonics):
    A0, phi0 = solar
    w0 = 2 * math.pi / SOLAR_PERIOD
    val = A0 * math.cos(w0 * tsec + phi0)
    w1 = 2 * math.pi / P1
    for i, (A, phi) in enumerate(harmonics, start=1):
        val += A * math.cos(i * w1 * tsec + phi)
    return val


def sensor_noise_sigma(t, solar, harmonics):
    """Noise floor is a per-instance FRACTION `k` of the signal's own RMS
    amplitude (a realistic SNR-style sensor model) -- this keeps the noise
    floor comparable to the signal on every instance, regardless of how big
    or small that instance's hidden amplitudes happen to be, and the `k`
    schedule below is what actually makes later test ids harder."""
    A0, _ = solar
    b = 0.5 * (A0 * A0 + sum(A * A for A, _ in harmonics))
    k_tbl = [0.25, 0.28, 0.30, 0.33, 0.36, 0.40, 0.44, 0.48, 0.55, 0.62]
    k = k_tbl[max(1, min(10, t)) - 1]
    return k * math.sqrt(b)


def schedule(t):
    """Difficulty ladder: fewer visible days as testId grows (noise handled
    separately by sensor_noise_sigma, scaled to each instance's amplitude)."""
    days_tbl = [7, 6, 6, 5, 5, 4, 4, 4, 3, 3]
    i = max(1, min(10, t)) - 1
    return days_tbl[i]


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    days = schedule(t)
    n = int(days * 86400.0 / DT)
    P1, solar, harmonics = hidden_params(t)
    sigma = sensor_noise_sigma(t, solar, harmonics)
    rng = random.Random(31337 + t * 92821)
    rows = []
    for k in range(n):
        tsec = k * DT
        val = true_signal(tsec, P1, solar, harmonics) + rng.gauss(0.0, sigma)
        rows.append("%.3f %.6f" % (tsec, val))
    out = ["%d %d" % (n, t)]
    out.extend(rows)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
