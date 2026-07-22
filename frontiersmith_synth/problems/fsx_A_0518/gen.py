#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN trace to stdout.

Reverse-engineer an antique greenhouse thermostat.  A hidden latch drives the
heater with (a) a hysteresis band [L,H] -- the heater only switches when the
drive temperature leaves the band, so INSIDE the band the state is path
dependent -- and (b) a fixed k-step actuation delay before the heater responds,
with a little fractional-lag blur.  Each testId fixes a DIFFERENT hidden
thermostat.

The solver only ever SEES this TRAIN trace, which is recorded under a SLOW,
monotone-ish drive (the greenhouse warming and cooling gently over a day).  In
that quasi-static regime the actuation delay is nearly invisible and the two
hysteresis branches look almost like a single static curve.  The held-out
grading trace lives in a FAST, non-monotone regime (a storm rattling the vents)
and is regenerated only inside the grader -- it is never printed here.

STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train rows
"<drive> <heater>".  The hidden band, delay, and seed are NOT printed.
"""
import sys, random, math

OFF, AMP = 0.2, 0.6


def params(t):
    """Hidden thermostat for this test id (lives in gen AND grader, never printed)."""
    rng = random.Random(4200191 + t * 7919)
    L = rng.uniform(0.34, 0.42)
    w = rng.uniform(0.14, 0.22)
    H = L + w
    k = rng.choice([2, 3])
    phi = rng.uniform(0.30, 0.55)
    return L, H, k, phi


def latch_roll(drive, L, H):
    """Schmitt latch: heater ON (1) below L, OFF (0) above H, HOLD inside [L,H]."""
    s = 0
    out = []
    for d in drive:
        if d < L:
            s = 1
        elif d > H:
            s = 0
        out.append(s)
    return out


def true_output(drive, L, H, k, phi, sigma, seed):
    """Heater trace = latch delayed by a fractional (k+phi) steps, plus sensor noise."""
    S = latch_roll(drive, L, H)
    rng = random.Random(seed)
    y = []
    for t in range(len(drive)):
        a = S[t - k] if t - k >= 0 else 0
        b = S[t - k - 1] if t - k - 1 >= 0 else 0
        y.append(OFF + AMP * ((1 - phi) * a + phi * b) + rng.gauss(0.0, sigma))
    return y


def slow_drive(t, n):
    """SLOW monotone-ish training drive (gentle daily warm/cool), spans the range."""
    rng = random.Random(900 + t * 104729)
    per1 = rng.uniform(80, 120)
    per2 = rng.uniform(170, 280)
    ph1 = rng.uniform(0, 6.283185)
    ph2 = rng.uniform(0, 6.283185)
    d = []
    for i in range(n):
        v = 0.5 + 0.36 * math.sin(2 * math.pi * i / per1 + ph1) \
                + 0.11 * math.sin(2 * math.pi * i / per2 + ph2)
        d.append(min(1.0, max(0.0, v)))
    return d


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sigma = 0.05 + 0.004 * (t - 1)
    n = 340 - 8 * (t - 1)
    L, H, k, phi = params(t)
    drive = slow_drive(t, n)
    y = true_output(drive, L, H, k, phi, sigma, 555 + t * 13)
    out = ["%d %d" % (n, t)]
    for i in range(n):
        out.append("%.6f %.6f" % (drive[i], y[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
