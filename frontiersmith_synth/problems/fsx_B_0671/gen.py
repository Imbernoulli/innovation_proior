#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

A shop fan is filmed with a fixed-rate strobe camera while its drive level
`x` is swept.  The blade-pass frequency obeys a hidden, per-test MONOTONE
power law:

    f_true(x) = a * x^b + c        (a > 0, b > 1, c >= 0, all hidden)

But the camera only samples at `fs` frames per second, well below the
blade-pass frequency for most of the sweep.  Every reading the camera
produces is therefore an ALIAS: it reports the true frequency folded into
the baseband window [0, fs/2] by the usual triangle-wave Nyquist fold

    fold(f, fs):  m = f mod fs;  return m if m <= fs/2 else fs - m

Each testId fixes a different hidden (a, b, c) AND a different (generally
low) camera rate `fs`, so the number of folds crossed varies per row -- the
solver only ever sees the folded, noisy readings.  The held-out grading grid
(reaching well past the swept drive-level range) is regenerated only inside
the checker; the seed, the law, and fs's role in producing it are never
printed here.

STDOUT prints ONLY: a header "<testId> <N> <fs> <F_MAX>" then N rows
"x_i  reading_i", one strobe frame per configuration, in SHUFFLED order.
"""
import sys, random, math

X_LO = 1.0
F_MAX = 900.0              # loose, test-independent physical ceiling (given to solver)
NOISE_SIGMA = 0.015        # small multiplicative noise on the true frequency pre-fold


def hidden_law(t):
    """Hidden frequency-parameter law for this test id. Lives in gen AND checker; never printed."""
    rng = random.Random(671000 + t * 7919)
    a = rng.uniform(0.8, 2.2)
    b = rng.uniform(1.25, 1.95)
    c = rng.uniform(1.0, 4.0)
    return a, b, c


def f_true(x, a, b, c):
    return a * (x ** b) + c


def x_range(t):
    return X_LO, 10.0 + 0.4 * t


_TARGET_ZMAX = {1: 0.45, 2: 0.8, 3: 1.3, 4: 2.2, 5: 3.5,
                6: 5.5, 7: 8.5, 8: 13.0, 9: 19.0, 10: 27.0}


def sampling_rate(t):
    """Camera rate chosen so the MAXIMUM Nyquist zone reached across the
    training sweep hits a per-testId difficulty target (roughly alias-free
    at low t, heavily folded at high t) regardless of the randomly drawn
    law's own scale."""
    a, b, c = hidden_law(t)
    _, X_HI = x_range(t)
    fmax_train = f_true(X_HI, a, b, c)
    target = _TARGET_ZMAX.get(t, 0.45 * (1.55 ** (t - 1)))
    rng = random.Random(671500 + t * 104729)
    jitter = rng.uniform(0.9, 1.1)
    half = fmax_train / (target * jitter)
    return 2.0 * half


def n_train(t):
    return 24 + 2 * t


def fold(freq, fs):
    half = fs / 2.0
    m = math.fmod(freq, fs)
    if m < 0.0:
        m += fs
    if m <= half:
        return m
    return fs - m


def train_rows(t):
    a, b, c = hidden_law(t)
    X_LO_, X_HI = x_range(t)
    fs = sampling_rate(t)
    half = fs / 2.0
    N = n_train(t)
    rng = random.Random(671800 + t * 13)
    log_lo, log_hi = math.log(X_LO_), math.log(X_HI)

    # Difficulty ladder: harder tests (t>=4) carve a deliberate GAP out of the
    # middle of the swept drive-level range.  Because the true frequency is
    # monotone in x, the fold-zone index is also non-decreasing in x -- but a
    # wide gap forces it to jump by several zones between the last point
    # before the gap and the first point after it, defeating any purely
    # LOCAL nearest-neighbour un-wrap and any regression that ignores the
    # fold structure entirely.
    has_gap = t >= 4
    xs = []
    if has_gap:
        gap_lo, gap_hi = 0.42, 0.68
        n1 = int(round(N * 0.55))
        n2 = N - n1
        for i in range(n1):
            frac = (i + rng.uniform(0.05, 0.95)) / n1 * gap_lo
            xs.append(frac)
        for j in range(n2):
            frac = gap_hi + (j + rng.uniform(0.05, 0.95)) / n2 * (1.0 - gap_hi)
            xs.append(frac)
    else:
        for i in range(N):
            frac = (i + rng.uniform(0.05, 0.95)) / N
            xs.append(frac)

    rows = []
    for frac in xs:
        frac = min(0.999999, max(0.000001, frac))
        x = math.exp(log_lo + frac * (log_hi - log_lo))
        ftrue = f_true(x, a, b, c)
        f_noisy_true = ftrue * math.exp(rng.gauss(0.0, NOISE_SIGMA))
        r = fold(f_noisy_true, fs)
        r = min(half, max(0.0, r))
        rows.append((x, r))

    rng.shuffle(rows)
    return rows, fs


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows, fs = train_rows(t)
    out = ["%d %d %.10g %.10g" % (t, len(rows), fs, F_MAX)]
    for x, r in rows:
        out.append("%.8f %.8f" % (x, r))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
