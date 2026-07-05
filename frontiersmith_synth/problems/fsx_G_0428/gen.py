#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training sample to stdout.

Integer-sequence oracle (recurrence discovery).  A hidden combinatorial
integer sequence a(0), a(1), ... is produced by a fixed but UNKNOWN law:
a clean integer linear-recurrence "signal" L(n) (n-nacci-style exponential
growth, or a figurate polynomial) corrupted by a bounded multiplicative
sampling jitter, as if each count were estimated by a noisy stochastic
sampler.  The solver only sees the FIRST T noisy terms of the sequence.

The grading split lives in a FAR-FUTURE window (indices well beyond the
training prefix) with FRESH, independent jitter, and is regenerated inside the
grader only -- it is never printed here.

Difficulty ladder (testId 1..10): higher recurrence order + heavier jitter +
fewer observed terms.

STDOUT prints ONLY: a header "<T> <test_id>" then T rows "<n> <a(n)>".
The hidden law, its recurrence coefficients, and its seed are NOT printed.
"""
import sys, random

# ---- ladder configuration (shared, identical in verify.py) ----
TLIST  = [24, 24, 22, 22, 20, 20, 18, 18, 16, 16]     # observed prefix length
DELTAS = [0.26, 0.20, 0.28, 0.28, 0.22, 0.30, 0.30, 0.24, 0.32, 0.34]  # jitter half-width
GAP    = 6      # blind gap between train prefix and held-out window
NHELD  = 14     # number of far-future held-out terms

LAW_BASE      = 770001
PRIME         = 100003
TRAIN_NOISE   = 880002
HELDOUT_NOISE = 990003

# (kind, order K, recurrence coefficients c_1..c_K with a(n)=sum c_i a(n-i))
TEMPLATES = [
    ("fib",   2, [1, 1]),
    ("poly2", 3, [3, -3, 1]),
    ("fib",   2, [1, 1]),
    ("trib",  3, [1, 1, 1]),
    ("poly3", 4, [4, -6, 4, -1]),
    ("trib",  3, [1, 1, 1]),
    ("tetra", 4, [1, 1, 1, 1]),
    ("poly3", 4, [4, -6, 4, -1]),
    ("penta", 5, [1, 1, 1, 1, 1]),
    ("penta", 5, [1, 1, 1, 1, 1]),
]


def make_seeds(kind, K, rng):
    if kind == "poly2":                       # figurate: exact quadratic
        A = rng.randint(1, 3); B = rng.randint(0, 3); C = rng.randint(1, 4)
        return [A * k * k + B * k + C for k in range(K)]
    if kind == "poly3":                       # exact cubic
        A = rng.randint(1, 2); B = rng.randint(0, 2)
        C = rng.randint(0, 3); D = rng.randint(1, 4)
        return [A * k ** 3 + B * k * k + C * k + D for k in range(K)]
    # n-nacci style: small positive integer seeds
    return [rng.randint(1, 5) for _ in range(K)]


def clean_signal(t, maxidx):
    """The exact integer linear-recurrence signal L(0..maxidx)."""
    kind, K, coeffs = TEMPLATES[(t - 1) % len(TEMPLATES)]
    rng = random.Random(LAW_BASE + t * PRIME)
    seeds = make_seeds(kind, K, rng)
    L = list(seeds)
    for n in range(K, maxidx + 1):
        L.append(sum(coeffs[i] * L[n - 1 - i] for i in range(K)))
    return L


def build_true(t):
    """Full noisy sequence true[0..maxidx] with region-dependent jitter."""
    T = TLIST[(t - 1) % len(TLIST)]
    delta = DELTAS[(t - 1) % len(DELTAS)]
    maxidx = T + GAP + NHELD - 1
    L = clean_signal(t, maxidx)
    tr_rng = random.Random(TRAIN_NOISE + t * PRIME)
    ho_rng = random.Random(HELDOUT_NOISE + t * PRIME)
    true = []
    for n in range(maxidx + 1):
        rng = tr_rng if n < T else ho_rng
        j = rng.uniform(-delta, delta)
        true.append(int(round(L[n] * (1.0 + j))))
    held = [T + GAP + i for i in range(NHELD)]
    return T, true, held


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    T, true, _held = build_true(t)
    out = ["%d %d" % (T, t)]
    for n in range(T):
        out.append("%d %d" % (n, true[n]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
