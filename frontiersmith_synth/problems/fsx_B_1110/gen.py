#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE calm-season TRAIN log to stdout.

Storm-season harbour congestion.  A hidden berth-priority queueing law drives
per-class mean waits: classes hold fixed (hidden) priority ranks; a class's
wait carries the capacity pole 1/((1 - P_c)(1 - X_c)) where P_c is the total
load of classes queued AHEAD of it and X_c = P_c + r_c its own cumulative
load.  The shared numerator grows with load and mix concentration.

This generator prints ONLY the calm-season regime (rho in [0.08, ~0.55]) with
small multiplicative observation noise.  The storm-season held-out rows
(rho in [0.85, 0.97]) are regenerated ONLY inside the grader.  STDOUT prints
a header "<n_train> <K> <test_id>" then data rows; the hidden constants and
the priority permutation are NEVER printed.
"""
import sys, random


def params(t):
    """Hidden harbour constants for this test id (duplicated in the grader)."""
    rng = random.Random(917331 + t * 7919)
    K = 3 + (t - 1) % 3                      # 3,4,5,3,4,5,...
    a0 = rng.uniform(0.10, 0.20)
    a1 = rng.uniform(0.90, 1.50)
    a2 = rng.uniform(0.50, 0.90)
    perm = list(range(K))                    # perm[rank] = class index (0-based)
    rng.shuffle(perm)
    skew = 1 if t >= 6 else 0                # later tests: skewed mixes
    return K, a0, a1, a2, perm, skew


def draw_mix(rng, K, skew, storm=False):
    lo = 0.15 if skew else 0.35
    pw = 1.7 if storm else 1.0            # storm season: sparser, concentrated mixes
    u = [rng.uniform(lo, 1.0) ** pw for _ in range(K)]
    s = sum(u)
    return [x / s for x in u]


def true_waits(rho, mix, K, a0, a1, a2, perm):
    r = [rho * m for m in mix]
    h = sum(m * m for m in mix)
    P = [0.0] * K
    cum = 0.0
    for rank in range(K):
        c = perm[rank]
        P[c] = cum
        cum += r[c]
    N = a0 * rho * (1.0 + a1 * h)
    w = []
    for c in range(K):
        X = P[c] + r[c]
        w.append(N * (1.0 + a2 * r[c]) / ((1.0 - P[c]) * (1.0 - X)))
    return w


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    K, a0, a1, a2, perm, skew = params(t)
    n_train = 90 - 3 * t
    rho_hi = 0.55 - 0.012 * (t - 1)          # ladder: less high-load cover later
    sig = 0.015 + 0.004 * t
    rng = random.Random(555121 + t * 104729)
    lines = ["%d %d %d" % (n_train, K, t)]
    for _ in range(n_train):
        rho = rng.uniform(0.08, rho_hi)
        mix = draw_mix(rng, K, skew)
        w = true_waits(rho, mix, K, a0, a1, a2, perm)
        row = [rho] + mix + [max(1e-4, wc * (1.0 + rng.gauss(0.0, sig))) for wc in w]
        lines.append(" ".join("%.6f" % x for x in row))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
