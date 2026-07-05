# TIER: strong
# Simulated-annealing shape search over the coolant profile, seeded from the
# flat schedule (c1 = 2). Perturbs individual slots to drive the autocorrelation
# peak below the flat value; keeps the best profile seen (never worse than flat).
# Fully deterministic (fixed seed). Emits an integer-scaled profile.
import sys
import math
import random

import numpy as np


def cf(f, n):
    s = f.sum()
    if s <= 0.0:
        return 1e9
    peak = np.convolve(f, f).max()
    return 2.0 * n * peak / (s * s)


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    M = int(toks[1])

    rnd = random.Random(20260701 + n)
    f = np.ones(n, dtype=np.float64)
    cur = cf(f, n)
    best = cur
    bestf = f.copy()

    iters = 12000
    for it in range(iters):
        T = 0.12 * (1.0 - it / iters) + 0.004
        i = rnd.randrange(n)
        old = f[i]
        cand = old * rnd.uniform(0.5, 1.7) + rnd.uniform(-0.10, 0.10)
        if cand < 0.0:
            cand = 0.0
        f[i] = cand
        v = cf(f, n)
        if v < cur or rnd.random() < math.exp(-min(50.0, (v - cur) / T)):
            cur = v
            if v < best:
                best = v
                bestf = f.copy()
        else:
            f[i] = old

    mx = bestf.max()
    if mx <= 0.0:
        out = [1] * n
    else:
        scale = 1000.0 / mx
        out = [int(round(float(x) * scale)) for x in bestf]
        if sum(out) <= 0:
            out = [1] * n
        out = [min(M, max(0, v)) for v in out]

    sys.stdout.write(" ".join(map(str, out)) + "\n")


if __name__ == "__main__":
    main()
