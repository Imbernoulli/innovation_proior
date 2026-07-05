# TIER: strong
"""Multi-restart adaptive local search.  Several seeded random restarts, each running
a hill-climb that repeatedly relocates a vertex of the current smallest-area triangle
with an annealed move radius, keeping only non-worsening moves.  Returns the best
layout across restarts.  Clearly beats greedy while staying well below saturation.
Deterministic (seeded by n)."""
import sys
import itertools
import random
import numpy as np


def triples(n):
    return np.array(list(itertools.combinations(range(n), 3)), dtype=np.int64)


def areas(P, T):
    a = P[T[:, 0]]; b = P[T[:, 1]]; c = P[T[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def hillclimb(n, T, rng, iters):
    P = np.array([[rng.random(), rng.random()] for _ in range(n)])
    ar = areas(P, T)
    step = 0.4
    for t in range(iters):
        i, j, k = T[int(np.argmin(ar))]
        idx = (i, j, k)[rng.randrange(3)]
        old = P[idx].copy()
        m = ar.min()
        P[idx, 0] = min(1.0, max(0.0, old[0] + rng.uniform(-step, step)))
        P[idx, 1] = min(1.0, max(0.0, old[1] + rng.uniform(-step, step)))
        nar = areas(P, T)
        if nar.min() >= m:
            ar = nar
        else:
            P[idx] = old
        if t % 300 == 299:
            step *= 0.82
    return ar.min(), P


def main():
    n = int(sys.stdin.read().split()[0])
    T = triples(n)
    rng = random.Random(9999 + 31 * n)
    # scale iterations down as n grows so the full run stays inside the time limit
    iters = max(400, int(30000 / n))
    restarts = 3
    best_v = -1.0
    best_P = None
    for r in range(restarts):
        v, P = hillclimb(n, T, rng, iters)
        if v > best_v:
            best_v = v
            best_P = P.copy()
    out = ["%.10f %.10f" % (best_P[i, 0], best_P[i, 1]) for i in range(n)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
