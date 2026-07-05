# TIER: greedy
"""Single-restart randomized local search: from a seeded random layout, repeatedly
find the smallest-area triangle and nudge one of its three vertices; keep the move
only if it does not shrink the minimum triangle area.  Beats the baseline but leaves
plenty of headroom.  Deterministic (seeded by n)."""
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


def main():
    n = int(sys.stdin.read().split()[0])
    T = triples(n)
    rng = random.Random(1234 + n)
    P = np.array([[rng.random(), rng.random()] for _ in range(n)])
    ar = areas(P, T)
    step = 0.35
    iters = 250
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
        if t % 400 == 399:
            step *= 0.8
    out = ["%.10f %.10f" % (P[i, 0], P[i, 1]) for i in range(n)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
