# TIER: strong
# Strong strategy: start from a full inscribed circle (the strongest simple ring),
# then run a seeded, monotone hill-climb that jitters one sensor at a time and keeps
# a move only if it does not shrink the minimum triangle area. Deterministic (fixed
# seed), so the score depends only on the emitted coordinates.
import sys, math
import numpy as np
from itertools import combinations

def build_index(n):
    tri = list(combinations(range(n), 3))
    a = np.array([t[0] for t in tri]); b = np.array([t[1] for t in tri]); c = np.array([t[2] for t in tri])
    return a, b, c

def min_area(P, a, b, c):
    ax, ay = P[a, 0], P[a, 1]
    bx, by = P[b, 0], P[b, 1]
    cx, cy = P[c, 0], P[c, 1]
    area = np.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) * 0.5
    return float(area.min())

def main():
    n = int(sys.stdin.read().split()[0])
    rng = np.random.default_rng(20260701 + n)
    a, b, c = build_index(n)

    # initial: inscribed circle
    t = np.arange(n) * 2.0 * math.pi / n
    P = np.stack([0.5 + 0.5 * np.cos(t), 0.5 + 0.5 * np.sin(t)], axis=1)
    cur = min_area(P, a, b, c)

    step = 0.20
    iters = 1200 if n <= 20 else 700
    for it in range(iters):
        k = int(rng.integers(n))
        old = P[k].copy()
        P[k] = np.clip(old + rng.normal(0.0, step, 2), 0.0, 1.0)
        v = min_area(P, a, b, c)
        if v >= cur:
            cur = v
        else:
            P[k] = old
        if (it + 1) % max(1, iters // 6) == 0:
            step *= 0.7

    out = ["%.15g %.15g" % (P[i, 0], P[i, 1]) for i in range(n)]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
