# TIER: strong
# Strong strategy: start from a fatter inscribed ellipse, then run a seeded,
# monotone hill-climb that jitters one station at a time and keeps a move only
# if it does not shrink the minimum triangle area AND stays inside the unit
# triangle. Deterministic (fixed seed), so the score depends only on the emitted
# coordinates. Stations drift toward the triangle's corners that a smooth ellipse
# wastes, beating the greedy ring.
import sys, math
import numpy as np
from itertools import combinations

CX = 1.0 / (2.0 + math.sqrt(2.0))
R0 = 0.28
B0 = 0.14

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

def inside(pt):
    x, y = pt
    m = 1e-7
    return (x >= m) and (y >= m) and (x + y <= 1.0 - m)

def main():
    n = int(sys.stdin.read().split()[0])
    rng = np.random.default_rng(20260702 + n)
    a, b, c = build_index(n)

    # initial: a fatter inscribed ellipse
    t = np.arange(n) * 2.0 * math.pi / n
    P = np.stack([CX + R0 * np.cos(t), CX + B0 * np.sin(t)], axis=1)
    cur = min_area(P, a, b, c)

    step = 0.12
    iters = 1500 if n <= 18 else 900
    for it in range(iters):
        k = int(rng.integers(n))
        old = P[k].copy()
        cand = old + rng.normal(0.0, step, 2)
        if not inside(cand):
            continue
        P[k] = cand
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
