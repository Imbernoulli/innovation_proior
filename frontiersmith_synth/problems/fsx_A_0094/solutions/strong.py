# TIER: strong
# Multi-start simulated annealing over station coordinates, maximizing the exact
# minimum triangle area. Seeded (deterministic). One restart begins from the
# incircle spread; the others from random feasible layouts, letting the search
# escape the concyclic local optimum and beat the greedy ring.
import sys
import math
import random

INR = (2.0 - math.sqrt(2.0)) / 2.0
CX = INR
CY = INR
RINC = INR * 0.999
PHASE = 0.1


def min_area(pts):
    n = len(pts)
    m = float("inf")
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            ax = xj - xi
            ay = yj - yi
            for k in range(j + 1, n):
                xk, yk = pts[k]
                a = abs(ax * (yk - yi) - ay * (xk - xi))
                if a < m:
                    m = a
    return m / 2.0


def clamp(p):
    x, y = p
    if x < 1e-4:
        x = 1e-4
    if y < 1e-4:
        y = 1e-4
    if x + y > 1.0:
        s = x + y + 1e-9
        x /= s
        y /= s
    return (x, y)


def incircle(N):
    return [(CX + RINC * math.cos(2.0 * math.pi * k / N + PHASE),
             CY + RINC * math.sin(2.0 * math.pi * k / N + PHASE)) for k in range(N)]


def anneal(N, seed):
    rng = random.Random(seed)
    iters = max(5000, 80000 // N)
    restarts = 5
    gbest = -1.0
    gpts = None
    for rs in range(restarts):
        if rs == 0:
            pts = incircle(N)
        else:
            pts = [clamp((rng.random(), rng.random())) for _ in range(N)]
        v = min_area(pts)
        bestv = v
        best = list(pts)
        T = max(v, 1e-4) * 3.0
        for it in range(iters):
            T *= 0.9997
            sig = 0.02 + (0.15 if it < iters * 0.6 else 0.0)
            i = rng.randrange(N)
            old = pts[i]
            pts[i] = clamp((old[0] + rng.gauss(0.0, sig),
                            old[1] + rng.gauss(0.0, sig)))
            nv = min_area(pts)
            if nv >= v or rng.random() < math.exp((nv - v) / (T + 1e-12)):
                v = nv
                if v > bestv:
                    bestv = v
                    best = list(pts)
            else:
                pts[i] = old
        if bestv > gbest:
            gbest = bestv
            gpts = best
    return gpts


def main():
    N = int(sys.stdin.read().split()[0])
    pts = anneal(N, seed=1234 + 7 * N)
    out = ["%.10f %.10f" % (x, y) for (x, y) in pts]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
