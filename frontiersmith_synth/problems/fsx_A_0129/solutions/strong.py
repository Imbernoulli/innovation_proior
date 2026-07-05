# TIER: strong
# Start from the Hammersley set, then run seeded multi-restart coordinate
# descent that directly minimizes the EXACT 2-D star discrepancy (the same
# objective the checker measures). Deterministic (fixed seeds), fast for the
# small M in this ladder, and strictly better than the greedy Hammersley set.
import sys
import random
import numpy as np


def disc_np(pts):
    pts = np.asarray(pts, float)
    n = len(pts)
    xs = np.unique(pts[:, 0])
    ys = np.unique(pts[:, 1])
    xg = np.append(xs, 1.0)
    yg = np.append(ys, 1.0)
    Xo = (pts[:, 0][:, None] < xg[None, :]).astype(np.float64)
    Yo = (pts[:, 1][:, None] < yg[None, :]).astype(np.float64)
    Co = Xo.T @ Yo
    plus = np.max(np.outer(xg, yg) - Co / n)
    Xc = (pts[:, 0][:, None] <= xs[None, :]).astype(np.float64)
    Yc = (pts[:, 1][:, None] <= ys[None, :]).astype(np.float64)
    Cc = Xc.T @ Yc
    minus = np.max(Cc / n - np.outer(xs, ys))
    return float(max(plus, minus))


def rev2(i):
    r = 0.0
    f = 0.5
    while i > 0:
        r += (i & 1) * f
        f *= 0.5
        i >>= 1
    return r


def hammersley(M):
    return [[(i + 0.5) / M, rev2(i)] for i in range(M)]


def optimize(M, seed=777, restarts=4, iters=1800):
    best_pts = None
    best = 9.9
    for s in range(restarts):
        rng = random.Random(seed + s * 101)
        if s == 0:
            pts = np.array(hammersley(M), float)
        else:
            pts = np.array([[rng.random(), rng.random()] for _ in range(M)])
        cur = disc_np(pts)
        step = 0.22
        for it in range(iters):
            k = rng.randrange(M)
            dim = rng.randrange(2)
            old = pts[k, dim]
            pts[k, dim] = min(1.0, max(0.0, old + rng.uniform(-step, step)))
            d = disc_np(pts)
            if d < cur - 1e-12:
                cur = d
            else:
                pts[k, dim] = old
            if (it + 1) % 600 == 0:
                step *= 0.7
        if cur < best:
            best = cur
            best_pts = pts.copy()
    return best_pts


def main():
    toks = sys.stdin.read().split()
    d, M = int(toks[0]), int(toks[1])
    pts = optimize(M)
    out = []
    for i in range(M):
        out.append("%.10f %.10f" % (pts[i, 0], pts[i, 1]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
