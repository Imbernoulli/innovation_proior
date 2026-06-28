#!/usr/bin/env python3
"""Instance generator for ale-05 "Relay Tower Placement" (p-median essence).

Usage:  python3 gen.py SEED  > instance.txt

Instance format (stdout):
    line 1:  N K
    next N lines:  x y        (integer coordinates, 0 <= x,y <= COORD_MAX)

Semantics: N households on a city plane; we must place K relay towers, each AT
a household location (p-median). Minimize the sum over households of the
Euclidean distance to the nearest chosen tower.

Households are drawn from a random mixture of 2D-Gaussian "neighbourhood"
clusters (clipped to the plane), which is the realistic, clustered regime where
medoid placement matters. Everything is a deterministic function of SEED.
"""
import sys
import random

COORD_MAX = 1_000_000


def gen(seed: int):
    rng = random.Random(seed * 2654435761 + 12345)

    # Problem size: moderate N so an O(N*K) PAM swap is the lever, K modest.
    N = rng.randint(800, 1200)
    K = rng.randint(8, 20)

    # Number of latent clusters (neighbourhoods) the households come from.
    n_clusters = rng.randint(K, K + 12)
    centers = []
    for _ in range(n_clusters):
        cx = rng.uniform(0.05, 0.95) * COORD_MAX
        cy = rng.uniform(0.05, 0.95) * COORD_MAX
        # spread per cluster; some tight, some loose
        sigma = rng.uniform(0.01, 0.08) * COORD_MAX
        weight = rng.uniform(0.5, 2.0)
        centers.append((cx, cy, sigma, weight))

    wsum = sum(c[3] for c in centers)
    pts = []
    for _ in range(N):
        # pick a cluster by weight
        r = rng.uniform(0, wsum)
        acc = 0.0
        ci = 0
        for j, c in enumerate(centers):
            acc += c[3]
            if r <= acc:
                ci = j
                break
        cx, cy, sigma, _ = centers[ci]
        # small fraction of uniform "rural" noise points
        if rng.random() < 0.05:
            x = rng.uniform(0, COORD_MAX)
            y = rng.uniform(0, COORD_MAX)
        else:
            x = rng.gauss(cx, sigma)
            y = rng.gauss(cy, sigma)
        xi = min(COORD_MAX, max(0, int(round(x))))
        yi = min(COORD_MAX, max(0, int(round(y))))
        pts.append((xi, yi))

    return N, K, pts


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py SEED\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    N, K, pts = gen(seed)
    out = [f"{N} {K}"]
    for (x, y) in pts:
        out.append(f"{x} {y}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
