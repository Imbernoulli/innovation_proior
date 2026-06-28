#!/usr/bin/env python3
"""Instance generator for "Continuous Point Placement then Snap" (ALE-Bench).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout:

    k L
    x_0 y_0
    x_1 y_1
    ...
    x_{m-1} y_{m-1}

`k` is the number of FACILITY points we must place; `L` is the side length of
the square domain [0, L] x [0, L]; then `m` DEMAND points follow, one per line,
each an integer coordinate pair in [0, L]^2. (`m` is not given explicitly on its
own line -- the solver reads demand points until EOF.)

We must place `k` facility points at INTEGER coordinates in [0, L]^2 so as to be
close to demand yet spread out, minimizing the energy

    E = coverage - LAMBDA * dispersion
        coverage   = sum over demand d of   dist(d, nearest facility)
        dispersion = sum over facility i of min_{j != i} dist(i, j)

(see context.md / score.py for the exact rule). LAMBDA is fixed.

The demand is deliberately CLUSTERED -- a few dense blobs plus light uniform
background -- so the placement problem is non-trivial: a uniform grid of
facilities (the reference) wastes points on empty regions, while a good solver
pulls facilities onto the demand clusters (lowering coverage) while keeping them
mutually apart (raising dispersion). k, L, m and the cluster layout are all
chosen deterministically from the seed.
"""
import sys
import math
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x38C0_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    # domain side length and number of facilities, deterministic from the seed
    L = rng.randint(200, 1000)
    k = rng.randint(8, 40)
    # number of demand points: comfortably larger than k so coverage dominates
    m = rng.randint(300, 900)

    # a handful of dense demand clusters (Gaussian blobs) on a light uniform bg
    num_clusters = rng.randint(3, 8)
    clusters = []
    for _ in range(num_clusters):
        cx = rng.uniform(0.05 * L, 0.95 * L)
        cy = rng.uniform(0.05 * L, 0.95 * L)
        sd = rng.uniform(0.03, 0.10) * L  # tight blobs
        w = rng.uniform(1.0, 3.0)         # relative cluster weight
        clusters.append((cx, cy, sd, w))
    wsum = sum(c[3] for c in clusters)

    # fraction of demand that is uniform background vs in clusters
    bg_frac = rng.uniform(0.10, 0.30)

    pts = []
    for _ in range(m):
        if rng.random() < bg_frac:
            x = rng.uniform(0, L)
            y = rng.uniform(0, L)
        else:
            # pick a cluster proportional to weight, sample a Gaussian around it
            t = rng.uniform(0, wsum)
            acc = 0.0
            cx, cy, sd = clusters[0][0], clusters[0][1], clusters[0][2]
            for (ccx, ccy, csd, cw) in clusters:
                acc += cw
                if t <= acc:
                    cx, cy, sd = ccx, ccy, csd
                    break
            x = rng.gauss(cx, sd)
            y = rng.gauss(cy, sd)
        # clamp to the domain and snap to an integer demand coordinate
        xi = int(round(min(max(x, 0.0), float(L))))
        yi = int(round(min(max(y, 0.0), float(L))))
        pts.append((xi, yi))

    out = [f"{k} {L}"]
    for (x, y) in pts:
        out.append(f"{x} {y}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
