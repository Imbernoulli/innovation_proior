#!/usr/bin/env python3
"""Instance generator for "Drone Survey Sweep" (ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout:

    n
    x_0 y_0
    x_1 y_1
    ...
    x_{n-1} y_{n-1}

with 0 <= x_i, y_i <= 1_000_000 integer coordinates. The number of stations n is
chosen deterministically from the seed in [800, 2000]. Stations are drawn from a
mixture of a few 2-D Gaussian "clusters" (survey hot-spots) plus a uniform
background, clipped to the grid, which is the layout regime where nearest-neighbour
construction leaves the most slack for local search to recover.
"""
import sys
import random

SIDE = 1_000_000  # coordinate grid is [0, SIDE] in each axis


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x5EED_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    # number of stations: deterministic from the seed, in [800, 2000]
    n = rng.randint(800, 2000)

    # number of clusters and the fraction of stations drawn uniformly (background)
    num_clusters = rng.randint(3, 8)
    centers = [(rng.uniform(0, SIDE), rng.uniform(0, SIDE)) for _ in range(num_clusters)]
    # per-cluster spread, as a fraction of the side
    spreads = [rng.uniform(0.02, 0.10) * SIDE for _ in range(num_clusters)]
    bg_frac = rng.uniform(0.10, 0.35)

    pts = []
    seen = set()
    attempts = 0
    while len(pts) < n and attempts < 50 * n:
        attempts += 1
        if rng.random() < bg_frac:
            x = rng.uniform(0, SIDE)
            y = rng.uniform(0, SIDE)
        else:
            c = rng.randrange(num_clusters)
            cx, cy = centers[c]
            s = spreads[c]
            x = rng.gauss(cx, s)
            y = rng.gauss(cy, s)
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0:
            xi = 0
        elif xi > SIDE:
            xi = SIDE
        if yi < 0:
            yi = 0
        elif yi > SIDE:
            yi = SIDE
        key = (xi, yi)
        if key in seen:
            continue  # keep coordinates distinct so every edge has positive length
        seen.add(key)
        pts.append(key)

    # If clustering collided too much, top up uniformly with distinct points.
    while len(pts) < n:
        xi = rng.randint(0, SIDE)
        yi = rng.randint(0, SIDE)
        key = (xi, yi)
        if key in seen:
            continue
        seen.add(key)
        pts.append(key)

    out = [str(n)]
    out.extend(f"{x} {y}" for (x, y) in pts)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
