#!/usr/bin/env python3
"""Instance generator for "Cable Layout" (rectilinear Steiner tree, ALE-Bench).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout:

    n SIDE
    x_0 y_0
    x_1 y_1
    ...
    x_{n-1} y_{n-1}

with 0 <= x_i, y_i <= SIDE integer coordinates, all DISTINCT. The number of
terminals n is chosen deterministically from the seed in [200, 600].

Terminals are drawn from a mixture of a few 2-D Gaussian "pin clusters" (the
dense functional blocks of a board) plus a uniform background. Clustered layouts
are exactly where a Steiner tree -- which can introduce shared trunk wires at
Hanan-grid junctions -- saves the most over a plain rectilinear MST that routes
every connection as an independent L-shape.
"""
import sys
import random

SIDE = 10_000  # coordinate grid is [0, SIDE] in each axis


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x5731_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    # number of terminals: deterministic from the seed, in [200, 600]
    n = rng.randint(200, 600)

    num_clusters = rng.randint(3, 8)
    centers = [(rng.uniform(0, SIDE), rng.uniform(0, SIDE)) for _ in range(num_clusters)]
    spreads = [rng.uniform(0.03, 0.12) * SIDE for _ in range(num_clusters)]
    bg_frac = rng.uniform(0.10, 0.35)

    pts = []
    seen = set()
    attempts = 0
    while len(pts) < n and attempts < 80 * n:
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
            continue  # keep coordinates distinct
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

    out = [f"{n} {SIDE}"]
    out.extend(f"{x} {y}" for (x, y) in pts)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
