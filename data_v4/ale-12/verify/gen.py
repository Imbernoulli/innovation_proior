#!/usr/bin/env python3
"""Instance generator for Lattice Antenna Coverage (ale-12).

Usage: python3 gen.py <seed>  ->  writes one instance to stdout.

Instance model
--------------
A square lattice of side G (G x G demand cells). Each cell (x, y) carries an
integer demand weight d(x, y) > 0, drawn from a clustered "population" field
(a mixture of 2D Gaussian hot-spots over a uniform floor) so that demand is
spatially correlated -- this is what makes the coverage objective non-trivial
and submodular-overlap-heavy.

There are M candidate antenna *sites*, each a lattice cell, with an integer
coverage radius r_i (Chebyshev / square footprint of side 2 r_i + 1) and an
integer power cost c_i. A site placed at (sx, sy) covers every lattice cell
within its square footprint (clipped to the lattice). We must pick a subset S
of sites with sum of costs <= budget B, maximizing the total demand of the
UNION of covered cells (each covered cell counts once, regardless of how many
chosen antennas cover it -- monotone submodular coverage).

Output format (stdin of the solver), all integers:
  line 1: G M B
  line 2: G*G demand weights d, row-major (y = 0..G-1 outer, x = 0..G-1 inner)
  next M lines: sx sy r c     (site center x, y; radius; cost)
"""
import sys
import random
import math


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <seed>", file=sys.stderr)
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(seed * 1000003 + 12345)

    # Lattice size and counts scale mildly with seed so the seed set is varied
    # but every instance stays inside a fixed envelope.
    G = rng.choice([60, 70, 80, 90, 100])
    M = rng.choice([200, 300, 400, 500, 600])
    # number of demand hot-spots (shoals of population)
    K = rng.randint(4, 10)

    # ---- demand field: uniform floor + Gaussian hot-spots ----
    floor = rng.randint(1, 5)
    demand = [[floor for _ in range(G)] for _ in range(G)]
    for _ in range(K):
        cx = rng.uniform(0, G - 1)
        cy = rng.uniform(0, G - 1)
        sigma = rng.uniform(G * 0.05, G * 0.18)
        peak = rng.uniform(20.0, 120.0)
        inv = 1.0 / (2.0 * sigma * sigma)
        rad = int(min(G, 3.5 * sigma))
        x0 = max(0, int(cx) - rad); x1 = min(G - 1, int(cx) + rad)
        y0 = max(0, int(cy) - rad); y1 = min(G - 1, int(cy) + rad)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                dx = x - cx; dy = y - cy
                add = peak * math.exp(-(dx * dx + dy * dy) * inv)
                if add >= 0.5:
                    demand[y][x] += int(round(add))

    # clamp into a sane integer band
    for y in range(G):
        for x in range(G):
            if demand[y][x] < 1:
                demand[y][x] = 1
            if demand[y][x] > 999:
                demand[y][x] = 999

    # ---- candidate antenna sites ----
    # Bias site centers toward high-demand cells (where one would plausibly
    # consider placing an antenna) but keep a fraction uniform so the search
    # cannot trivially win by taking every site.
    flat = []
    total_d = 0
    for y in range(G):
        for x in range(G):
            flat.append((x, y, demand[y][x]))
            total_d += demand[y][x]
    weights = [c for (_, _, c) in flat]

    sites = []
    for _ in range(M):
        if rng.random() < 0.7:
            idx = rng.choices(range(len(flat)), weights=weights, k=1)[0]
            sx, sy, _ = flat[idx]
        else:
            sx = rng.randrange(G); sy = rng.randrange(G)
        r = rng.randint(2, max(3, G // 8))
        # cost loosely correlates with footprint area but with noise, so the
        # cost-benefit trade-off (big-cheap vs small-pricey) actually bites.
        area = (2 * r + 1) ** 2
        c = max(1, int(round(area * rng.uniform(0.6, 1.6))))
        if c > 9999:
            c = 9999
        sites.append((sx, sy, r, c))

    # Budget: enough to take a handful of antennas but nowhere near all of them.
    avg_c = sum(c for (_, _, _, c) in sites) / M
    B = int(avg_c * rng.uniform(4.0, 9.0))
    if B < 1:
        B = 1

    out = []
    out.append(f"{G} {M} {B}")
    row = []
    for y in range(G):
        for x in range(G):
            row.append(str(demand[y][x]))
    out.append(" ".join(row))
    for (sx, sy, r, c) in sites:
        out.append(f"{sx} {sy} {r} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
