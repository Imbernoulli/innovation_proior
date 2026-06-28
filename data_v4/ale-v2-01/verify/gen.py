#!/usr/bin/env python3
"""Instance generator for ale-v2-01 (Capacitated Vehicle Routing).

Usage: python3 gen.py SEED  >  instance.txt

Instance format (stdin contract of the solver):
    line 1: n cap
    line 2: depot_x depot_y          (the depot, client id 0 conceptually)
    next n lines: x y demand         (client i, 1-based id, demand >= 1)

All coordinates are integers in [0, GRID]. Demands are integers in
[1, cap//2] so that no single client exceeds capacity and a route holds
at least two clients on average. n is drawn so instances are non-trivial
but solvable within the time budget.
"""
import sys
import random

GRID = 1000


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rng = random.Random(seed * 1_000_003 + 12345)

    # Problem size: 120..200 clients. Capacity scaled so a route serves a
    # handful of clients (forces multiple vehicles -> capacity is binding).
    n = rng.randint(120, 200)
    cap = rng.randint(40, 70)
    max_demand = max(2, cap // 4)

    # Depot near the centre.
    dx = rng.randint(GRID // 2 - 50, GRID // 2 + 50)
    dy = rng.randint(GRID // 2 - 50, GRID // 2 + 50)

    # Clients: a mixture of a few Gaussian clusters plus uniform noise,
    # which is the standard structure that makes savings/LNS interesting.
    k_clusters = rng.randint(3, 6)
    centers = [(rng.randint(0, GRID), rng.randint(0, GRID)) for _ in range(k_clusters)]

    clients = []
    for _ in range(n):
        if rng.random() < 0.75:
            cx, cy = rng.choice(centers)
            x = int(round(rng.gauss(cx, 90)))
            y = int(round(rng.gauss(cy, 90)))
        else:
            x = rng.randint(0, GRID)
            y = rng.randint(0, GRID)
        x = min(GRID, max(0, x))
        y = min(GRID, max(0, y))
        d = rng.randint(1, max_demand)
        clients.append((x, y, d))

    out = []
    out.append(f"{n} {cap}")
    out.append(f"{dx} {dy}")
    for (x, y, d) in clients:
        out.append(f"{x} {y} {d}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
