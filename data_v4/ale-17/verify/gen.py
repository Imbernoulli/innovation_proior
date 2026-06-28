#!/usr/bin/env python3
"""Instance generator for the Capacitated Multi-Vehicle Routing problem (ale-17).

Usage: python3 gen.py <seed>  ->  writes one instance to stdout.

Instance format (stdin of the solver):
    line 1: n K Q
    line 2: depot_x depot_y          (the depot, index 0 conceptually)
    next n lines: x y demand         (customer i, for i = 1..n)

All coordinates are integers in [0, 1000]. Demands are positive integers.
The instance is generated so that a feasible assignment of customers to K
vehicles each with capacity Q always exists: sum(demand) <= K * Q, and every
single demand <= Q.
"""
import sys
import random


def gen(seed: int):
    rng = random.Random(seed * 1_000_003 + 12345)

    # Problem size: medium-scale heuristic instances.
    n = rng.randint(120, 200)          # number of customers
    K = rng.randint(6, 12)             # number of vehicles

    L = 1000                           # coordinate grid side

    # Depot near the centre (classic CVRP layout) or a corner sometimes.
    if rng.random() < 0.5:
        depot = (L // 2, L // 2)
    else:
        depot = (rng.randint(0, L), rng.randint(0, L))

    # Customers: a mix of clustered shoals and uniform scatter, integer coords.
    n_clusters = rng.randint(2, 6)
    centres = [(rng.randint(0, L), rng.randint(0, L)) for _ in range(n_clusters)]
    pts = []
    for _ in range(n):
        if rng.random() < 0.7:
            cx, cy = rng.choice(centres)
            x = int(round(rng.gauss(cx, L * 0.07)))
            y = int(round(rng.gauss(cy, L * 0.07)))
        else:
            x = rng.randint(0, L)
            y = rng.randint(0, L)
        x = max(0, min(L, x))
        y = max(0, min(L, y))
        pts.append((x, y))

    # Demands: positive integers. Each demand small relative to Q.
    demands = [rng.randint(1, 30) for _ in range(n)]
    total = sum(demands)

    # Capacity Q chosen so the instance is tight but feasible: leave some slack
    # so K vehicles can carry everything, but not so much that 1-2 routes do all
    # the work (which would make balancing trivial).
    # Need K * Q >= total and Q >= max(demand). Aim for utilisation ~0.80-0.92.
    max_d = max(demands)
    util = rng.uniform(0.80, 0.92)
    Q = int(total / (K * util))
    Q = max(Q, max_d, total // K + 1)
    # Guarantee feasibility margin.
    while K * Q < total:
        Q += 1

    out = []
    out.append(f"{n} {K} {Q}")
    out.append(f"{depot[0]} {depot[1]}")
    for (x, y), d in zip(pts, demands):
        out.append(f"{x} {y} {d}")
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    sys.stdout.write(gen(seed))


if __name__ == "__main__":
    main()
