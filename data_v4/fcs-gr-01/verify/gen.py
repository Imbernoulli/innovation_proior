#!/usr/bin/env python3
"""Random small-case generator for Project Selection (Max-Weight Closure).

Usage: python3 gen.py <seed>

Prints a valid instance on stdout in the problem's input format:
    n m
    profit[0] ... profit[n-1]
    cost[0] ... cost[m-1]
    E
    (i j) * E      # project i (1-based) requires machine j (1-based)

Kept small so the 2^n brute force is feasible (n <= 12).
"""
import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(0, 12)
    m = rng.randint(0, 8)

    # Profits: mostly positive but allow negatives and zeros to exercise the
    # general closure formulation. Magnitudes vary so the trade-off is real.
    profit = []
    for _ in range(n):
        r = rng.random()
        if r < 0.7:
            profit.append(rng.randint(1, 40))
        elif r < 0.85:
            profit.append(rng.randint(-30, -1))
        else:
            profit.append(0)

    # Costs are non-negative.
    cost = [rng.randint(0, 35) for _ in range(m)]

    # Prerequisite edges (no duplicates). Each project independently requires a
    # random subset of machines.
    edges = []
    if m > 0:
        for i in range(n):
            for j in range(m):
                if rng.random() < 0.35:
                    edges.append((i + 1, j + 1))

    out = []
    out.append(f"{n} {m}")
    out.append(" ".join(map(str, profit)))
    out.append(" ".join(map(str, cost)))
    out.append(str(len(edges)))
    for (i, j) in edges:
        out.append(f"{i} {j}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
