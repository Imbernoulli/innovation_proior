#!/usr/bin/env python3
"""Deterministic scorer for Lattice Antenna Coverage (ale-12).

Usage:
    python3 score.py <instance_file> <solution_file>
prints a single integer: the score (0 if the solution is infeasible).

Scoring rule
------------
The solution lists a subset S of antenna-site indices. It is FEASIBLE iff:
  * the first token is k = |S| with 0 <= k <= M;
  * exactly k further indices follow, each an integer in [0, M-1];
  * the indices are pairwise distinct (no antenna chosen twice);
  * the total power cost sum_{i in S} c_i <= B.
For a feasible solution the score is the total demand of the UNION of all
lattice cells covered by the chosen antennas (each cell counted once even if
several antennas cover it). An empty set is feasible and scores 0.

Any feasibility violation -> score 0 (the hard floor). This is computed by
recomputing coverage directly from the output, independent of the solver.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    G = int(next(it)); M = int(next(it)); B = int(next(it))
    demand = [0] * (G * G)  # index = y*G + x
    for i in range(G * G):
        demand[i] = int(next(it))
    sites = []
    for _ in range(M):
        sx = int(next(it)); sy = int(next(it)); r = int(next(it)); c = int(next(it))
        sites.append((sx, sy, r, c))
    return G, M, B, demand, sites


def score(instance_path, solution_path):
    G, M, B, demand, sites = read_instance(instance_path)
    with open(solution_path) as f:
        toks = f.read().split()
    if not toks:
        return 0
    try:
        nums = [int(t) for t in toks]
    except ValueError:
        return 0
    k = nums[0]
    chosen = nums[1:]
    if k < 0 or k > M:
        return 0
    if len(chosen) != k:
        return 0
    seen = set()
    total_cost = 0
    for idx in chosen:
        if idx < 0 or idx >= M:
            return 0
        if idx in seen:
            return 0
        seen.add(idx)
        total_cost += sites[idx][3]
    if total_cost > B:
        return 0

    # Recompute the covered union from scratch and sum its demand.
    covered = bytearray(G * G)
    for idx in chosen:
        sx, sy, r, c = sites[idx]
        x0 = sx - r;  x1 = sx + r
        y0 = sy - r;  y1 = sy + r
        if x0 < 0: x0 = 0
        if y0 < 0: y0 = 0
        if x1 > G - 1: x1 = G - 1
        if y1 > G - 1: y1 = G - 1
        for y in range(y0, y1 + 1):
            base = y * G
            for x in range(x0, x1 + 1):
                covered[base + x] = 1

    total = 0
    for i in range(G * G):
        if covered[i]:
            total += demand[i]
    return total


def main():
    if len(sys.argv) < 3:
        print("usage: score.py <instance> <solution>", file=sys.stderr)
        sys.exit(1)
    print(score(sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    main()
