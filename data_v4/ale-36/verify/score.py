#!/usr/bin/env python3
"""Scorer for single-machine total weighted tardiness (1 || sum w_j T_j).

Usage: python3 score.py INSTANCE_FILE SOLUTION_FILE  ->  prints one float (the score).

Scoring rule (continuous, higher is better, infeasible -> 0):

  1. Read the instance: n jobs with (proc, weight, due).
  2. Read the solution: a sequence of job ids. It is FEASIBLE iff it is exactly a
     permutation of {0, ..., n-1} (each id once, all in range). If not a permutation
     -> score 0 (the feasibility floor).
  3. Completion times along the order: C_k = sum of proc of the first k jobs.
     Weighted tardiness WT = sum_j w_j * max(0, C_j - d_j).
  4. Baseline: the earliest-due-date (EDD) order, ties broken by job id. Let
     WT_edd be its weighted tardiness.
  5. Score = 1e6 * (WT_edd + 1) / (WT + 1).
     Lower weighted tardiness => higher score. The EDD baseline scores ~1e6;
     any schedule strictly better than EDD scores above 1e6, worse scores below.
     An optimal (WT = 0) schedule approaches 1e6 * (WT_edd + 1).

This scorer is deterministic and self-contained.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    p = [0] * n
    w = [0] * n
    d = [0] * n
    for i in range(n):
        p[i] = int(next(it))
        w[i] = int(next(it))
        d[i] = int(next(it))
    return n, p, w, d


def weighted_tardiness(order, p, w, d):
    C = 0
    tot = 0
    for j in order:
        C += p[j]
        late = C - d[j]
        if late > 0:
            tot += w[j] * late
    return tot


def main():
    inst_path = sys.argv[1]
    sol_path = sys.argv[2]
    n, p, w, d = read_instance(inst_path)

    with open(sol_path) as f:
        sol_toks = f.read().split()

    # Parse the solution as a list of ints. Any non-int token => infeasible.
    order = []
    try:
        for t in sol_toks:
            order.append(int(t))
    except ValueError:
        print(0.0)
        return

    # Feasibility: must be exactly a permutation of 0..n-1.
    if len(order) != n:
        print(0.0)
        return
    if sorted(order) != list(range(n)):
        print(0.0)
        return

    wt = weighted_tardiness(order, p, w, d)

    # EDD baseline (ties by job id).
    edd = sorted(range(n), key=lambda j: (d[j], j))
    wt_edd = weighted_tardiness(edd, p, w, d)

    score = 1e6 * (wt_edd + 1) / (wt + 1)
    print(score)


if __name__ == "__main__":
    main()
