#!/usr/bin/env python3
"""Deterministic scorer for ale-05 "Relay Tower Placement" (p-median).

Usage:
    python3 score.py INSTANCE_FILE SOLUTION_FILE [--cost]

Reads the instance and a candidate solution, validates feasibility, and prints
a single number on stdout:

  * default            -> the ALE-Bench continuous SCORE (higher is better,
                          0.0 if the solution is infeasible).
  * with --cost flag   -> the raw objective COST (sum of nearest distances,
                          the thing to MINIMIZE). Infeasible -> +inf is NOT
                          printed; instead the script exits non-zero so callers
                          must use the default score for the floor semantics.

Solution format (the solver's stdout):
    line 1:  M               (number of chosen towers; must equal K)
    next M lines: an integer index in [1..N]  (1-based household id)

Feasibility requirements (any violation => SCORE 0):
    * M == K
    * every index parses as an integer in [1..N]
    * all K indices are DISTINCT

Objective (to minimize):
    cost = sum over all households h of  min over chosen towers t of
           euclidean_distance(h, t)

ALE score (to maximize; this is what the judge reports):
    score = 0.0                              if infeasible
    score = round(SCORE_SCALE / (1 + cost/N))   otherwise
  where N is the household count. Lower cost  ->  higher score; an empty / wrong
  output floors the score to exactly 0.0.
"""
import sys
import math

SCORE_SCALE = 1_000_000_000.0  # fixed normalisation constant (frozen)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    N = int(next(it))
    K = int(next(it))
    pts = []
    for _ in range(N):
        x = int(next(it))
        y = int(next(it))
        pts.append((x, y))
    return N, K, pts


def parse_solution(path, N, K):
    """Return list of 0-based distinct indices, or None if infeasible."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if not toks:
        return None
    try:
        M = int(toks[0])
    except ValueError:
        return None
    if M != K:
        return None
    body = toks[1:]
    if len(body) < K:
        return None
    chosen = []
    seen = set()
    for i in range(K):
        try:
            idx = int(body[i])
        except ValueError:
            return None
        if idx < 1 or idx > N:
            return None
        if idx in seen:
            return None  # duplicates are infeasible
        seen.add(idx)
        chosen.append(idx - 1)
    return chosen


def compute_cost(pts, chosen):
    """Sum over households of distance to nearest chosen tower."""
    total = 0.0
    cx = [pts[c][0] for c in chosen]
    cy = [pts[c][1] for c in chosen]
    for (x, y) in pts:
        best = None
        for j in range(len(chosen)):
            dx = x - cx[j]
            dy = y - cy[j]
            d2 = dx * dx + dy * dy
            if best is None or d2 < best:
                best = d2
        total += math.sqrt(best)
    return total


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_cost = "--cost" in sys.argv[1:]
    if len(args) < 2:
        sys.stderr.write("usage: score.py INSTANCE SOLUTION [--cost]\n")
        sys.exit(2)
    inst_path, sol_path = args[0], args[1]
    N, K, pts = read_instance(inst_path)
    chosen = parse_solution(sol_path, N, K)
    if chosen is None:
        if want_cost:
            sys.stderr.write("infeasible\n")
            sys.exit(1)
        print(0.0)
        return
    cost = compute_cost(pts, chosen)
    if want_cost:
        print(repr(cost))
        return
    score = round(SCORE_SCALE / (1.0 + cost / N))
    print(score)


if __name__ == "__main__":
    main()
