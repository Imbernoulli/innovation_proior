#!/usr/bin/env python3
"""Deterministic scorer for ale-v2-05 "Facility Location with Opening Cost" (UFLP).

Usage:
    python3 score.py INSTANCE_FILE SOLUTION_FILE [--cost]

Reads the instance and a candidate solution, validates feasibility, and prints
a single number on stdout:

  * default            -> the ALE-Bench continuous SCORE (higher is better,
                          0.0 if the solution is infeasible).
  * with --cost flag   -> the raw objective COST (open cost + service cost,
                          the thing to MINIMIZE). On an infeasible solution the
                          script exits non-zero (callers must use the default
                          score for the floor semantics).

Instance format (the generator's stdout):
    line 1:  F C
    next F lines:  fx fy fcost          (facility coords and opening cost)
    next C lines:  cx cy                (client coords)

Solution format (the solver's stdout):
    line 1:  M                          (number of opened facilities; M >= 1)
    next M lines: an integer index in [1..F]   (1-based facility id)

Feasibility requirements (any violation => SCORE 0.0):
    * M parses as an integer with 1 <= M <= F          (must open at least one)
    * exactly M further integer tokens are present
    * every index parses as an integer in [1..F]
    * all M indices are DISTINCT

Objective (to MINIMIZE):
    total = sum_{i in S} opening_cost[i]
          + sum_{c}      min_{i in S} euclid(client c, facility i)
where S is the opened set. With S non-empty every client has a nearest open
facility, so service cost is always finite.

ALE score (to MAXIMIZE; this is what the judge reports):
    score = 0.0                                  if infeasible
    score = round(SCORE_SCALE / (1 + total/C))   otherwise
  where C is the client count. Lower total -> higher score; an empty / wrong /
  out-of-range / duplicate / M==0 output floors the score to exactly 0.0.
"""
import sys
import math

SCORE_SCALE = 1_000_000_000.0  # fixed normalisation constant (frozen)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    F = int(next(it))
    C = int(next(it))
    fac = []
    for _ in range(F):
        x = int(next(it))
        y = int(next(it))
        cost = int(next(it))
        fac.append((x, y, cost))
    cli = []
    for _ in range(C):
        x = int(next(it))
        y = int(next(it))
        cli.append((x, y))
    return F, C, fac, cli


def parse_solution(path, F):
    """Return list of 0-based distinct facility indices, or None if infeasible."""
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
    if M < 1 or M > F:          # must open at least one, at most F
        return None
    body = toks[1:]
    if len(body) != M:
        return None
    chosen = []
    seen = set()
    for i in range(M):
        try:
            idx = int(body[i])
        except ValueError:
            return None
        if idx < 1 or idx > F:
            return None
        if idx in seen:
            return None          # duplicates are infeasible
        seen.add(idx)
        chosen.append(idx - 1)
    return chosen


def compute_cost(fac, cli, chosen):
    """open cost + sum over clients of distance to the nearest open facility."""
    total = 0.0
    for c in chosen:
        total += fac[c][2]
    fx = [fac[c][0] for c in chosen]
    fy = [fac[c][1] for c in chosen]
    for (x, y) in cli:
        best = None
        for j in range(len(chosen)):
            dx = x - fx[j]
            dy = y - fy[j]
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
    F, C, fac, cli = read_instance(inst_path)
    chosen = parse_solution(sol_path, F)
    if chosen is None:
        if want_cost:
            sys.stderr.write("infeasible\n")
            sys.exit(1)
        print(0.0)
        return
    cost = compute_cost(fac, cli, chosen)
    if want_cost:
        print(repr(cost))
        return
    score = round(SCORE_SCALE / (1.0 + cost / C))
    print(score)


if __name__ == "__main__":
    main()
