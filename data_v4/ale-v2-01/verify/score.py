#!/usr/bin/env python3
"""Deterministic local scorer for ale-v2-01 (Capacitated Vehicle Routing).

Usage:
    python3 score.py INSTANCE_FILE SOLUTION_FILE

Reads the instance and a candidate solution, validates feasibility, and
prints a single floating-point score to stdout (higher is better).

Objective (minimize): the total Euclidean length of all vehicle routes,
where every route starts and ends at the depot.

Feasibility (ALE-Bench floor-to-0): the score is 0.0 if ANY of the
following holds:
  * a route's total client demand exceeds the capacity `cap`;
  * some client id is served more than once, or never served;
  * a client id is out of range / not an integer;
  * the solution is malformed (cannot be parsed).
A route may be empty; empty routes contribute length 0.

Reported score (continuous, higher = better):
    score = SCORE_SCALE / (total_length + 1.0)        (feasible)
    score = 0.0                                       (infeasible)
with SCORE_SCALE a fixed constant so the number is O(1e3) on these
instances. Because total_length > 0 always for a non-trivial instance,
this is a strictly decreasing function of length: a shorter (better)
set of routes yields a strictly higher score.
"""
import sys
import math

SCORE_SCALE = 1_000_000.0


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    cap = int(next(it))
    dx = int(next(it))
    dy = int(next(it))
    depot = (dx, dy)
    clients = {}      # id (1..n) -> (x, y)
    demand = {}       # id -> demand
    for i in range(1, n + 1):
        x = int(next(it))
        y = int(next(it))
        d = int(next(it))
        clients[i] = (x, y)
        demand[i] = d
    return n, cap, depot, clients, demand


def read_solution(path):
    """Solution format:
        line 1: K            (number of routes)
        next K lines: m id_1 id_2 ... id_m   (m clients on this route, in order)
    Returns a list of routes (each a list of client ids), or raises ValueError.
    """
    with open(path) as f:
        lines = [ln.strip() for ln in f.read().splitlines() if ln.strip() != ""]
    if not lines:
        raise ValueError("empty solution")
    first = lines[0].split()
    K = int(first[0])
    routes = []
    for r in range(1, K + 1):
        if r >= len(lines):
            raise ValueError("fewer route lines than K")
        parts = lines[r].split()
        m = int(parts[0])
        ids = [int(t) for t in parts[1:]]
        if len(ids) != m:
            raise ValueError("route length mismatch")
        routes.append(ids)
    return routes


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def evaluate(instance_path, solution_path):
    n, cap, depot, clients, demand = read_instance(instance_path)
    try:
        routes = read_solution(solution_path)
    except Exception:
        return 0.0, None  # malformed -> infeasible

    seen = set()
    total_len = 0.0
    for route in routes:
        load = 0
        prev = depot
        for cid in route:
            if cid < 1 or cid > n:
                return 0.0, None            # out of range
            if cid in seen:
                return 0.0, None            # duplicate service
            seen.add(cid)
            load += demand[cid]
            total_len += dist(prev, clients[cid])
            prev = clients[cid]
        if route:
            total_len += dist(prev, depot)  # return to depot
        if load > cap:
            return 0.0, None                # capacity violated

    if len(seen) != n:
        return 0.0, None                    # some client unserved

    score = SCORE_SCALE / (total_len + 1.0)
    return score, total_len


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: score.py INSTANCE SOLUTION\n")
        sys.exit(2)
    score, total_len = evaluate(sys.argv[1], sys.argv[2])
    # Print only the score on stdout (deterministic). Length to stderr for debugging.
    print(f"{score:.6f}")
    if total_len is not None:
        sys.stderr.write(f"total_length={total_len:.4f}\n")
    else:
        sys.stderr.write("INFEASIBLE\n")


if __name__ == "__main__":
    main()
