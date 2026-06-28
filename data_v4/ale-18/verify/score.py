#!/usr/bin/env python3
"""
Deterministic local scorer for "TSP with Time Windows (soft lateness)".

Usage:  python3 score.py INSTANCE_FILE SOLUTION_FILE
        -> prints a single floating-point score to stdout.

Solution format (stdout of the solver):
    a whitespace-separated permutation of the node ids 1..n (the visit order).
    The tour is:  depot(time 0) -> perm[0] -> perm[1] -> ... -> perm[n-1].
    The tour does NOT return to the depot.

Cost of a tour:
    Start at the depot at time t = 0.
    For each visited node v in order:
        arrival   = t_prev_departure + dist(prev, v)
        wait      : if arrival < e_v the courier waits for free until e_v
        open_time = max(arrival, e_v)        # service starts here (departure)
        lateness  = max(0.0, arrival - l_v)  # lateness measured at ARRIVAL
        t_prev_departure = open_time
    total_distance = sum of dist(prev, v) over the tour
    total_lateness = sum of lateness over all nodes
    cost = total_distance + lambda * total_lateness

Score (higher is better):
    Let EDF_cost be the cost of the earliest-deadline-first ordering (sort the
    nodes by l_i ascending; ties broken by e_i then id).  Then

        score = EDF_cost / solver_cost          (clamped to [0, SCORE_CAP])

    A solver that exactly reproduces EDF scores 1.0; a solver that halves the
    cost scores 2.0.  Lower cost  =>  higher score.

Feasibility floor:
    The score is 0.0 if the solution is NOT a valid permutation of 1..n
    (wrong length, out-of-range id, duplicate, missing id, or unparseable),
    matching ALE-Bench's "infeasible output floors the score to 0" rule.
"""
import sys
import math


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    lam = float(next(it))
    dx = float(next(it)); dy = float(next(it))
    depot = (dx, dy)
    nodes = []  # 1-indexed; nodes[0] is a placeholder
    nodes.append(None)
    for _ in range(n):
        x = float(next(it)); y = float(next(it))
        e = float(next(it)); l = float(next(it))
        nodes.append((x, y, e, l))
    return n, lam, depot, nodes


def tour_cost(perm, n, lam, depot, nodes):
    """Cost of visiting the customers in the order given by perm (1-indexed ids)."""
    total_dist = 0.0
    total_late = 0.0
    px, py = depot
    t = 0.0  # departure time from the previous location
    for v in perm:
        x, y, e, l = nodes[v]
        d = math.hypot(x - px, y - py)
        total_dist += d
        arrival = t + d
        lateness = arrival - l
        if lateness > 0.0:
            total_late += lateness
        # depart when service can start (wait for free if early)
        t = arrival if arrival >= e else e
        px, py = x, y
    return total_dist + lam * total_late


def edf_order(n, nodes):
    ids = list(range(1, n + 1))
    ids.sort(key=lambda v: (nodes[v][3], nodes[v][2], v))  # by l, then e, then id
    return ids


SCORE_CAP = 1e9  # effectively no cap; cost is always positive for n>=1


def main():
    inst_path = sys.argv[1]
    sol_path = sys.argv[2]
    n, lam, depot, nodes = read_instance(inst_path)

    # parse solution
    try:
        with open(sol_path) as f:
            sol = [int(t) for t in f.read().split()]
    except Exception:
        print(0.0)
        return

    # feasibility: must be a permutation of 1..n
    if len(sol) != n:
        print(0.0)
        return
    if sorted(sol) != list(range(1, n + 1)):
        print(0.0)
        return

    solver_cost = tour_cost(sol, n, lam, depot, nodes)
    edf_cost = tour_cost(edf_order(n, nodes), n, lam, depot, nodes)

    # both costs are > 0 because there is at least one positive-distance leg
    # (depot != node in general); guard against pathological zero anyway.
    if solver_cost <= 0.0:
        # a degenerate but valid tour with zero cost is the best possible
        score = SCORE_CAP
    else:
        score = edf_cost / solver_cost
    score = max(0.0, min(SCORE_CAP, score))
    print(repr(score))


if __name__ == "__main__":
    main()
