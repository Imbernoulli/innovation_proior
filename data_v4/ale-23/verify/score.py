#!/usr/bin/env python3
"""Deterministic local scorer for "Resource-Constrained Project Scheduling (RCPSP)".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score. HIGHER is better. INFEASIBLE -> 0.

Scoring rule (see context.md "Evaluation settings"):

  * INSTANCE:
        n R
        cap_1 ... cap_R
        then n lines, task i (1-indexed):
            dur  d_{i,1} ... d_{i,R}  p  pred_1 ... pred_p
    Discrete time starting at 0. Resource k has constant capacity cap_k. Task i
    runs for dur_i time units over [s_i, s_i + dur_i) and consumes d_{i,k} of
    resource k while running. Finish-to-start precedence: s_i >= s_j + dur_j for
    every predecessor j of i.

  * SOLUTION format (read from <solution_file>): n integers s_1 ... s_n
    (whitespace-separated, any layout), the start time of each task in input
    order.

  * FEASIBILITY (any violation -> score 0):
      - the file parses as exactly n non-negative integers;
      - every start time s_i >= 0;
      - precedence: for every task i and predecessor j, s_i >= s_j + dur_j;
      - resource: at no integer time t does the total demand of the tasks running
        at t exceed any cap_k. (Task i is "running at t" iff s_i <= t < s_i+dur_i.)
    If any of these fail, the whole solution is INFEASIBLE and scores 0.

  * MAKESPAN (lower is better) of a feasible solution:
        makespan = max_i (s_i + dur_i)          (0 if n == 0)

  * SCORE (higher better), normalized against the deterministic earliest-start
    list baseline the scorer recomputes itself (serial SGS with the input-order
    priority list):
        score = round(1_000_000 * baseline_makespan / max(1, solver_makespan))
    The earliest-start baseline scores ~1_000_000; a shorter (better) schedule
    scores more. INFEASIBLE -> 0.
"""
import sys


# ----------------------------------------------------------------------------- IO
def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    R = int(next(it))
    caps = [int(next(it)) for _ in range(R)]
    dur = [0] * n
    demand = [[0] * R for _ in range(n)]
    preds = [[] for _ in range(n)]
    for i in range(n):
        dur[i] = int(next(it))
        for k in range(R):
            demand[i][k] = int(next(it))
        p = int(next(it))
        for _ in range(p):
            j = int(next(it)) - 1  # to 0-indexed
            preds[i].append(j)
    return n, R, caps, dur, demand, preds


def read_solution(path, n):
    """Parse n non-negative integer start times. Return list or None if bad."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if len(toks) != n:
        return None
    starts = []
    for t in toks:
        try:
            v = int(t)
        except ValueError:
            return None
        if v < 0:
            return None
        starts.append(v)
    return starts


# ----------------------------------------------------------------- feasibility
def feasible_makespan(n, R, caps, dur, demand, preds, starts):
    """Return (True, makespan) if the schedule is valid, else (False, None)."""
    if n == 0:
        return True, 0
    # precedence
    for i in range(n):
        for j in preds[i]:
            if starts[i] < starts[j] + dur[j]:
                return False, None
    # resource profile via a sweep of start/end events. For each time unit in
    # [s_i, s_i + dur_i) the task adds demand; we accumulate usage with a
    # difference array over the (small) time horizon.
    horizon = 0
    for i in range(n):
        horizon = max(horizon, starts[i] + dur[i])
    # difference arrays, one per resource, length horizon+1
    diff = [[0] * (horizon + 2) for _ in range(R)]
    for i in range(n):
        s = starts[i]
        e = s + dur[i]
        for k in range(R):
            d = demand[i][k]
            if d:
                diff[k][s] += d
                diff[k][e] -= d
    for k in range(R):
        run = 0
        for t in range(horizon + 1):
            run += diff[k][t]
            if run > caps[k]:
                return False, None
    return True, horizon


# ---------------------------------------------------- baseline: earliest-start SGS
def baseline_makespan(n, R, caps, dur, demand, preds):
    """Serial schedule-generation scheme with the input-order priority list.

    Repeatedly pick, among the not-yet-scheduled tasks whose predecessors are all
    scheduled, the one with the smallest input index (earliest in the trivial
    priority list); schedule it at the earliest time >= max predecessor finish at
    which every resource has room for the whole duration. This is the standard
    "earliest-start list" reference -- always feasible, so a legitimate
    normalizer.
    """
    if n == 0:
        return 0
    scheduled = [False] * n
    starts = [0] * n
    indeg = [len(preds[i]) for i in range(n)]
    # successors
    succ = [[] for _ in range(n)]
    for i in range(n):
        for j in preds[i]:
            succ[j].append(i)
    # resource usage profile: usage[k][t]
    horizon_cap = sum(dur) + 1
    usage = [[0] * (horizon_cap + 1) for _ in range(R)]

    def earliest_feasible(i, t0):
        d = dur[i]
        t = t0
        while True:
            ok = True
            for k in range(R):
                dem = demand[i][k]
                if dem == 0:
                    continue
                for tt in range(t, t + d):
                    if usage[k][tt] + dem > caps[k]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return t
            t += 1

    for _ in range(n):
        # ready tasks = unscheduled with indeg 0
        cand = -1
        for i in range(n):
            if not scheduled[i] and indeg[i] == 0:
                cand = i
                break
        # earliest start = max predecessor finish
        t0 = 0
        for j in preds[cand]:
            t0 = max(t0, starts[j] + dur[j])
        s = earliest_feasible(cand, t0)
        starts[cand] = s
        scheduled[cand] = True
        for k in range(R):
            dem = demand[cand][k]
            if dem:
                for tt in range(s, s + dur[cand]):
                    usage[k][tt] += dem
        for c in succ[cand]:
            indeg[c] -= 1

    return max(starts[i] + dur[i] for i in range(n))


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, R, caps, dur, demand, preds = read_instance(sys.argv[1])

    starts = read_solution(sys.argv[2], n)
    if starts is None:
        print(0)
        return

    ok, mk = feasible_makespan(n, R, caps, dur, demand, preds, starts)
    if not ok:
        print(0)
        return

    base = baseline_makespan(n, R, caps, dur, demand, preds)
    score = int(round(1_000_000.0 * base / max(1, mk)))
    print(score)


if __name__ == "__main__":
    main()
