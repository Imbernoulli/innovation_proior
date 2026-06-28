#!/usr/bin/env python3
"""Deterministic local scorer for "Machine Assignment with Sequence-Dependent
Setups".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score. HIGHER is better.

Scoring rule (see context.md "Evaluation settings"):
  * Instance:
        n M T
        d_0 ... d_{n-1}          (durations)
        c_0 ... c_{n-1}          (types in [0, T))
        init_0 ... init_{T-1}    (initial setup per type)
        s[a][b]                  (T x T setup matrix)
  * SOLUTION format (per machine an ordered job list):
        M lines. Line m (0-based machine m) is:
            k  j_1 j_2 ... j_k
        where k >= 0 is the number of jobs on machine m and j_1..j_k are the
        0-based job indices in the order they run on that machine. A machine may
        be empty (k = 0). Extra leading/trailing whitespace is ignored.

  * FEASIBILITY (any violation -> score 0):
      - the file parses as exactly M lines, each starting with a non-negative k
        followed by exactly k integers;
      - every listed job index is in [0, n);
      - the multiset of all listed jobs is EXACTLY {0, 1, ..., n-1}: every job
        appears exactly once (no job unassigned, no job duplicated).
    If any of these fail, the solution is INFEASIBLE and scores 0.

  * COST (lower is better) of a feasible solution. For each machine m with job
    order (j_1, ..., j_k):
        load(m) = sum_t d[j_t]                                  # processing
                + init[c[j_1]]                                  # first setup
                + sum_{t=2..k} s[c[j_{t-1}]][c[j_t]]            # changeovers
    (an empty machine has load 0). The objective is the TOTAL:
        cost = sum_m load(m).
    (The maximum-load / makespan variant is NOT used; the stated objective is
    total completion + setup time.)

  * SCORE (higher better), normalized against a deterministic balanced
    round-robin baseline the scorer recomputes itself:
        score = round(1_000_000 * baseline_cost / max(1, solver_cost))
    The baseline scores ~1_000_000; a lower-cost (better) schedule scores more.
    INFEASIBLE -> 0.
"""
import sys


# ----------------------------------------------------------------------------- IO
def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    M = int(next(it))
    T = int(next(it))
    d = [int(next(it)) for _ in range(n)]
    c = [int(next(it)) for _ in range(n)]
    init = [int(next(it)) for _ in range(T)]
    s = [[int(next(it)) for _ in range(T)] for _ in range(T)]
    return n, M, T, d, c, init, s


def read_solution(path, n, M):
    """Parse + validate the per-machine job lists.

    Return list-of-lists (machine -> ordered job indices) or None if infeasible.
    """
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    # Drop trailing blank lines but keep structure: we require exactly M
    # non-empty (token-bearing) lines. We tolerate blank lines anywhere by
    # ignoring lines with no tokens, then requiring exactly M remaining.
    rows = []
    for ln in lines:
        toks = ln.split()
        if toks:
            rows.append(toks)
    if len(rows) != M:
        return None

    machines = []
    seen = [0] * n
    for toks in rows:
        try:
            vals = [int(t) for t in toks]
        except ValueError:
            return None
        k = vals[0]
        if k < 0:
            return None
        if len(vals) - 1 != k:
            return None
        jobs = vals[1:]
        for j in jobs:
            if j < 0 or j >= n:
                return None
            seen[j] += 1
            if seen[j] > 1:
                return None
        machines.append(jobs)

    # every job assigned exactly once
    if any(v != 1 for v in seen):
        return None
    return machines


# --------------------------------------------------------------------------- cost
def total_cost(machines, d, c, init, s):
    total = 0
    for jobs in machines:
        if not jobs:
            continue
        load = 0
        for j in jobs:
            load += d[j]
        load += init[c[jobs[0]]]
        for t in range(1, len(jobs)):
            load += s[c[jobs[t - 1]]][c[jobs[t]]]
        total += load
    return total


# ------------------------------------------------------- baseline: round-robin
def baseline_cost(n, M, T, d, c, init, s):
    """Balanced round-robin: job j -> machine (j mod M), in input order. This
    ignores types entirely, so it pays a cross-type setup on almost every
    adjacency -- exactly what a good solver avoids by grouping types."""
    machines = [[] for _ in range(M)]
    for j in range(n):
        machines[j % M].append(j)
    return total_cost(machines, d, c, init, s)


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, M, T, d, c, init, s = read_instance(sys.argv[1])

    machines = read_solution(sys.argv[2], n, M)
    if machines is None:
        print(0)  # INFEASIBLE -> floored to 0
        return

    cost = total_cost(machines, d, c, init, s)
    base = baseline_cost(n, M, T, d, c, init, s)
    score = int(round(1_000_000.0 * base / max(1, cost)))
    print(score)


if __name__ == "__main__":
    main()
