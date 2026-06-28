#!/usr/bin/env python3
"""Deterministic local scorer for "Factory Job-Shop Scheduling".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single number: the score (an integer). A higher score is better.

Scoring rule (see context.md "Evaluation settings"):
  * The instance has n jobs and m machines. Job j is a chain of m operations
    op(j,0), op(j,1), ..., op(j,m-1) that MUST run in that order. Operation
    op(j,t) runs on machine M[j][t] (the machine lists are permutations of
    0..m-1, so each machine handles exactly one operation of each job, n in all)
    and takes P[j][t] time. A machine runs one operation at a time (no preemption).

  * A SOLUTION specifies, for each machine k (k = 0..m-1, one line per machine in
    increasing k), the ORDER in which that machine processes the n jobs' operations
    -- a permutation of the job indices {0,...,n-1}.

  * Given those m machine orders together with the per-job chain order, every
    operation gets a start time S(j,t) determined by the longest-path / earliest
    feasible schedule:
        S(j,t) >= S(j,t-1) + P(j,t-1)                 (job chain precedence)
        S(j,t) >= S(prev_on_machine) + P(prev_on_mach) (machine order precedence)
    The MAKESPAN is max over all ops of S(j,t) + P(j,t), i.e. the longest path in
    the disjunctive graph induced by the chosen machine orders. The objective is to
    MINIMIZE the makespan.

  * FEASIBILITY: the output must be exactly m lines, line k a permutation of
    {0,...,n-1}. If the token shape is wrong (wrong line/token count, an
    out-of-range or repeated job index, garbage, a missing file) OR if the chosen
    machine orders + job chains contain a CYCLE (a deadlock: no consistent start
    times exist), the solution is INFEASIBLE and the score is 0.

  * SCORE = round(1_000_000 * B / makespan) for a feasible schedule, where B is the
    makespan of the scorer's own DETERMINISTIC list-scheduling baseline (recomputed
    here, independent of the solver). A schedule with a smaller (better) makespan
    scores strictly more; the list-scheduling baseline scores exactly 1_000_000;
    a worse makespan scores less but stays positive. Infeasible -> 0.

The scorer is self-contained and deterministic: it does not trust the solver and
recomputes B itself.
"""
import sys
from collections import deque


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    M = [[0] * m for _ in range(n)]
    P = [[0] * m for _ in range(n)]
    for j in range(n):
        for t in range(m):
            M[j][t] = int(next(it))
            P[j][t] = int(next(it))
    return n, m, M, P


def read_solution(path, n, m):
    """Return order[k] = list of n job indices (a permutation) for each machine k,
    or None if the file is not exactly m permutations of {0,...,n-1}."""
    try:
        with open(path) as f:
            lines = [ln for ln in f.read().splitlines()]
    except OSError:
        return None
    # The contract says exactly m output lines, each with exactly n integer tokens.
    # Blank lines are therefore invalid rows, not ignorable padding.
    if len(lines) != m:
        return None
    rows = [ln.split() for ln in lines]
    if len(rows) != m:
        return None
    order = []
    for parts in rows:
        if len(parts) != n:
            return None
        perm = []
        seen = [False] * n
        for tk in parts:
            try:
                v = int(tk)
            except ValueError:
                return None
            if v < 0 or v >= n or seen[v]:
                return None
            seen[v] = True
            perm.append(v)
        order.append(perm)
    return order


def makespan_from_orders(n, m, M, P, order):
    """Longest-path makespan from the disjunctive graph induced by `order`.

    Nodes = operations (j,t). Returns (makespan, feasible_bool). feasible is False
    iff the induced precedence graph has a cycle (deadlock).
    op id = j*m + t. proc of op id = P[j][t].
    """
    NOP = n * m

    def opid(j, t):
        return j * m + t

    proc = [0] * NOP
    for j in range(n):
        for t in range(m):
            proc[opid(j, t)] = P[j][t]

    # Build successor edges + indegree for topological longest path.
    succ = [[] for _ in range(NOP)]
    indeg = [0] * NOP

    # 1) job-chain precedence: op(j,t) -> op(j,t+1)
    for j in range(n):
        for t in range(m - 1):
            a = opid(j, t)
            b = opid(j, t + 1)
            succ[a].append(b)
            indeg[b] += 1

    # 2) machine-order precedence: for machine k, the chosen order of jobs gives
    #    op(order[k][i]) -> op(order[k][i+1]) where the op of job j on machine k is
    #    the unique t with M[j][t] == k.
    # Precompute, for each job j and machine k, the operation index t.
    op_t_of = [dict() for _ in range(n)]
    for j in range(n):
        for t in range(m):
            op_t_of[j][M[j][t]] = t

    for k in range(m):
        seq = order[k]
        for i in range(len(seq) - 1):
            j1 = seq[i]
            j2 = seq[i + 1]
            t1 = op_t_of[j1].get(k)
            t2 = op_t_of[j2].get(k)
            if t1 is None or t2 is None:
                return (None, False)
            a = opid(j1, t1)
            b = opid(j2, t2)
            succ[a].append(b)
            indeg[b] += 1

    # Topological longest path (Kahn). If we cannot process all nodes -> cycle.
    start = [0] * NOP
    q = deque(i for i in range(NOP) if indeg[i] == 0)
    processed = 0
    makespan = 0
    while q:
        u = q.popleft()
        processed += 1
        fin = start[u] + proc[u]
        if fin > makespan:
            makespan = fin
        for v in succ[u]:
            cand = start[u] + proc[u]
            if cand > start[v]:
                start[v] = cand
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if processed != NOP:
        return (None, False)  # cycle -> infeasible
    return (makespan, True)


def list_scheduling_makespan(n, m, M, P):
    """Deterministic non-delay list-scheduling baseline (reference B).

    Repeatedly: among all jobs whose next un-dispatched operation can be scheduled,
    pick the one that can START earliest (ties broken by the lowest job index), and
    dispatch it on its machine. This yields the active-schedule machine orders, from
    which the makespan is the resulting longest path. Self-contained and independent
    of the solver.
    """
    op_M = M  # M[j][t]
    next_t = [0] * n          # next op index to dispatch for each job
    job_ready = [0] * n       # earliest time job j's next op can start (chain)
    mach_free = [0] * m       # earliest time machine k is free
    remaining = n * m
    INF = float("inf")
    while remaining > 0:
        best_job = -1
        best_start = INF
        for j in range(n):
            t = next_t[j]
            if t >= m:
                continue
            k = op_M[j][t]
            s = job_ready[j] if job_ready[j] > mach_free[k] else mach_free[k]
            if s < best_start or (s == best_start and (best_job < 0 or j < best_job)):
                best_start = s
                best_job = j
        j = best_job
        t = next_t[j]
        k = op_M[j][t]
        s = best_start
        fin = s + P[j][t]
        mach_free[k] = fin
        job_ready[j] = fin
        next_t[j] += 1
        remaining -= 1
    return max(mach_free) if m > 0 else 0


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, m, M, P = read_instance(sys.argv[1])

    order = read_solution(sys.argv[2], n, m)
    if order is None:
        print(0)  # INFEASIBLE -> floored to 0
        return

    mk, feasible = makespan_from_orders(n, m, M, P, order)
    if not feasible or mk is None or mk <= 0:
        print(0)  # cyclic / degenerate -> infeasible
        return

    B = list_scheduling_makespan(n, m, M, P)
    if B <= 0:
        # Degenerate (no ops); any schedule is optimal -> full credit.
        print(1_000_000)
        return

    score = int(round(1_000_000.0 * B / mk))
    print(score)


if __name__ == "__main__":
    main()
