#!/usr/bin/env python3
"""Deterministic local scorer for "Job-Shop Scheduling (makespan)".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single number: the score (an integer). HIGHER is better.

Scoring rule (see context.md "Evaluation settings"):

  * INSTANCE (stdin of the solver):
        n m
        then n job rows, each with m pairs "machine duration".
    Operation (j, k) is the k-th operation of job j; it runs on machine
    instance[j][k].machine for instance[j][k].dur time units, and may start only
    after operation (j, k-1) has finished. A machine runs one operation at a time.

  * SOLUTION (stdout of the solver): a *machine order*. Exactly m lines; line i is
    the order in which machine i processes its operations, given as a permutation
    of the job indices whose operation on machine i is listed. Because every job
    visits every machine exactly once, a job index on machine i's line uniquely
    identifies operation (j, k) with instance[j][k].machine == i.

  * FEASIBILITY (any violation -> score 0):
      - the file parses as exactly m lines (ignoring blank lines), line i a
        permutation of exactly the set of jobs that use machine i (here: all n
        jobs, since every job visits every machine once);
      - the implied schedule has no cyclic deadlock: building start times by the
        longest path over (a) job-precedence arcs (op k-1 -> op k of the same job)
        and (b) machine-order arcs (the given consecutive order on each machine)
        must succeed -- i.e. the combined precedence digraph is acyclic.
    A schedule that decodes (acyclic) is automatically non-overlapping on every
    machine and respects every job precedence, because start times are set as the
    longest path; we additionally re-verify machine-exclusivity and precedence
    explicitly as a guard. Any failure floors the score to 0.

  * MAKESPAN (lower is better) of a feasible schedule: the maximum completion time
    over all operations, where completion(j,k) = start(j,k) + dur(j,k) and start is
    the longest-path value in the disjunctive graph implied by the machine order.

  * SCORE (higher better), normalized against a deterministic Shortest-Processing-
    Time (SPT) list-scheduling baseline the scorer recomputes itself:
        score = round(1_000_000 * baseline_makespan / max(1, solver_makespan))
    The SPT baseline scores ~1_000_000; a smaller (better) makespan scores more.
    INFEASIBLE -> 0.
"""
import sys


# ----------------------------------------------------------------------------- IO
def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    # ops[j] = list of (machine, dur) for k = 0..m-1
    ops = []
    for _j in range(n):
        row = []
        for _k in range(m):
            mc = int(next(it))
            dr = int(next(it))
            row.append((mc, dr))
        ops.append(row)
    return n, m, ops


def read_solution(path, n, m, ops):
    """Parse the machine order. Return mach_order: list (len m) of job-index lists,
    or None if the structure is malformed (wrong shape / not a permutation)."""
    try:
        with open(path) as f:
            lines = [ln for ln in f.read().splitlines()]
    except OSError:
        return None
    # Drop blank lines, keep order.
    rows = []
    for ln in lines:
        s = ln.split()
        if s:
            rows.append(s)
    if len(rows) != m:
        return None

    # Which jobs use machine i? (every job uses every machine exactly once)
    jobs_on = [set() for _ in range(m)]
    for j in range(n):
        for k in range(m):
            mc = ops[j][k][0]
            jobs_on[mc].add(j)

    mach_order = []
    for i in range(m):
        try:
            seq = [int(t) for t in rows[i]]
        except ValueError:
            return None
        if set(seq) != jobs_on[i] or len(seq) != len(jobs_on[i]):
            return None
        mach_order.append(seq)
    return mach_order


# --------------------------------------------------------------- decode (long path)
def decode_makespan(n, m, ops, mach_order):
    """Compute start times via longest path in the disjunctive graph implied by
    `mach_order`. Returns (feasible, makespan).

    Nodes: operations (j, k). Arcs:
      * job precedence: (j, k-1) -> (j, k), weight dur(j, k-1);
      * machine order: for consecutive a, b on a machine, a -> b, weight dur(a).
    start(node) = max over incoming arcs of start(pred) + weight. We compute it by
    a Kahn topological order; if not all nodes get processed, there is a cycle
    (deadlock) -> infeasible.
    """
    # index operations: op_id = j*m + k
    # Build, for each job j, the position k of machine mc: pos_of[j][mc] = k
    pos_of = [dict() for _ in range(n)]
    for j in range(n):
        for k in range(m):
            mc = ops[j][k][0]
            pos_of[j][mc] = k

    N = n * m
    indeg = [0] * N
    adj = [[] for _ in range(N)]  # list of (dst, weight)

    def oid(j, k):
        return j * m + k

    # job precedence arcs
    for j in range(n):
        for k in range(1, m):
            u = oid(j, k - 1)
            v = oid(j, k)
            adj[u].append((v, ops[j][k - 1][1]))
            indeg[v] += 1
    # machine order arcs
    for i in range(m):
        seq = mach_order[i]
        for t in range(1, len(seq)):
            ja = seq[t - 1]
            jb = seq[t]
            ka = pos_of[ja][i]
            kb = pos_of[jb][i]
            u = oid(ja, ka)
            v = oid(jb, kb)
            adj[u].append((v, ops[ja][ka][1]))
            indeg[v] += 1

    # Kahn longest path
    start = [0] * N
    stack = [u for u in range(N) if indeg[u] == 0]
    seen = 0
    head = 0
    while head < len(stack):
        u = stack[head]
        head += 1
        seen += 1
        su = start[u]
        for (v, w) in adj[u]:
            if su + w > start[v]:
                start[v] = su + w
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    if seen != N:
        return False, None  # cycle -> deadlock -> infeasible

    makespan = 0
    for j in range(n):
        for k in range(m):
            c = start[oid(j, k)] + ops[j][k][1]
            if c > makespan:
                makespan = c

    # ---- explicit guards (defensive; longest-path already enforces these) ----
    # job precedence: start(j,k) >= start(j,k-1) + dur(j,k-1)
    for j in range(n):
        for k in range(1, m):
            if start[oid(j, k)] < start[oid(j, k - 1)] + ops[j][k - 1][1]:
                return False, None
    # machine exclusivity: intervals on a machine, in given order, are disjoint
    for i in range(m):
        seq = mach_order[i]
        prev_end = -1
        for j in seq:
            k = pos_of[j][i]
            s = start[oid(j, k)]
            if s < prev_end:
                return False, None
            prev_end = s + ops[j][k][1]
    return True, makespan


# -------------------------------------------------------------- baseline: SPT list
def baseline_makespan(n, m, ops):
    """Deterministic Shortest-Processing-Time (SPT) list-scheduling (non-delay).

    Simulate: each job has a 'next operation pointer' next_k; each machine has a
    'free at' time; each job has a 'ready at' time (when its previous op finished).
    Repeatedly, among all operations that are currently the next op of their job,
    schedule the one we can start earliest; ties are broken by the SPT rule
    (shortest duration), then by job index. We schedule it at
    max(machine_free, job_ready) and advance. This always produces a valid
    schedule and a deterministic makespan we normalize against.
    """
    pos_dur = ops  # ops[j][k] = (machine, dur)
    next_k = [0] * n
    job_ready = [0] * n
    mach_free = [0] * m
    remaining = n * m
    makespan = 0
    while remaining > 0:
        best = None  # (start_time, dur, job)
        for j in range(n):
            k = next_k[j]
            if k >= m:
                continue
            mc, dr = pos_dur[j][k]
            st = max(job_ready[j], mach_free[mc])
            key = (st, dr, j)
            if best is None or key < best:
                best = key
        st, dr, j = best
        k = next_k[j]
        mc, dr2 = pos_dur[j][k]
        end = st + dr2
        mach_free[mc] = end
        job_ready[j] = end
        next_k[j] = k + 1
        remaining -= 1
        if end > makespan:
            makespan = end
    return makespan


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, m, ops = read_instance(sys.argv[1])

    mach_order = read_solution(sys.argv[2], n, m, ops)
    if mach_order is None:
        print(0)  # INFEASIBLE -> floored to 0
        return

    ok, mk = decode_makespan(n, m, ops, mach_order)
    if not ok:
        print(0)
        return

    base = baseline_makespan(n, m, ops)
    score = int(round(1_000_000.0 * base / max(1, mk)))
    print(score)


if __name__ == "__main__":
    main()
