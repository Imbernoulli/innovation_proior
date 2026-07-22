import sys

# Format D checker -- pit-lane arm scheduling (exact-tick makespan under a precedence DAG
# and a per-sector mutex).
#
#   1) Parse the instance (N tasks, K arms, durations d[], sectors s[], precedence edges).
#   2) Parse the participant's straight-line plan: for every task i (1..N, in input order)
#      an (arm, start_tick) pair.
#   3) Feasibility gate (ANY violation -> Ratio: 0.0):
#        - well-formed integers, arm in [0,K), start >= 0 and bounded, finite
#        - precedence: start[v] >= start[u] + d[u] for every edge (u,v)
#        - arm capacity: an arm processes at most one task at a time (no overlap)
#        - sector mutex: at most one task of a given sector is "in progress" at a time,
#          regardless of which arm runs it
#   4) Objective (minimize) = makespan F = max_i (start_i + d_i)  [exact tick count].
#      Baseline B = the checker's own trivial construction: run every task strictly
#      sequentially, one at a time, in input (topological) order on a single arm -- always
#      feasible, B = sum(d_i).  Ratio = min(1, B / (10*F)).

MAXSTART = 10 ** 7


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    inp = open(sys.argv[1]).read().split()
    out_raw = open(sys.argv[2]).read().split()

    it = iter(inp)
    try:
        N = int(next(it)); K = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= N <= 100000 and 1 <= K <= 64):
        fail("bad N/K")

    try:
        d = [int(next(it)) for _ in range(N)]
        s = [int(next(it)) for _ in range(N)]
        E = int(next(it))
        edges = []
        for _ in range(E):
            u = int(next(it)); v = int(next(it))
            edges.append((u, v))
    except Exception:
        fail("bad instance")

    if any(x < 1 for x in d):
        fail("bad instance durations")
    if any(x < 0 or x > 5 for x in s):
        fail("bad instance sectors")
    for (u, v) in edges:
        if not (1 <= u < v <= N):
            fail("bad instance edge")

    NSEC = 6

    # ---- parse participant output: N pairs (arm, start), in task-index order ----
    need = 2 * N
    if len(out_raw) != need:
        fail("wrong token count (got %d, need %d)" % (len(out_raw), need))
    try:
        toks = [int(t) for t in out_raw]
    except Exception:
        fail("non-integer / non-finite token")

    arm = [0] * (N + 1)
    start = [0] * (N + 1)
    for i in range(1, N + 1):
        a = toks[2 * (i - 1)]
        t0 = toks[2 * (i - 1) + 1]
        if not (0 <= a < K):
            fail("arm index out of range at task %d" % i)
        if not (0 <= t0 <= MAXSTART):
            fail("start out of range at task %d" % i)
        arm[i] = a
        start[i] = t0

    # ---- precedence ----
    for (u, v) in edges:
        if start[v] < start[u] + d[u - 1]:
            fail("precedence violated on edge (%d,%d)" % (u, v))

    # ---- arm capacity: no two tasks on the same arm overlap ----
    by_arm = {}
    for i in range(1, N + 1):
        by_arm.setdefault(arm[i], []).append(i)
    for a, tasks in by_arm.items():
        tasks.sort(key=lambda i: (start[i], i))
        for j in range(1, len(tasks)):
            p, c = tasks[j - 1], tasks[j]
            if start[c] < start[p] + d[p - 1]:
                fail("arm %d double-booked (tasks %d,%d)" % (a, p, c))

    # ---- sector mutex: no two tasks in the same sector overlap, any arm ----
    by_sec = {}
    for i in range(1, N + 1):
        by_sec.setdefault(s[i - 1], []).append(i)
    for sec, tasks in by_sec.items():
        tasks.sort(key=lambda i: (start[i], i))
        for j in range(1, len(tasks)):
            p, c = tasks[j - 1], tasks[j]
            if start[c] < start[p] + d[p - 1]:
                fail("sector %d mutex violated (tasks %d,%d)" % (sec, p, c))

    F = max(start[i] + d[i - 1] for i in range(1, N + 1))
    if F <= 0:
        fail("degenerate makespan")

    B = sum(d)  # checker's own fully-sequential single-arm reference construction

    ratio = min(1.0, B / (10.0 * F))
    print("F=%d B=%d Ratio: %.6f" % (F, B, ratio))


if __name__ == "__main__":
    main()
