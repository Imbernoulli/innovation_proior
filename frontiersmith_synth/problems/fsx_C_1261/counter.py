import sys

# counter.py <in> <out> <ans> -- deterministic scorer for the DVFS
# deadline-schedule problem. Verifies the submitted per-slot level
# sequence is feasible (every job's work completed inside its
# [release, deadline) window, given ramp-reduced capacity right after
# every level switch), then scores total energy (active power + switch
# taxes) against the checker's own "race to idle" reference plan.

M = 4
IDLE, LOW, MID, MAX = 0, 1, 2, 3


def fail(msg):
    print("INFEASIBLE: %s Ratio: 0.0" % msg)
    sys.exit(0)


def read_instance(path):
    toks = open(path).read().split()
    pos = [0]

    def nxt():
        v = toks[pos[0]]
        pos[0] += 1
        return v

    T = int(nxt()); m = int(nxt()); J = int(nxt())
    assert m == M, "instance/library mismatch"
    s = []; pw = []
    for _ in range(m):
        s.append(int(nxt())); pw.append(int(nxt()))
    ramp = int(nxt())
    trans = [[int(nxt()) for _ in range(m)] for _ in range(m)]
    jobs = []
    for _ in range(J):
        r = int(nxt()); d = int(nxt()); w = int(nxt())
        jobs.append((r, d, w))
    return T, m, J, s, pw, ramp, trans, jobs


def parse_int_token(tok):
    if tok is None:
        return None
    s = tok
    neg = False
    if s.startswith('+'):
        s = s[1:]
    elif s.startswith('-'):
        neg = True
        s = s[1:]
    if s == '' or not s.isdigit():
        return None
    v = int(s)
    return -v if neg else v


def simulate(levels, T, s, ramp, jobs):
    """Returns (feasible, cap[]) -- cap[] = per-slot usable cycles after
    ramp loss on transitions; feasible checked via preemptive EDF packing
    (earliest-deadline-first is optimal for feasibility under a
    time-varying, but here output-determined, capacity sequence)."""
    import heapq
    cap = [0] * T
    prev = None
    for t in range(T):
        lvl = levels[t]
        c = s[lvl]
        if prev is not None and lvl != prev:
            c = max(0, c - ramp)
        cap[t] = c
        prev = lvl
    by_release = {}
    by_deadline = {}
    for idx, (r, d, w) in enumerate(jobs):
        by_release.setdefault(r, []).append(idx)
        by_deadline.setdefault(d, []).append(idx)
    remaining = [w for (r, d, w) in jobs]
    heap = []
    feasible = True
    reason = ""
    for t in range(T):
        for idx in by_release.get(t, []):
            heapq.heappush(heap, (jobs[idx][1], idx))
        avail = cap[t]
        tmp = []
        while avail > 0 and heap:
            dl, idx = heapq.heappop(heap)
            if remaining[idx] <= 0:
                continue
            use = min(avail, remaining[idx])
            remaining[idx] -= use
            avail -= use
            if remaining[idx] > 0:
                tmp.append((dl, idx))
        for item in tmp:
            heapq.heappush(heap, item)
        for idx in by_deadline.get(t + 1, []):
            if remaining[idx] > 0:
                feasible = False
                reason = "job %d missed deadline %d with %d work left" % (idx, jobs[idx][1], remaining[idx])
    return feasible, reason


def energy(levels, T, pw, trans):
    E = sum(pw[levels[t]] for t in range(T))
    for t in range(1, T):
        E += trans[levels[t - 1]][levels[t]]
    return E


def race_to_idle_baseline(T, s, pw, ramp, trans, jobs):
    """Checker's own always-feasible reference plan: run at the max level
    for the shortest prefix that still meets every deadline, then idle for
    the rest. Always constructible because gen.py guarantees constant-max
    for the FULL horizon is feasible (P = T is a fallback)."""
    for P in range(0, T + 1):
        levels = [MAX] * P + [IDLE] * (T - P)
        feas, _ = simulate(levels, T, s, ramp, jobs)
        if feas:
            return energy(levels, T, pw, trans)
    # unreachable given gen.py's guarantee, but keep a safe fallback
    levels = [MAX] * T
    return energy(levels, T, pw, trans)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    T, m, J, s, pw, ramp, trans, jobs = read_instance(in_path)

    raw = open(out_path).read().split()
    if len(raw) != T:
        fail("expected %d level tokens, got %d" % (T, len(raw)))

    levels = []
    for i, tok in enumerate(raw):
        v = parse_int_token(tok)
        if v is None:
            fail("non-integer / non-finite token at slot %d: %r" % (i, tok))
        if v < 0 or v >= m:
            fail("level %d at slot %d out of range [0,%d]" % (v, i, m - 1))
        levels.append(v)

    feas, reason = simulate(levels, T, s, ramp, jobs)
    if not feas:
        fail(reason)

    F = energy(levels, T, pw, trans)
    B = race_to_idle_baseline(T, s, pw, ramp, trans, jobs)
    B = max(B, 1)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("OK energy=%d baseline=%d T=%d J=%d Ratio: %.6f" % (F, B, T, J, sc / 1000.0))


if __name__ == "__main__":
    main()
