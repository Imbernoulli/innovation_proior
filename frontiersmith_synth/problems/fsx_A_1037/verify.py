import sys

MAX_PAIRS = 2_000_000


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def parse_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); R = int(next(it)); K = int(next(it))
    values = [int(next(it)) for _ in range(N)]
    cap = [0] + [int(next(it)) for _ in range(M)]  # 1-indexed
    return N, M, R, K, values, cap


def scenario_values(N, M, values, task_machines):
    """task_machines: list of sets of machine ids (1-indexed), one set per task index 0..N-1.
    Returns the ascending-sorted list of M*(M-1)/2 surviving-value totals."""
    svals = []
    for a in range(1, M + 1):
        for b in range(a + 1, M + 1):
            tot = 0
            for i in range(N):
                for m in task_machines[i]:
                    if m != a and m != b:
                        tot += values[i]
                        break
            svals.append(tot)
    svals.sort()
    return svals


def baseline_construct(N, M, R, cap):
    """The checker's own weak reference: only bothers spending 40% of the replica budget
    (a shard-value-oblivious triager that gives up early), one replica per covered shard,
    cycling machine index order and skipping full machines."""
    budget = max(1, round(R * 0.4))
    remaining = cap[:]
    task_machines = [set() for _ in range(N)]
    mach_ptr = 1
    for i in range(N):
        if budget <= 0:
            break
        m = mach_ptr
        placed = False
        for _ in range(M):
            if remaining[m] > 0:
                task_machines[i].add(m)
                remaining[m] -= 1
                budget -= 1
                placed = True
                mach_ptr = m % M + 1
                break
            m = m % M + 1
        if not placed:
            mach_ptr = mach_ptr % M + 1
    return task_machines


def main():
    if len(sys.argv) < 3:
        fail("usage")

    try:
        N, M, R, K, values, cap = parse_instance(sys.argv[1])
    except Exception as e:
        fail("bad instance: %s" % e)

    try:
        out_toks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output file")

    try:
        it = iter(out_toks)
        P = int(next(it))
        if P < 0:
            fail("negative P")
        if P > MAX_PAIRS:
            fail("P too large")
        pairs = []
        for _ in range(P):
            t = int(next(it))
            m = int(next(it))
            pairs.append((t, m))
        leftover = list(it)
        if leftover:
            fail("extra tokens after declared P pairs")
    except StopIteration:
        fail("truncated output")
    except ValueError:
        fail("non-integer / non-finite token")

    if P > R:
        fail("P=%d exceeds budget R=%d" % (P, R))

    seen = set()
    counts = [0] * (M + 1)
    task_machines = [set() for _ in range(N)]
    for (t, m) in pairs:
        if not (1 <= t <= N):
            fail("task id out of range: %d" % t)
        if not (1 <= m <= M):
            fail("machine id out of range: %d" % m)
        if (t, m) in seen:
            fail("duplicate replica (%d,%d)" % (t, m))
        seen.add((t, m))
        counts[m] += 1
        task_machines[t - 1].add(m)

    for m in range(1, M + 1):
        if counts[m] > cap[m]:
            fail("machine %d over capacity: %d > %d" % (m, counts[m], cap[m]))

    P2 = M * (M - 1) // 2
    if not (1 <= K <= P2):
        fail("bad K in instance")  # defensive; should never trigger with a correct gen.py

    svals = scenario_values(N, M, values, task_machines)
    F = svals[K - 1]

    base_tm = baseline_construct(N, M, R, cap)
    bvals = scenario_values(N, M, values, base_tm)
    B = max(1e-9, float(bvals[K - 1]))

    sc = min(1000.0, 100.0 * F / B)
    print("F=%d B=%.4f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
