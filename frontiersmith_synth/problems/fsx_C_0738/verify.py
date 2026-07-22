import sys, math


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")
    try:
        out_toks = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    try:
        it = iter(inp)
        N = int(next(it)); K = int(next(it)); L = int(next(it))
        W_same = int(next(it)); W_diff = int(next(it)); s0 = int(next(it))
        if N <= 0 or K <= 0 or L <= 0:
            fail("bad header")
        if W_diff < 0 or W_same <= W_diff:
            fail("bad costs")
        if s0 not in (0, 1):
            fail("bad s0")
        dirs = [0] * N; arr = [0] * N; dl = [0] * N; wt = [0] * N
        for i in range(N):
            d = int(next(it)); a = int(next(it)); dd = int(next(it)); w = int(next(it))
            if d not in (0, 1) or a < 0 or dd < a or w <= 0:
                fail("bad boat record")
            dirs[i] = d; arr[i] = a; dl[i] = dd; wt[i] = w
    except Exception:
        fail("bad input")

    def cost_of_schedule(lockages):
        # lockages: list of (t, dirn, [0-indexed boat ids])
        total = 0.0
        prev_dir = s0
        for (t, dirn, ids) in lockages:
            total += float(W_same if dirn == prev_dir else W_diff)
            prev_dir = dirn
            for bid in ids:
                if t > dl[bid]:
                    total += wt[bid] * (t - dl[bid])
        return total

    # ---- internal baseline B: batch each direction to capacity K
    # (earliest-deadline-first within a direction, so it is not
    # gratuitously late), but run ALL of one direction's lockages before
    # ever switching to the other -- deadline-aware and capacity-aware, but
    # blind to the water-parity bit. This is the worst case for the
    # same-direction "run" structure (every lockage but the block's first
    # repeats the previous direction), giving a stable, size-independent
    # disadvantage relative to any plan that spreads the switches out. ----
    d0 = sorted([i for i in range(N) if dirs[i] == 0], key=lambda i: (dl[i], arr[i]))
    d1 = sorted([i for i in range(N) if dirs[i] == 1], key=lambda i: (dl[i], arr[i]))
    first_dir = 0
    if d1 and (not d0 or dl[d1[0]] < dl[d0[0]]):
        first_dir = 1
    pools = {0: d0, 1: d1}
    baseline_lockages = []
    t_prev = None
    for bd in (first_dir, 1 - first_dir):
        pool = pools[bd]
        for start in range(0, len(pool), K):
            ids = pool[start:start + K]
            if not ids:
                continue
            max_a = max(arr[i] for i in ids)
            t = max_a if t_prev is None else max(t_prev + L, max_a)
            baseline_lockages.append((t, bd, ids))
            t_prev = t
    B = cost_of_schedule(baseline_lockages)
    B = max(B, 1e-6)

    # ---- parse participant output ----
    try:
        oit = iter(out_toks)
        M = int(next(oit))
    except Exception:
        fail("cannot parse M")
    if M < 1 or M > 3 * N:
        fail("M out of range")

    lockages = []
    covered = [False] * N
    try:
        for _ in range(M):
            t = float(next(oit))
            if not math.isfinite(t) or t < 0:
                fail("bad lockage time")
            dirn = int(next(oit))
            if dirn not in (0, 1):
                fail("bad lockage direction")
            k = int(next(oit))
            if k < 1 or k > K:
                fail("bad batch size")
            ids0 = []
            for _ in range(k):
                bid1 = int(next(oit))
                if bid1 < 1 or bid1 > N:
                    fail("boat id out of range")
                bid0 = bid1 - 1
                if covered[bid0]:
                    fail("boat listed twice")
                covered[bid0] = True
                if dirs[bid0] != dirn:
                    fail("direction mismatch")
                if t < arr[bid0]:
                    fail("lockage starts before a rider arrived")
                ids0.append(bid0)
            lockages.append((t, dirn, ids0))
    except SystemExit:
        raise
    except Exception:
        fail("malformed lockage record")

    # no leftover garbage tokens
    remaining = list(oit)
    if remaining:
        fail("trailing garbage tokens")

    if not all(covered):
        fail("not every boat was scheduled")

    prev_t = None
    for (t, dirn, ids0) in lockages:
        if prev_t is not None:
            if t + 1e-9 < prev_t:
                fail("lockage times not non-decreasing")
            if t + 1e-9 < prev_t + L:
                fail("lockages too close together (chamber not reset)")
        prev_t = t

    F = cost_of_schedule(lockages)
    F = max(F, 0.0)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.4f B=%.4f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
