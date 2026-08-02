import sys

# Format-D style checker for family 'vliw-bundle-pack'.
#
#   1) Parse the dependence DAG (op types / latencies / predecessors), the bundle
#      slot template, the register-file size R and the per-spill cost from <in>.
#   2) Parse the participant's SCHEDULE from <out>: for every op i, an issue cycle
#      c_i and a slot index s_i.  (The checker -- not the solver -- derives the
#      unique cost-MINIMAL register allocation implied by that schedule; the
#      solver's only lever is WHEN/WHERE to issue each op.)
#   3) Feasibility gate (exact, integer only): slot-type match, dependency+latency
#      ordering, no two ops sharing a (cycle,slot) bundle slot. Any violation, or
#      any non-finite/garbage token, scores 0.
#   4) Objective (minimize) = total_cycles + spill_count * SPILL_COST, where
#      spill_count is the PROVABLY MINIMAL number of live values that cannot be
#      held in R registers simultaneously, computed by an exact sweep over the
#      live-range interval graph induced by the schedule (see min_spills: this is
#      the classical optimal "keep the earliest-dying value, evict the
#      furthest-dying one" interval-capacity greedy, verified against brute force).
#   5) Baseline B = the checker's own serial (one op per cycle, in DAG order)
#      construction, scored the same way.  Ratio = min(1, 0.1 * B / F).

MAX_CYCLE = 2_000_000


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def parse_input(text):
    it = iter(text.split())

    def nxt():
        return next(it)

    try:
        N = int(nxt()); W = int(nxt()); R = int(nxt()); spill_cost = int(nxt())
    except Exception:
        fail("bad header")
    if not (1 <= N <= 2000 and 1 <= W <= 64 and 1 <= R <= 64 and 0 <= spill_cost <= 10 ** 6):
        fail("header out of range")

    try:
        slot_types = nxt()
    except Exception:
        fail("missing slot template")
    if len(slot_types) != W:
        fail("slot template length mismatch")

    ops = [None]  # 1-indexed
    consumers = [[] for _ in range(N + 1)]
    try:
        for i in range(1, N + 1):
            typ = nxt()
            lat = int(nxt())
            k = int(nxt())
            preds = [int(nxt()) for _ in range(k)]
            if typ not in slot_types:
                fail("op %d requires slot type never present in template" % i)
            if lat < 1:
                fail("op %d has non-positive latency" % i)
            for p in preds:
                if not (1 <= p < i):
                    fail("op %d has a bad predecessor index" % i)
                consumers[p].append(i)
            ops.append((typ, lat, preds))
    except StopIteration:
        fail("truncated op list")
    except ValueError:
        fail("non-integer op field")

    return N, W, R, spill_cost, slot_types, ops, consumers


def min_spills(intervals, R):
    """Exact minimum number of intervals that must be evicted so that no point is
    covered by more than R kept intervals. Sweep by start time; when a new
    interval would exceed capacity, evict whichever of {active, new} has the
    LARGEST die time (furthest-in-future eviction over interval lifetimes -- the
    interval-graph analogue of Belady's optimal offline cache rule). Verified
    against brute force on thousands of random small instances."""
    order = sorted(range(len(intervals)), key=lambda i: (intervals[i][0], intervals[i][1]))
    active = []  # list of (die, idx) currently "kept in a register"
    spilled = 0
    for idx in order:
        s, d = intervals[idx]
        active = [(dd, ii) for (dd, ii) in active if dd >= s]
        if len(active) < R:
            active.append((d, idx))
        else:
            mx_die, mx_idx = max(active + [(d, idx)])
            if mx_idx == idx:
                spilled += 1
            else:
                active = [(dd, ii) for (dd, ii) in active if ii != mx_idx]
                spilled += 1
                active.append((d, idx))
    return spilled


def evaluate(N, R, spill_cost, ops, consumers, sched):
    """sched: dict i -> (cycle, slot). Returns total scalar cost (lower better)."""
    total_cycles = max(c for (c, s) in sched.values())
    intervals = []
    for i in range(1, N + 1):
        ci = sched[i][0]
        cons = consumers[i]
        die = max(sched[j][0] for j in cons) if cons else ci
        intervals.append((ci, die))
    spills = min_spills(intervals, R)
    return total_cycles + spills * spill_cost, spills, total_cycles


def baseline_schedule(N, W, slot_types, ops):
    """Trivial feasible construction: one op issued per cycle, in DAG (index)
    order, each in the first slot matching its type. No bundle-packing at all."""
    type_slot = {}
    for si, t in enumerate(slot_types):
        type_slot.setdefault(t, si)
    sched = {}
    cyc = [0] * (N + 1)
    last_used = 0
    for i in range(1, N + 1):
        typ, lat, preds = ops[i]
        bound = 1
        for p in preds:
            bound = max(bound, cyc[p] + ops[p][1])
        c = max(bound, last_used + 1)
        cyc[i] = c
        last_used = c
        sched[i] = (c, type_slot[typ])
    return sched


def main():
    in_text = open(sys.argv[1]).read()
    out_text = open(sys.argv[2]).read()

    N, W, R, spill_cost, slot_types, ops, consumers = parse_input(in_text)

    toks = out_text.split()
    if len(toks) != 2 * N:
        fail("wrong token count (got %d, need %d)" % (len(toks), 2 * N))
    try:
        vals = [int(t) for t in toks]
    except ValueError:
        fail("non-integer / non-finite schedule entry")

    sched = {}
    used = set()
    for i in range(1, N + 1):
        c = vals[2 * (i - 1)]
        s = vals[2 * (i - 1) + 1]
        if not (1 <= c <= MAX_CYCLE):
            fail("op %d cycle out of range" % i)
        if not (0 <= s < W):
            fail("op %d slot out of range" % i)
        typ = ops[i][0]
        if slot_types[s] != typ:
            fail("op %d placed in a slot of the wrong type" % i)
        if (c, s) in used:
            fail("two ops share bundle slot (cycle=%d, slot=%d)" % (c, s))
        used.add((c, s))
        sched[i] = (c, s)

    for i in range(1, N + 1):
        typ, lat, preds = ops[i]
        ci = sched[i][0]
        for p in preds:
            cp, sp = sched[p]
            latp = ops[p][1]
            if ci < cp + latp:
                fail("op %d issued before predecessor %d's latency elapses" % (i, p))

    F, spills_f, cyc_f = evaluate(N, R, spill_cost, ops, consumers, sched)

    base_sched = baseline_schedule(N, W, slot_types, ops)
    B, spills_b, cyc_b = evaluate(N, R, spill_cost, ops, consumers, base_sched)

    ratio = min(1.0, 0.1 * B / max(1e-9, F))
    print("F=%d (cycles=%d spills=%d) B=%d (cycles=%d spills=%d) Ratio: %.6f"
          % (F, cyc_f, spills_f, B, cyc_b, spills_b, ratio))


if __name__ == "__main__":
    main()
