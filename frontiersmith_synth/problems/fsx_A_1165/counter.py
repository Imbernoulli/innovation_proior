import sys

MAX_TOKEN_LEN = 12
SEAL_FIXED = 20
SEAL_PER_RECORD = 1


def fail(reason):
    print("Ratio: 0.0  # %s" % reason)
    sys.exit(0)


def parse_int(tok, lo=None, hi=None):
    if len(tok) == 0 or len(tok) > MAX_TOKEN_LEN:
        return None
    t = tok
    neg = False
    if t[0] in "+-":
        neg = t[0] == "-"
        t = t[1:]
    if len(t) == 0 or not t.isdigit():
        return None
    try:
        v = -int(t) if neg else int(t)
    except Exception:
        return None
    if lo is not None and v < lo:
        return None
    if hi is not None and v > hi:
        return None
    return v


def replay_cost(pages_sorted_by_tick, crashes):
    """pages_sorted_by_tick: list of (tick, [ids in scan order]), strictly
    increasing tick. crashes: list of (c_tick, needed_ids_list).
    Returns (total_cost, ok) -- ok False only on an internal invariant
    failure (should not happen for a feasible submission given the
    generator's guarantee that every needed id's deadline < its crash tick)."""
    ticks = [p[0] for p in pages_sorted_by_tick]
    total = 0
    for c_tick, needed in crashes:
        # prefix of pages with tick <= c_tick
        # binary search for rightmost index with tick <= c_tick
        lo, hi = 0, len(ticks)
        while lo < hi:
            mid = (lo + hi) // 2
            if ticks[mid] <= c_tick:
                lo = mid + 1
            else:
                hi = mid
        prefix_len = lo  # pages[0:prefix_len] have tick <= c_tick
        remaining = set(needed)
        idx = prefix_len - 1
        cost = 0
        while idx >= 0 and remaining:
            _, order = pages_sorted_by_tick[idx]
            for rec in order:
                cost += 1
                if rec in remaining:
                    remaining.discard(rec)
                    if not remaining:
                        break
            idx -= 1
        if remaining:
            return total, False
        total += cost
    return total, True


def main():
    if len(sys.argv) < 3:
        fail("bad_args")
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        with open(in_path, "r") as f:
            in_lines = f.read().split("\n")
        N, C, TMAX = (int(x) for x in in_lines[0].split())
        if N < 1 or C < 1 or TMAX < N:
            fail("bad_input_header")
        A = [0] * (N + 1)
        Dslack = [0] * (N + 1)
        for i in range(1, N + 1):
            a_i, d_i = in_lines[i].split()
            A[i] = int(a_i)
            Dslack[i] = int(d_i)
        deadline = [0] * (N + 1)
        for i in range(1, N + 1):
            deadline[i] = A[i] + Dslack[i]
        crashes = []
        ptr = N + 1
        for j in range(C):
            c_tick, m = (int(x) for x in in_lines[ptr].split())
            needed = [int(x) for x in in_lines[ptr + 1].split()]
            if len(needed) != m:
                fail("bad_input_crash")
            crashes.append((c_tick, needed))
            ptr += 2
    except Exception:
        fail("unparseable_input")

    try:
        with open(out_path, "r") as f:
            out_text = f.read()
    except Exception:
        fail("no_output")

    raw_lines = [ln for ln in out_text.split("\n") if ln.strip() != ""]
    if len(raw_lines) == 0:
        fail("no_seals")
    if len(raw_lines) > N:
        fail("too_many_seal_lines")

    pages = []
    sealed_of = {}  # id -> seal tick
    used = set()
    prev_tick = -1
    total_seal_cost = 0
    for ln in raw_lines:
        toks = ln.split()
        if len(toks) < 2:
            fail("bad_seal_line")
        tick = parse_int(toks[0], lo=1, hi=10 ** 9)
        k = parse_int(toks[1], lo=1, hi=N)
        if tick is None or k is None:
            fail("unparseable_tick_or_k")
        if tick > TMAX:
            fail("tick_exceeds_tmax")
        if tick <= prev_tick:
            fail("ticks_not_strictly_increasing")
        prev_tick = tick
        if len(toks) != 2 + k:
            fail("token_count_mismatch")
        order = []
        seen_here = set()
        for t in toks[2:]:
            rid = parse_int(t, lo=1, hi=N)
            if rid is None:
                fail("unparseable_id")
            if rid in seen_here:
                fail("duplicate_id_within_seal")
            seen_here.add(rid)
            if rid in used:
                fail("id_sealed_twice")
            used.add(rid)
            if A[rid] > tick:
                fail("record_not_yet_arrived")
            if tick > deadline[rid]:
                fail("deadline_violated")
            sealed_of[rid] = tick
            order.append(rid)
        pages.append((tick, order))
        total_seal_cost += SEAL_FIXED + SEAL_PER_RECORD * k

    if len(used) != N:
        fail("not_all_ids_sealed")

    replay_total, rok = replay_cost(pages, crashes)
    if not rok:
        fail("internal_replay_invariant_broken")

    F = total_seal_cost + replay_total
    if F <= 0:
        fail("nonpositive_cost")

    # ---- internal baseline B: seal every record alone, the instant it
    # arrives (tick = a_i). Always feasible (a_i <= a_i <= deadline_i since
    # d_i >= 0). Safe but wasteful: N seals at fixed overhead each.
    base_pages = [(A[i], [i]) for i in range(1, N + 1)]
    base_seal_cost = N * (SEAL_FIXED + SEAL_PER_RECORD)
    base_replay, brok = replay_cost(base_pages, crashes)
    if not brok:
        fail("internal_baseline_invariant_broken")
    B = base_seal_cost + base_replay
    if B <= 0:
        fail("nonpositive_baseline")

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("seals=%d seal_cost=%d replay_cost=%d F=%d B=%d Ratio: %.6f" %
          (len(pages), total_seal_cost, replay_total, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
