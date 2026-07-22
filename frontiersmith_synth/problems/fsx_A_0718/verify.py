import sys, math

MAX_LINES = 20000

def fail(msg):
    print("INVALID: %s Ratio: 0.0" % msg)
    sys.exit(0)

def read_input(path):
    with open(path) as f:
        toks = f.read().split()
    pos = 0
    def nxt():
        nonlocal pos
        v = toks[pos]; pos += 1
        return v
    S = int(nxt()); K = int(nxt()); P = int(nxt()); R = int(nxt()); T = int(nxt())
    sats = []
    for _ in range(S):
        acc = int(nxt()); cap = int(nxt()); ph = int(nxt())
        sats.append((acc, cap, ph))
    stations = []
    for _ in range(K):
        drain = int(nxt()); off = int(nxt()); dur = int(nxt())
        stations.append((drain, off, dur))
    return S, K, P, R, T, sats, stations

def window_bounds(sats, stations, i, k, c):
    acc, cap, ph = sats[i]
    drain, off, dur = stations[k]
    base = ph + off
    start = c * P_GLOBAL + base
    end = start + dur
    return start, end

def simulate(S, K, P, R, T, sats, stations, sched):
    """sched: list of (i,k) -> sorted list of (start,end) intervals, already validated
    to be within visibility windows and non-overlapping per satellite."""
    # build per-tick station active-sender lists via sweep (O(T + intervals))
    events = []  # (tick, +1/-1, sat, station)
    for (i, k), ivs in sched.items():
        for (st, en) in ivs:
            events.append((st, 1, i, k))
            events.append((en, -1, i, k))
    events.sort(key=lambda e: (e[0], e[1]))

    buffers = [0] * S
    delivered_total = 0
    active = {}  # station -> set of sats currently active
    for k in range(K):
        active[k] = set()

    ev_idx = 0
    n_ev = len(events)
    accs = [s[0] for s in sats]
    caps = [s[1] for s in sats]
    drains = [st[0] for st in stations]

    for t in range(T):
        # accrual with cap (overflow lost)
        for i in range(S):
            b = buffers[i] + accs[i]
            if b > caps[i]:
                b = caps[i]
            buffers[i] = b
        # apply events at this tick (start events add, end events remove -- end is exclusive)
        while ev_idx < n_ev and events[ev_idx][0] == t:
            _, sign, i, k = events[ev_idx]
            if sign == 1:
                active[k].add(i)
            else:
                active[k].discard(i)
            ev_idx += 1
        # drain per station
        for k in range(K):
            senders = active[k]
            if not senders:
                continue
            n = len(senders)
            if n == 1:
                rate = drains[k]
            else:
                # co-channel interference: the station's channel collapses to a
                # quarter of nominal capacity when shared, split evenly among the
                # simultaneous senders (floor division; can floor to 0).
                rate = (drains[k] // 4) // n
            if rate <= 0:
                continue
            for i in senders:
                d = buffers[i] if buffers[i] < rate else rate
                buffers[i] -= d
                delivered_total += d
    return delivered_total

def main():
    global P_GLOBAL
    inf, outf = sys.argv[1], sys.argv[2]
    S, K, P, R, T, sats, stations = read_input(inf)
    P_GLOBAL = P

    # precompute per-(i,k) window list: [(start,end)] one per cycle, cycle-major
    win = {}
    for i in range(S):
        acc, cap, ph = sats[i]
        for k in range(K):
            drain, off, dur = stations[k]
            base = ph + off
            if base < 0 or base + dur > P:
                fail("internal window construction error")
            win[(i, k)] = (base, dur)  # local start within a cycle, duration

    # ---- parse participant output ----
    try:
        with open(outf) as f:
            raw = f.read()
    except Exception:
        fail("cannot read output")

    toks = raw.split()
    if len(toks) == 0:
        fail("empty output")
    try:
        m = int(toks[0])
    except Exception:
        fail("first token not an integer count")
    if m < 0 or m > MAX_LINES:
        fail("interval count out of range")
    need = 1 + 4 * m
    if len(toks) < need:
        fail("too few tokens for declared interval count")

    per_sat_intervals = [[] for _ in range(S)]
    pos = 1
    for _ in range(m):
        try:
            i = int(toks[pos]); k = int(toks[pos+1])
            st = int(toks[pos+2]); en = int(toks[pos+3])
        except Exception:
            fail("non-integer token")
        pos += 4
        for v in (i, k, st, en):
            if not math.isfinite(v):
                fail("non-finite token")
        if i < 0 or i >= S:
            fail("satellite index out of range")
        if k < 0 or k >= K:
            fail("station index out of range")
        if st < 0 or en > T or st >= en:
            fail("interval out of [0,T) or empty")

        base, dur = win[(i, k)]
        # find cycle c such that c*P+base <= st and en <= c*P+base+dur
        if dur <= 0:
            fail("zero-duration station window")
        shifted = st - base
        if shifted < 0:
            fail("interval starts before any visibility window")
        c = shifted // P
        if c < 0 or c >= R:
            fail("interval references a nonexistent cycle")
        wstart = c * P + base
        wend = wstart + dur
        if st < wstart or en > wend:
            fail("interval not fully contained in a visibility window")

        per_sat_intervals[i].append((st, en, k))

    sched = {}
    for i in range(S):
        ivs = sorted(per_sat_intervals[i])
        for j in range(1, len(ivs)):
            if ivs[j][0] < ivs[j-1][1]:
                fail("satellite has overlapping transmissions (single transceiver)")
        for (st, en, k) in ivs:
            sched.setdefault((i, k), []).append((st, en))
    for key in sched:
        sched[key].sort()

    F = simulate(S, K, P, R, T, sats, stations, sched)

    # ---- checker's own trivial baseline: every satellite uses ONLY station 0's
    # full window, every cycle, with no coordination among satellites ----
    base_sched = {}
    for i in range(S):
        base_l, dur = win[(i, 0)]
        ivs = []
        for c in range(R):
            wstart = c * P + base_l
            ivs.append((wstart, wstart + dur))
        base_sched[(i, 0)] = ivs
    B = simulate(S, K, P, R, T, sats, stations, base_sched)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
