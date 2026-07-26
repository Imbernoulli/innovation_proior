#!/usr/bin/env python3
# Deterministic checker for Pack-Ice Convoy Leases (format C, minimize weighted arrival).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1] on its own final line, then exits 0.
import sys, re
from bisect import bisect_right

MAXTICK = 10 ** 7
INT_RE = re.compile(r'^[+-]?\d+$')


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
        p = 0
        L = int(itoks[p]); p += 1
        M = int(itoks[p]); p += 1
        B = int(itoks[p]); p += 1
        c = int(itoks[p]); p += 1
        r = int(itoks[p]); p += 1
        ships = []  # 1-indexed via ships[id-1] = (s, d, w)
        for _ in range(M):
            s = int(itoks[p]); p += 1
            d = int(itoks[p]); p += 1
            w = int(itoks[p]); p += 1
            ships.append((s, d, w))
    except Exception:
        fail("bad instance")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    pos = [0]

    def nxt_int():
        if pos[0] >= len(otoks):
            raise ValueError("unexpected eof")
        tok = otoks[pos[0]]
        pos[0] += 1
        if not INT_RE.match(tok):
            raise ValueError("non-integer token %r" % tok)
        return int(tok)

    trips = []   # (breaker, t0, [ship_ids])
    piggy = []   # (ship_id, t0)
    used = set()

    try:
        T = nxt_int()
        if T < 0 or T > M:
            raise ValueError("bad T")
        for _ in range(T):
            b = nxt_int()
            if b < 0 or b >= B:
                raise ValueError("breaker id out of range")
            t0 = nxt_int()
            if t0 < 0 or t0 > MAXTICK:
                raise ValueError("t0 out of range")
            k = nxt_int()
            if k < 1 or k > c:
                raise ValueError("convoy size out of [1,c]")
            ids = []
            for _ in range(k):
                sid = nxt_int()
                if sid < 1 or sid > M:
                    raise ValueError("ship id out of range")
                if sid in used:
                    raise ValueError("ship id reused")
                used.add(sid)
                ids.append(sid)
            trips.append((b, t0, ids))
        P = nxt_int()
        if P < 0 or P > M:
            raise ValueError("bad P")
        for _ in range(P):
            sid = nxt_int()
            if sid < 1 or sid > M:
                raise ValueError("ship id out of range")
            if sid in used:
                raise ValueError("ship id reused")
            used.add(sid)
            t0 = nxt_int()
            if t0 < 0 or t0 > MAXTICK:
                raise ValueError("t0 out of range")
            piggy.append((sid, t0))
        if pos[0] != len(otoks):
            raise ValueError("trailing tokens after parse")
        if len(used) != M:
            raise ValueError("not every ship id used exactly once")
    except Exception as e:
        fail("parse: %s" % e)

    # ---- simulate escorted trips, per breaker, in departure-tick order ----
    by_breaker = {b: [] for b in range(B)}
    for (b, t0, ids) in trips:
        by_breaker[b].append((t0, ids))

    arrivals = {}
    clear_events = {}  # cell -> list of clearing ticks (multiple trips may clear it)

    try:
        for b in range(B):
            lst = sorted(by_breaker[b], key=lambda x: x[0])
            avail = 0
            for (t0, ids) in lst:
                if t0 < avail:
                    raise ValueError("breaker %d trip departs before it is free" % b)
                convoy = [(sid, ships[sid - 1][0], ships[sid - 1][1]) for sid in ids]
                maxD = max(d for (_, _, d) in convoy)
                if maxD > L:
                    raise ValueError("destination beyond channel")
                cur = t0
                for cell in range(1, maxD + 1):
                    active = [s for (_, s, d) in convoy if d >= cell]
                    step = max(1, max(active))
                    cur += step
                    clear_events.setdefault(cell, []).append(cur)
                    for (sid, s, d) in convoy:
                        if d == cell:
                            arrivals[sid] = cur
                finish = cur
                avail = finish + maxD  # unescorted return transit before the next trip
    except Exception as e:
        fail("escort sim: %s" % e)

    for cell in clear_events:
        clear_events[cell].sort()

    try:
        for (sid, t0) in piggy:
            s, d, w = ships[sid - 1]
            if d > L:
                raise ValueError("destination beyond channel")
            for cell in range(1, d + 1):
                need_tick = t0 + cell * s
                lst = clear_events.get(cell)
                if not lst:
                    raise ValueError("cell %d never cleared" % cell)
                idx = bisect_right(lst, need_tick) - 1
                if idx < 0:
                    raise ValueError("cell %d not yet clear at tick %d" % (cell, need_tick))
                if not (need_tick < lst[idx] + r):
                    raise ValueError("cell %d refrozen by tick %d" % (cell, need_tick))
            arrivals[sid] = t0 + d * s
    except Exception as e:
        fail("lease sim: %s" % e)

    if len(arrivals) != M:
        fail("missing arrivals")

    F = sum(ships[j][2] * arrivals[j + 1] for j in range(M))

    # ---- internal trivial baseline: escort every ship alone, round-robin, ASAP ----
    avail = [0] * B
    Fb = 0
    for j in range(1, M + 1):
        s, d, w = ships[j - 1]
        b = (j - 1) % B
        t0 = avail[b]
        arrival = t0 + d * s
        avail[b] = arrival + d
        Fb += w * arrival

    sc = min(1000.0, 100.0 * Fb / max(1e-9, F))
    print("F=%d B=%d Ratio: %.6f" % (F, Fb, sc / 1000.0))


if __name__ == "__main__":
    main()
