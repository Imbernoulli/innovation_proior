import sys, heapq

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_WAYPOINTS_PER_PLANE = 20000


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def dijkstra(R, C, delay, origins, lakeset, fireproof):
    """Multi-source shortest-ignite-time map. A cell popped at time t is blocked
    (never ignites, never propagates) iff it was fireproofed at or before t."""
    dist = [[None] * C for _ in range(R)]
    heap = []
    for (r, c) in origins:
        if (r, c) in lakeset:
            continue
        heapq.heappush(heap, (0, r, c))
    while heap:
        t, r, c = heapq.heappop(heap)
        if dist[r][c] is not None:
            continue
        ft = fireproof.get((r, c))
        if ft is not None and ft <= t:
            continue  # saved just in time (or earlier) -> never ignites, never spreads
        dist[r][c] = t
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in lakeset and dist[nr][nc] is None:
                heapq.heappush(heap, (t + delay[nr][nc], nr, nc))
    return dist


def area_le(dist, R, C, Tmax):
    return sum(1 for r in range(R) for c in range(C) if dist[r][c] is not None and dist[r][c] <= Tmax)


def simulate(R, C, delay, origins, lakes, Tmax, planes, itins, move, Rfill):
    """planes: list of (r,c,tank_cap). itins: list of list of (r,c,'D'|'F').
    Returns (fireproof_dict, None) or (None, error_message)."""
    lakeset = set(lakes)
    lake_free = {l: 0 for l in lakes}
    fireproof = {}
    heap = []
    state = []
    for i, (pr, pc, tank_cap) in enumerate(planes):
        state.append({"pos": (pr, pc), "time": 0, "tank": tank_cap})
        if itins[i]:
            r, c, a = itins[i][0]
            travel = (abs(r - pr) + abs(c - pc)) * move
            heapq.heappush(heap, (state[i]["time"] + travel, i, 0))
    while heap:
        t, i, wi = heapq.heappop(heap)
        pr0, pc0, tank_cap = planes[i]
        st = state[i]
        r, c, a = itins[i][wi]
        if a == "D":
            if st["tank"] <= 0:
                return None, f"plane {i} drops with empty tank at waypoint {wi}"
            if (r, c) not in fireproof:
                fireproof[(r, c)] = t
            st["tank"] -= 1
            st["pos"] = (r, c)
            st["time"] = t
        else:  # 'F' refill
            rs = max(t, lake_free[(r, c)])
            re_ = rs + Rfill
            lake_free[(r, c)] = re_
            st["tank"] = tank_cap
            st["pos"] = (r, c)
            st["time"] = re_
        if wi + 1 < len(itins[i]):
            nr, nc, na = itins[i][wi + 1]
            travel = (abs(nr - st["pos"][0]) + abs(nc - st["pos"][1])) * move
            heapq.heappush(heap, (st["time"] + travel, i, wi + 1))
    return fireproof, None


def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    try:
        it = iter(inp)
        R = int(next(it)); C = int(next(it)); Tmax = int(next(it))
        move = int(next(it)); Rfill = int(next(it))
        S = int(next(it))
        origins = [(int(next(it)), int(next(it))) for _ in range(S)]
        L = int(next(it))
        lakes = [(int(next(it)), int(next(it))) for _ in range(L)]
        K = int(next(it))
        planes = [(int(next(it)), int(next(it)), int(next(it))) for _ in range(K)]
        delay = [[int(next(it)) for _ in range(C)] for _ in range(R)]
    except Exception:
        fail("bad input")

    lakeset = set(lakes)
    originset = set(origins)

    # ---- internal baseline B: do-nothing burn area ----
    base_dist = dijkstra(R, C, delay, origins, lakeset, {})
    B = max(1, area_le(base_dist, R, C, Tmax))

    # ---- parse participant output ----
    try:
        ot = iter(out)
        Kp = int(next(ot))
    except Exception:
        fail("parse K")
    if Kp != K:
        fail("plane count mismatch: expected %d got %d" % (K, Kp))

    itins = []
    try:
        for i in range(K):
            m = int(next(ot))
            if m < 0 or m > MAX_WAYPOINTS_PER_PLANE:
                fail("bad waypoint count for plane %d" % i)
            wps = []
            for _ in range(m):
                r = int(next(ot)); c = int(next(ot)); a = next(ot)
                if not (0 <= r < R and 0 <= c < C):
                    fail("waypoint out of bounds (%d,%d)" % (r, c))
                if a not in ("D", "F"):
                    fail("bad action token %r" % a)
                if a == "F":
                    if (r, c) not in lakeset:
                        fail("refill target (%d,%d) is not a lake" % (r, c))
                else:  # D
                    if (r, c) in lakeset:
                        fail("cannot drop on a lake cell (%d,%d)" % (r, c))
                    if (r, c) in originset:
                        fail("cannot drop on a fire-origin cell (%d,%d)" % (r, c))
                wps.append((r, c, a))
            itins.append(wps)
    except Exception:
        fail("parse waypoints")

    fireproof, err = simulate(R, C, delay, origins, lakes, Tmax, planes, itins, move, Rfill)
    if err is not None:
        fail(err)

    final_dist = dijkstra(R, C, delay, origins, lakeset, fireproof)
    F = area_le(final_dist, R, C, Tmax)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("B=%d F=%d Ratio: %.6f" % (B, F, sc / 1000.0))


if __name__ == "__main__":
    main()
