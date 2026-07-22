# TIER: greedy
# Obvious recipe: compute the deterministic fire-arrival-time map (public information --
# the terrain and origin are known), then reactively chase whichever still-unprotected
# cell is about to ignite soonest, sending whichever plane can reach it first (refuelling
# if its tank is empty). This is exactly "dab at the current largest flame front": it
# never plans a closed perimeter, so in an open area it just pokes isolated holes that
# the fire flows straight around, and it never recognizes the far, cheap choke point.
import sys, heapq

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def dijkstra(R, C, delay, origins, lakeset):
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
        dist[r][c] = t
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in lakeset and dist[nr][nc] is None:
                heapq.heappush(heap, (t + delay[nr][nc], nr, nc))
    return dist


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def try_visit(pos, time, tank, tank_cap, lakes, move, Rfill, deadline, target):
    """If the plane (at pos/time/tank) can reach target on/before deadline, possibly
    refuelling first, return (arrive, extra_waypoints, new_tank); else None."""
    extra = []
    if tank <= 0:
        bl = min(lakes, key=lambda l: manh(l, pos))
        trav = manh(bl, pos) * move
        time = time + trav + Rfill
        pos = bl
        tank = tank_cap
        extra = [(bl[0], bl[1], "F")]
    trav = manh(target, pos) * move
    arrive = time + trav
    if arrive > deadline:
        return None
    return arrive, extra, tank - 1


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    R = int(next(it)); C = int(next(it)); Tmax = int(next(it))
    move = int(next(it)); Rfill = int(next(it))
    S = int(next(it))
    origins = [(int(next(it)), int(next(it))) for _ in range(S)]
    L = int(next(it))
    lakes = [(int(next(it)), int(next(it))) for _ in range(L)]
    K = int(next(it))
    planes = [(int(next(it)), int(next(it)), int(next(it))) for _ in range(K)]
    delay = [[int(next(it)) for _ in range(C)] for _ in range(R)]

    lakeset = set(lakes)
    originset = set(origins)
    base = dijkstra(R, C, delay, origins, lakeset)

    cells = [(base[r][c], r, c) for r in range(R) for c in range(C)
             if base[r][c] is not None and base[r][c] <= Tmax
             and (r, c) not in lakeset and (r, c) not in originset]
    cells.sort()  # soonest-igniting first -- the "chase the current flame front" order

    pos = [(planes[i][0], planes[i][1]) for i in range(K)]
    time = [0] * K
    tank = [planes[i][2] for i in range(K)]
    tank_cap = [planes[i][2] for i in range(K)]
    itins = [[] for _ in range(K)]

    for d_, r, c in cells:
        best = None
        for i in range(K):
            res = try_visit(pos[i], time[i], tank[i], tank_cap[i], lakes, move, Rfill, d_, (r, c))
            if res is None:
                continue
            arrive, extra, newtank = res
            if best is None or arrive < best[0]:
                best = (arrive, i, extra, newtank)
        if best is None:
            continue  # nobody can beat the fire here -- skip and keep chasing
        arrive, i, extra, newtank = best
        itins[i].extend(extra)
        itins[i].append((r, c, "D"))
        pos[i] = (r, c); time[i] = arrive; tank[i] = newtank

    out = [str(K)]
    for i in range(K):
        out.append(str(len(itins[i])))
        for (r, c, a) in itins[i]:
            out.append(f"{r} {c} {a}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
