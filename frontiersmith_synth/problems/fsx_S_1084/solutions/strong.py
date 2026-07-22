# TIER: strong
# Insight: the fire's spread is fully deterministic, so its future is a computable
# geometry (a shortest-path "arrival time" map). Instead of reacting to whatever is
# currently burning, compute the whole isochrone family {S_T = cells that ignite by
# time T} up front, and for each candidate cut time T ask a scheduling question: can
# the fleet -- given travel time and refill-queue logistics -- fully fireproof the
# boundary ring of S_T (each ring cell strictly before its own arrival time)? Among
# every T for which that schedule is feasible, keep the one that leaves the SMALLEST
# burned interior. This is a minimum-cut-in-time search: a cut far from the origin can
# be far cheaper (a narrow ring) than one close to the origin (which is tight on time
# even though it's geometrically small), so the winning cut is wherever ring length
# matches the fleet's sustainable drop rate -- not wherever the fire currently is.
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
    if deadline is not None and arrive > deadline:
        return None
    return arrive, extra, tank - 1


class Fleet:
    """Mutable per-plane state used while building an itinerary plan."""

    def __init__(self, planes):
        self.pos = [(p[0], p[1]) for p in planes]
        self.time = [0] * len(planes)
        self.tank = [p[2] for p in planes]
        self.cap = [p[2] for p in planes]

    def snapshot(self):
        return (list(self.pos), list(self.time), list(self.tank))

    def restore(self, snap):
        self.pos, self.time, self.tank = list(snap[0]), list(snap[1]), list(snap[2])


def schedule_cells(fleet, cells_with_deadline, lakes, move, Rfill, itins):
    """Greedily assign each (r,c,deadline) -- in ascending deadline order -- to
    whichever plane can reach it earliest; abort (restoring fleet state) if any
    cell has no plane able to make its deadline. Mutates itins in place on success."""
    snap = fleet.snapshot()
    K = len(fleet.pos)
    added = [[] for _ in range(K)]
    for (r, c, deadline) in cells_with_deadline:
        best = None
        for i in range(K):
            res = try_visit(fleet.pos[i], fleet.time[i], fleet.tank[i], fleet.cap[i],
                             lakes, move, Rfill, deadline, (r, c))
            if res is None:
                continue
            arrive, extra, newtank = res
            if best is None or arrive < best[0]:
                best = (arrive, i, extra, newtank)
        if best is None:
            fleet.restore(snap)
            return False
        arrive, i, extra, newtank = best
        added[i].extend(extra)
        added[i].append((r, c, "D"))
        fleet.pos[i] = (r, c); fleet.time[i] = arrive; fleet.tank[i] = newtank
    for i in range(K):
        itins[i].extend(added[i])
    return True


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

    allc = [(r, c) for r in range(R) for c in range(C)
            if base[r][c] is not None and base[r][c] <= Tmax
            and (r, c) not in lakeset and (r, c) not in originset]
    Tvals = sorted(set(base[r][c] for (r, c) in allc))

    def ring_for_T(T):
        Sset = set((r, c) for (r, c) in allc if base[r][c] <= T)
        ring = set()
        for (r, c) in Sset:
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if (0 <= nr < R and 0 <= nc < C and (nr, nc) not in Sset
                        and (nr, nc) not in lakeset and (nr, nc) not in originset
                        and base[nr][nc] is not None and base[nr][nc] <= Tmax):
                    ring.add((nr, nc))
        return ring

    best_T, best_S, best_plan = None, None, None
    for T in Tvals:
        ring = ring_for_T(T)
        if not ring:
            continue
        fleet = Fleet(planes)
        itins = [[] for _ in range(K)]
        cells = sorted(((r, c, base[r][c]) for (r, c) in ring), key=lambda x: x[2])
        if schedule_cells(fleet, cells, lakes, move, Rfill, itins):
            Ssize = sum(1 for (r, c) in allc if base[r][c] <= T)
            if best_S is None or Ssize < best_S:
                best_S, best_T, best_plan = Ssize, T, (fleet, itins)

    if best_plan is None:
        # No isochrone cut is schedulable at all (very constrained fleet): fall back
        # to the myopic soonest-reachable-first recipe so we never do worse than it.
        fleet = Fleet(planes)
        itins = [[] for _ in range(K)]
        cells = sorted(((r, c, base[r][c]) for (r, c) in allc), key=lambda x: x[2])
        schedule_cells(fleet, cells, lakes, move, Rfill, itins)
    else:
        fleet, itins = best_plan
        # Mop up with leftover fleet capacity: the interior S_T still burns
        # unavoidably, but any spare drops can still shave a few more cells,
        # soonest-deadline first, at no cost to the sealed cut.
        remaining = sorted(((r, c, base[r][c]) for (r, c) in allc if base[r][c] <= best_T),
                            key=lambda x: x[2])
        schedule_cells(fleet, remaining, lakes, move, Rfill, itins)

    out = [str(K)]
    for i in range(K):
        out.append(str(len(itins[i])))
        for (r, c, a) in itins[i]:
            out.append(f"{r} {c} {a}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
