# TIER: greedy
"""Reactive perimeter defense (the 'obvious' approach).

Every stage, before the flood advances, look at every dry cell that is about to
become eligible (elevation <= this stage's level) and sits next to already-wet
water; if that cell is within a small grid-distance of the PROTECTED zone, wall
off its incoming wet edges, closest cells first, until this stage's remaining
budget runs out. This purely reactive, distance-to-target heuristic never looks
ahead at the terrain's actual connectivity, never distinguishes a real corridor
from a nearby dead end, and never bothers with the (secondary) forbidden zone at
all -- exactly the kind of first-pass solution an average solver would write."""
import sys
from collections import deque

DIST_THRESH = 2


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    R = int(next(it)); C = int(next(it)); K = int(next(it))
    elev = [[int(next(it)) for _ in range(C)] for _ in range(R)]
    NB = int(next(it))
    p_cells = []
    for _ in range(NB):
        sz = int(next(it))
        for _ in range(sz):
            r = int(next(it)); c = int(next(it))
            p_cells.append((r, c))
    Q = int(next(it))
    for _ in range(Q):
        next(it); next(it)  # forbidden zone: ignored entirely by this heuristic
    levels = [int(next(it)) for _ in range(K)]
    W = [int(next(it)) for _ in range(K)]
    next(it)  # alpha: ignored

    CumW = []
    s = 0
    for k in range(K):
        s += W[k]
        CumW.append(s)

    # multi-source BFS distance-to-nearest-protected-cell, plain grid adjacency
    dist = [[10**9] * C for _ in range(R)]
    dq = deque()
    for (r, c) in p_cells:
        if dist[r][c] > 0:
            dist[r][c] = 0
            dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and dist[nr][nc] > dist[r][c] + 1:
                dist[nr][nc] = dist[r][c] + 1
                dq.append((nr, nc))

    wet = [[False] * C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if elev[r][c] <= 0:
                wet[r][c] = True

    built = set()
    built_count = 0
    plan = []  # (stage, r1, c1, r2, c2)

    def neighbors(r, c):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                yield nr, nc

    for k in range(1, K + 1):
        level = levels[k - 1]
        budget_avail = CumW[k - 1] - built_count

        # frontier: dry, eligible cells adjacent (via an open edge) to a wet cell
        frontier = {}
        for r in range(R):
            for c in range(C):
                if wet[r][c] or elev[r][c] > level:
                    continue
                for nr, nc in neighbors(r, c):
                    if wet[nr][nc]:
                        e = frozenset(((r, c), (nr, nc)))
                        if e in built:
                            continue
                        d = dist[r][c]
                        if d <= DIST_THRESH:
                            frontier[(r, c)] = min(frontier.get((r, c), 10**9), d)

        order = sorted(frontier.keys(), key=lambda rc: (frontier[rc], rc[0], rc[1]))
        for (r, c) in order:
            if budget_avail <= 0:
                break
            for nr, nc in neighbors(r, c):
                if budget_avail <= 0:
                    break
                if not wet[nr][nc]:
                    continue
                e = frozenset(((r, c), (nr, nc)))
                if e in built:
                    continue
                built.add(e)
                built_count += 1
                budget_avail -= 1
                plan.append((k, r, c, nr, nc))

        # advance the flood for this stage with the walls built so far
        dq = deque((r, c) for r in range(R) for c in range(C) if wet[r][c])
        while dq:
            r, c = dq.popleft()
            for nr, nc in neighbors(r, c):
                if wet[nr][nc] or elev[nr][nc] > level:
                    continue
                e = frozenset(((r, c), (nr, nc)))
                if e in built:
                    continue
                wet[nr][nc] = True
                dq.append((nr, nc))

    out = [str(len(plan))]
    for (st, r1, c1, r2, c2) in plan:
        out.append(f"{st} {r1} {c1} {r2} {c2}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
