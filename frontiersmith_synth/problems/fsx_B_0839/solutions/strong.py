# TIER: strong
"""The insight: every valve multiplies the batch's volume by a fixed
rational, so a state is fully described by (tank, exact volume) -- and the
set of volumes reachable within any bounded number of hops is a FINITE
multiplicative lattice (capacities and hop-count bound it), not a continuum.
Instead of a numeric-distance heuristic, dedupe reachable (tank, volume)
pairs and run BFS over that finite lattice: it costs nothing extra on easy
instances and finds the short "overshoot now, cancel later" plans that a
one-step-lookahead nearest-value search can never see, because it never
commits to a locally "worse" move."""
import sys
from fractions import Fraction
from collections import deque


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it))
    caps = [int(next(it)) for _ in range(N)]
    V0 = int(next(it)); target = int(next(it)); Nb = int(next(it))
    Lo = Fraction(int(next(it)), int(next(it)))
    Hi = Fraction(int(next(it)), int(next(it)))
    backbone_ids = [int(next(it)) for _ in range(Nb)]
    edges = []
    for _ in range(M):
        u = int(next(it)); v = int(next(it))
        num = int(next(it)); den = int(next(it))
        edges.append((u, v, num, den))

    by_tank = {}
    for eid, (u, v, num, den) in enumerate(edges):
        by_tank.setdefault(u, []).append((eid, v, num, den))

    start_tank, start_vol = 0, Fraction(V0)
    if start_tank == target and Lo <= start_vol <= Hi:
        print(1)
        print(0)
        return

    visited = {(start_tank, start_vol)}
    q = deque([(start_tank, start_vol, [])])
    MAXDEPTH = 60
    best = None

    while q and best is None:
        tank, vol, path = q.popleft()
        if len(path) >= MAXDEPTH:
            continue
        for (eid, v, num, den) in by_tank.get(tank, []):
            nv = vol * Fraction(num, den)
            if nv > caps[v]:
                continue
            if v == target and Lo <= nv <= Hi:
                best = path + [eid]
                break
            state = (v, nv)
            if state in visited:
                continue
            visited.add(state)
            q.append((v, nv, path + [eid]))

    if best is None:
        best = backbone_ids             # fallback: the guaranteed backbone

    print(len(best))
    print(" ".join(map(str, best)))


if __name__ == "__main__":
    main()
