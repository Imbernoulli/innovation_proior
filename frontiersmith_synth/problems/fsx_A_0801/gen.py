#!/usr/bin/env python3
"""gen.py <testId> -> one instance on stdout.

testId 1..10 is a difficulty ladder: terminal count and grid size grow, and
from testId 4 onward the terminals sit on a circle around an open interior
("hub candidate" region), while random wall segments (seeded, connectivity
verified) force detours -- the classic Steiner trap: pairwise shortest paths
between terminals stay near the rim, but a single interior junction point
reaches all of them more cheaply once the walls are in place.

Instance format (whitespace tokens):
    R C K
    r_1 c_1 r_2 c_2 ... r_K c_K      (terminal coordinates, fixed order)
    W
    or_1 oc_1 ... or_W oc_W          (obstacle cell coordinates)

Everything is seeded from testId only, so generation is bit-for-bit
reproducible.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physlib as pl

# (grid size, #terminals, ring radius or None for scattered, #wall segments, wall length)
LADDER = [
    (7, 2, None, 0, 0),
    (8, 3, 3, 2, 3),
    (9, 3, 3, 4, 3),
    (10, 4, 4, 3, 3),
    (10, 4, 4, 6, 4),
    (11, 5, 4, 4, 4),
    (12, 5, 5, 6, 4),
    (12, 5, 5, 9, 5),
    (13, 5, 5, 10, 5),
    (13, 6, 5, 12, 5),
]


def build_instance(t):
    size, k, radius, n_walls, wall_len = LADDER[t - 1]
    R = C = size
    rng = random.Random(90000 + 131 * t)

    if radius is None:
        pts = set()
        while len(pts) < k:
            pts.add((rng.randrange(R), rng.randrange(C)))
        terminals = list(pts)
    else:
        phase = 0.35 * t

        def jitter(i):
            return (rng.randint(-1, 1), rng.randint(-1, 1))

        raw = pl.polar_terminals(R, C, k, radius, phase, jitter)
        seen = set()
        terminals = []
        for (r, c) in raw:
            while (r, c) in seen:
                c = min(C - 1, c + 1)
            seen.add((r, c))
            terminals.append((r, c))

    # protect terminals and a small central hub-candidate zone from obstacles
    protected = set(terminals)
    cr, cc = R // 2, C // 2
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, ccc = cr + dr, cc + dc
            if 0 <= rr < R and 0 <= ccc < C:
                protected.add((rr, ccc))

    obstacles = set()
    attempts = 0
    placed = 0
    while placed < n_walls and attempts < n_walls * 20 + 40:
        attempts += 1
        r0 = rng.randrange(R)
        c0 = rng.randrange(C)
        horiz = rng.random() < 0.5
        cells = []
        ok = True
        for i in range(wall_len):
            rr = r0 + (0 if horiz else i)
            ccc = c0 + (i if horiz else 0)
            if not (0 <= rr < R and 0 <= ccc < C) or (rr, ccc) in protected:
                ok = False
                break
            cells.append((rr, ccc))
        if not ok:
            continue
        trial = obstacles | set(cells)
        node_id, coords, edges, term_nodes = pl.build_grid(R, C, trial, terminals)
        if pl.bfs_reachable_all(len(coords), edges, term_nodes[0], term_nodes):
            obstacles = trial
            placed += 1

    return R, C, terminals, obstacles


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    t = max(1, min(len(LADDER), t))

    R, C, terminals, obstacles = build_instance(t)
    K = len(terminals)
    W = len(obstacles)

    out = []
    out.append("%d %d %d" % (R, C, K))
    out.append(" ".join("%d %d" % (r, c) for (r, c) in terminals))
    out.append("%d" % W)
    if W:
        out.append(" ".join("%d %d" % (r, c) for (r, c) in sorted(obstacles)))
    else:
        out.append("")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
