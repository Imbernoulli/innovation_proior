# TIER: trivial
# An even more naive first idea than "spread them out": drop A anchors on a
# single ring at a modest, fixed radius around the site's overall centroid,
# evenly spaced BY ANGLE. This "looks" principled (a ring is angularly
# diverse from the *centroid*) but the checker only cares about angular
# diversity as seen FROM each scenario's own targets, and a small fixed-
# radius ring leaves most of the site (plaza2, corridor, perimeter, the
# junctions) far outside the ring, seeing it edge-on -- worse than actually
# spreading anchors across the whole reachable site (the greedy tier).
import math
import sys


def read_instance():
    data = sys.stdin.read().splitlines()
    p = 0
    W, H, A, K, T = map(int, data[p].split()); p += 1
    grid = []
    for _ in range(H):
        grid.append(data[p]); p += 1
    return W, H, A, grid


def open_cells(grid):
    H = len(grid); W = len(grid[0])
    return [(r, c) for r in range(H) for c in range(W) if grid[r][c] == '.']


def nearest_open(grid, r, c):
    best = None; bestd = None
    for (rr, cc) in open_cells(grid):
        d = (rr - r) ** 2 + (cc - c) ** 2
        if bestd is None or d < bestd:
            bestd = d; best = (rr, cc)
    return best


def ring_construction(grid, A, frac=0.45):
    cells = open_cells(grid)
    cr = sum(r for r, c in cells) / len(cells)
    cc = sum(c for r, c in cells) / len(cells)
    rmax = max(math.hypot(r - cr, c - cc) for r, c in cells)
    radius = frac * rmax
    chosen = []; seen = set(); k = 0; attempts = 0
    H = len(grid); W = len(grid[0])
    while len(chosen) < A and attempts < A * 6:
        ang = 2 * math.pi * k / A
        rr = max(0, min(H - 1, cr + radius * math.sin(ang)))
        ccx = max(0, min(W - 1, cc + radius * math.cos(ang)))
        pt = nearest_open(grid, round(rr), round(ccx))
        if pt not in seen:
            chosen.append(pt); seen.add(pt)
        k += 1; attempts += 1
    if len(chosen) < A:
        remaining = [p for p in cells if p not in seen]
        while len(chosen) < A and remaining:
            best = None; bestd = -1
            for p in remaining:
                dmin = min((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 for q in chosen) if chosen else 1e9
                if dmin > bestd:
                    bestd = dmin; best = p
            chosen.append(best); remaining.remove(best)
    return chosen[:A]


def main():
    W, H, A, grid = read_instance()
    anchors = ring_construction(grid, A)
    out = []
    for (r, c) in anchors:
        out.append(f"{r} {c}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
