# TIER: greedy
# The "obvious" recipe: read the site map, ignore the K scenarios entirely,
# and spread A anchors as far apart from each other as possible (iterated
# farthest-point sampling on Euclidean distance). This is exactly the
# checker's own internal normalizer construction -- a solver that submits
# this deserves, and gets, the reference score.
import sys


def read_instance():
    data = sys.stdin.read().splitlines()
    p = 0
    W, H, A, K, T = map(int, data[p].split()); p += 1
    grid = []
    for _ in range(H):
        grid.append(data[p]); p += 1
    # scenarios are ignored by this tier
    return W, H, A, grid


def open_cells(grid):
    H = len(grid); W = len(grid[0])
    return [(r, c) for r in range(H) for c in range(W) if grid[r][c] == '.']


def farthest_point(grid, A):
    # Tie-break is fully specified: scan candidates in sorted (r, c) order and
    # keep the first (lexicographically smallest) cell achieving the max
    # min-distance -- not left to hash/set iteration order.
    cells = sorted(open_cells(grid))
    start = cells[len(cells) // 2]
    chosen = [start]
    remaining = [p for p in cells if p != start]
    while len(chosen) < A and remaining:
        best = None; bestd = -1
        for p in remaining:
            dmin = min((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 for q in chosen)
            if dmin > bestd:
                bestd = dmin; best = p
        chosen.append(best)
        remaining.remove(best)
    return chosen


def main():
    W, H, A, grid = read_instance()
    anchors = farthest_point(grid, A)
    out = []
    for (r, c) in anchors:
        out.append(f"{r} {c}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
