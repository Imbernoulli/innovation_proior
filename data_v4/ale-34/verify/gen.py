#!/usr/bin/env python3
"""Instance generator for ale-34 "Maze Carving to a Target Difficulty".

Usage: python3 gen.py SEED  > instance.txt

Instance format (stdin of the solver):
    H W B
    sr sc tr tc
    H lines of W chars each, '#' = wall, '.' = open corridor.

Guarantees:
  - 1 <= H, W; cells (sr,sc) and (tr,tc) are distinct and open ('.').
  - B is a positive carving budget with B <= (number of wall cells), so a
    feasible carving (carving exactly B distinct wall cells) always exists.
  - There is always at least one feasible solution that connects S to T
    (the grid is sized / B chosen so a connecting snake fits in budget).
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed * 1000003 + 12345)

    # Grid size: varied but bounded so BFS/SA stay cheap.
    H = rng.randint(20, 30)
    W = rng.randint(20, 30)

    # Start near top-left, end near bottom-right, kept apart.
    sr, sc = rng.randint(0, 2), rng.randint(0, 2)
    tr, tc = H - 1 - rng.randint(0, 2), W - 1 - rng.randint(0, 2)

    # Initial grid: mostly walls, a sprinkling of pre-open cells (p in [0.05,0.20]).
    p_open = rng.uniform(0.05, 0.20)
    grid = [['#'] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if rng.random() < p_open:
                grid[r][c] = '.'
    grid[sr][sc] = '.'
    grid[tr][tc] = '.'

    # Count current walls.
    walls = sum(1 for r in range(H) for c in range(W) if grid[r][c] == '#')

    # Budget: a fraction of free area, large enough to wind, small enough that
    # the whole grid cannot be opened. Manhattan distance is a lower bound on
    # the carves needed just to connect, so keep B comfortably above it.
    manh = abs(sr - tr) + abs(sc - tc)
    lo = manh + 2
    hi = max(lo + 5, (H * W) // 3)
    B = rng.randint(lo, hi)
    # Never exceed available walls (must be able to carve B distinct walls).
    B = min(B, walls)

    out = []
    out.append(f"{H} {W} {B}")
    out.append(f"{sr} {sc} {tr} {tc}")
    for r in range(H):
        out.append(''.join(grid[r]))
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
