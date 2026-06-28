#!/usr/bin/env python3
"""Random + edge-case generator for the largest all-ones square problem.

Usage: gen.py SEED [MODE]
Prints a single test case to stdout in the format:
    H W
    row 0 (W ints)
    ...
    row H-1

Modes are chosen by SEED to give a spread of structures that exercise the DP
and stress the brute oracle's k-loop:
  - tiny grids (1x1 .. small)
  - sparse / dense random
  - planted large square inside noise
  - full-ones and full-zeros
  - single row / single column (degenerate squares)
"""
import random
import sys


def emit(H, W, grid):
    out = [f"{H} {W}"]
    for r in grid:
        out.append(" ".join(map(str, r)))
    sys.stdout.write("\n".join(out) + "\n")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    mode = seed % 9

    if mode == 0:
        # very tiny
        H = rng.randint(1, 4)
        W = rng.randint(1, 4)
        p = rng.random()
        grid = [[1 if rng.random() < p else 0 for _ in range(W)] for _ in range(H)]
    elif mode == 1:
        # small dense (many ones -> big squares likely)
        H = rng.randint(1, 12)
        W = rng.randint(1, 12)
        grid = [[1 if rng.random() < 0.85 else 0 for _ in range(W)] for _ in range(H)]
    elif mode == 2:
        # small sparse
        H = rng.randint(1, 12)
        W = rng.randint(1, 12)
        grid = [[1 if rng.random() < 0.25 else 0 for _ in range(W)] for _ in range(H)]
    elif mode == 3:
        # medium balanced
        H = rng.randint(1, 25)
        W = rng.randint(1, 25)
        grid = [[rng.randint(0, 1) for _ in range(W)] for _ in range(H)]
    elif mode == 4:
        # all ones
        H = rng.randint(1, 20)
        W = rng.randint(1, 20)
        grid = [[1] * W for _ in range(H)]
    elif mode == 5:
        # all zeros
        H = rng.randint(1, 20)
        W = rng.randint(1, 20)
        grid = [[0] * W for _ in range(H)]
    elif mode == 6:
        # single row or column
        if rng.random() < 0.5:
            H = 1
            W = rng.randint(1, 30)
        else:
            H = rng.randint(1, 30)
            W = 1
        grid = [[rng.randint(0, 1) for _ in range(W)] for _ in range(H)]
    elif mode == 7:
        # planted square inside noise: this is the case greedy/area-scan
        # heuristics get wrong, so make it common.
        H = rng.randint(8, 25)
        W = rng.randint(8, 25)
        grid = [[1 if rng.random() < 0.4 else 0 for _ in range(W)] for _ in range(H)]
        k = rng.randint(2, min(H, W))
        r0 = rng.randint(0, H - k)
        c0 = rng.randint(0, W - k)
        for i in range(r0, r0 + k):
            for j in range(c0, c0 + k):
                grid[i][j] = 1
        # Also plant a wider-but-shorter all-ones rectangle to lure area-scan
        # heuristics that confuse "largest all-ones rectangle" with "square".
        rh = rng.randint(1, 2)
        rw = rng.randint(min(W, k + 2), W)
        rr = rng.randint(0, H - rh)
        rc = rng.randint(0, W - rw)
        for i in range(rr, rr + rh):
            for j in range(rc, rc + rw):
                grid[i][j] = 1
    else:
        # medium-large randomish to push the brute a bit
        H = rng.randint(20, 40)
        W = rng.randint(20, 40)
        p = rng.choice([0.5, 0.7, 0.9, 0.95])
        grid = [[1 if rng.random() < p else 0 for _ in range(W)] for _ in range(H)]

    emit(H, W, grid)


if __name__ == "__main__":
    main()
