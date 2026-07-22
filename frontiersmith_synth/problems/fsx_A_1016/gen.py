#!/usr/bin/env python3
"""gen.py <testId> -- stick-weld-routing instance generator (deterministic, seeded by testId).

Prints:
  R C
  R grid rows ('.'=free, '#'=obstacle)
  Ar Ac Br Bc
  L a W
"""
import sys, random


def carve_zigzag(grid, rmid, c0, width, amp, seg_len):
    """Carve a forced zigzag corridor of 1-cell width from (rmid,c0) to (rmid,c0+width-1),
    oscillating between row rmid-amp and rmid+amp every seg_len columns, then reconnect to
    rmid at the far end. Returns nothing; mutates grid in place."""
    c1 = c0 + width - 1
    row, col = rmid, c0
    grid[row][col] = '.'
    target = rmid - amp
    while col < c1:
        steps = min(seg_len, c1 - col)
        for _ in range(steps):
            col += 1
            grid[row][col] = '.'
        if col >= c1:
            break
        step = 1 if target > row else -1
        while row != target:
            row += step
            grid[row][col] = '.'
        target = (rmid + amp) if target == (rmid - amp) else (rmid - amp)
    step = 1 if rmid > row else (-1 if rmid < row else 0)
    while row != rmid:
        row += step
        grid[row][col] = '.'


def bfs_connected(grid, R, C, a, b):
    from collections import deque
    q = deque([a]); seen = {a}
    while q:
        r, c = q.popleft()
        if (r, c) == b:
            return True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == '.' and (nr, nc) not in seen:
                seen.add((nr, nc)); q.append((nr, nc))
    return False


def ensure_connected(grid, R, C, a, b):
    if bfs_connected(grid, R, C, a, b):
        return
    # carve a guaranteed straight-ish channel along a's row, then b's column (or vice versa)
    ar, ac = a; br, bc = b
    r = ar
    c0, c1 = min(ac, bc), max(ac, bc)
    for c in range(c0, c1 + 1):
        grid[r][c] = '.'
    c = bc
    r0, r1 = min(ar, br), max(ar, br)
    for rr in range(r0, r1 + 1):
        grid[rr][c] = '.'


PARAMS = {
    1: dict(mode="straight", L=10, a=2, W=5),
    2: dict(mode="random", R=10, C=16, density=0.12, L=9, a=2, W=5),
    3: dict(mode="random", R=13, C=21, density=0.24, L=8, a=2, W=7),
    4: dict(mode="zigzag", amp=2, seg_len=3, segs=3, L=8, a=2, W=6),
    5: dict(mode="zigzag", amp=2, seg_len=2, segs=5, L=8, a=2, W=8),
    6: dict(mode="zigzag", amp=3, seg_len=2, segs=6, L=9, a=2, W=8),
    7: dict(mode="zigzag", amp=3, seg_len=2, segs=7, L=9, a=3, W=9),
    8: dict(mode="zigzag", amp=4, seg_len=2, segs=8, L=10, a=3, W=9),
    9: dict(mode="zigzag", amp=5, seg_len=2, segs=9, L=10, a=3, W=10),
    10: dict(mode="zigzag", amp=5, seg_len=2, segs=10, L=11, a=3, W=10),
}


def build(testId):
    p = PARAMS[testId]
    rng = random.Random(1000 + 7 * testId)
    mode = p["mode"]

    if mode == "straight":
        R, C = 9, 15
        grid = [['.'] * C for _ in range(R)]
        rmid = R // 2
        A = (rmid, 0); B = (rmid, C - 1)

    elif mode == "random":
        R, C, density = p["R"], p["C"], p["density"]
        rmid = R // 2
        A = (rmid, 0); B = (rmid, C - 1)
        grid = [['.'] * C for _ in range(R)]
        for r in range(R):
            for c in range(C):
                if (r, c) in (A, B):
                    continue
                if rng.random() < density:
                    grid[r][c] = '#'
        ensure_connected(grid, R, C, A, B)

    elif mode == "zigzag":
        amp, seg_len, segs = p["amp"], p["seg_len"], p["segs"]
        width = segs * seg_len
        c0 = 4
        margin_after = 4
        rmid = amp + 4
        R = 2 * rmid + 1
        C = c0 + width + margin_after
        A = (rmid, 0); B = (rmid, C - 1)
        grid = [['.'] * C for _ in range(R)]
        # block the zigzag band [rmid-amp, rmid+amp] x [c0, c0+width-1] fully, then carve corridor
        c1 = c0 + width - 1
        for r in range(rmid - amp, rmid + amp + 1):
            for c in range(c0, c1 + 1):
                grid[r][c] = '#'
        carve_zigzag(grid, rmid, c0, width, amp, seg_len)
        ensure_connected(grid, R, C, A, B)
    else:
        raise ValueError(mode)

    return R, C, grid, A, B, p["L"], p["a"], p["W"]


def main():
    testId = int(sys.argv[1])
    R, C, grid, A, B, L, a, W = build(testId)
    out = []
    out.append(f"{R} {C}")
    for row in grid:
        out.append(''.join(row))
    out.append(f"{A[0]} {A[1]} {B[0]} {B[1]}")
    out.append(f"{L} {a} {W}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
