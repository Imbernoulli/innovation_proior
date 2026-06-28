#!/usr/bin/env python3
"""Trivial baseline for ale-45: a BOUSTROPHEDON / spanning-tree SWEEP route.

The guard walks a depth-first sweep of the connected free region from the start,
descending to unvisited neighbours and backtracking, which is a closed Euler
tour of a spanning tree (always returns to the start).  The descent is cut off
as soon as the remaining budget can no longer afford "one more step down plus
the full backtrack to the start", guaranteeing the emitted route is closed and
within the budget L.  Neighbours are tried in a fixed serpentine order
(prefer continuing along the current row, giving the classic snake sweep).

Emits the move string on stdout (or "0" for the empty route) in the scorer's
format.
"""
import sys

MOVE = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
# move order tried at each cell: Right, Down, Left, Up (snake-ish)
ORDER = ['R', 'D', 'L', 'U']


def main():
    data = sys.stdin.read().split('\n')
    i = 0
    while i < len(data) and data[i].strip() == '':
        i += 1
    H, W, L = map(int, data[i].split())
    i += 1
    grid = []
    for r in range(H):
        row = data[i + r]
        if len(row) < W:
            row = row + '.' * (W - len(row))
        grid.append(list(row[:W]))
    sr = sc = -1
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 'S':
                sr, sc = r, c

    visited = [[False] * W for _ in range(H)]
    visited[sr][sc] = True
    moves = []
    stack = []           # the descent path of moves (so we know how to backtrack)

    def free(r, c):
        return 0 <= r < H and 0 <= c < W and grid[r][c] != '#'

    r, c = sr, sc
    progress = True
    while progress:
        progress = False
        depth = len(stack)
        # Try to descend to an unvisited neighbour, but only if we can still
        # afford this step AND the backtrack of the whole stack afterwards.
        for ch in ORDER:
            dr, dc = MOVE[ch]
            nr, nc = r + dr, c + dc
            if free(nr, nc) and not visited[nr][nc]:
                # cost after descending = len(moves)+1 ; backtrack cost = depth+1
                if len(moves) + 1 + (depth + 1) <= L:
                    visited[nr][nc] = True
                    moves.append(ch)
                    stack.append(ch)
                    r, c = nr, nc
                    progress = True
                    break
        if progress:
            continue
        # No descent possible: backtrack one step if we can.
        if stack:
            ch = stack.pop()
            dr, dc = MOVE[ch]
            # reverse move
            inv = {'R': 'L', 'L': 'R', 'U': 'D', 'D': 'U'}[ch]
            moves.append(inv)
            r, c = r - dr, c - dc
            progress = True

    if not moves:
        sys.stdout.write("0\n")
    else:
        sys.stdout.write("".join(moves) + "\n")


if __name__ == "__main__":
    main()
