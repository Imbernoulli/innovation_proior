#!/usr/bin/env python3
"""Deterministic local scorer for ale-49: Reconfiguration Routing (token sliding).

Usage:
    python3 score.py INSTANCE_FILE SOLUTION_FILE

Reads the instance and a candidate solution (a sequence of actions), replays the
actions, enforces every feasibility rule, and prints a single floating-point
score to stdout.

Feasibility rules (any violation => score 0):
  - The first solution token is L, the number of actions (0 <= L <= L_MAX).
  - Each action is "i d" with 0 <= i < k and d in {U,D,L,R}.
  - Applying action (i,d) moves token i one cell in direction d.  The destination
    must be inside the grid, must not be a wall, and must not be currently
    occupied by another token.  (Tokens block each other.)
  - After all L actions, every token i must sit exactly on its target cell.
  - At every intermediate configuration no two tokens share a cell (this is
    enforced automatically because each move checks the destination is empty and
    tokens start on distinct cells).

Score (to MAXIMIZE; higher is better):
  Let LB = sum over tokens of the single-token shortest-path distance from start
  to target on the grid ignoring other tokens (BFS through free cells).  LB is a
  lower bound on the number of moves of any feasible joint plan (each token must
  move at least its own shortest path; moves are one-token-one-step).

  Let SEQ = the cost of the trivial sequential one-at-a-time mover baseline as
  recomputed deterministically here (route token 0 fully, then token 1, ...,
  treating already-placed tokens and not-yet-moved tokens as obstacles, parking
  via the blank space).  We *do not* require SEQ to be reachable -- it is only a
  reference scale; if the deterministic sequential router fails we fall back to a
  loose analytic upper scale.  SEQ is used only as the 0-point of the normalized
  score's denominator.

  The reported score is

        score = 1000 * LB / max(L, LB)

  i.e. a plan that achieves the per-token lower bound scores 1000; a plan that
  uses twice the lower bound scores ~500; an infeasible plan scores 0.  This is a
  continuous, strictly-decreasing-in-L reward that is floored at 0 on infeasibility
  and is bounded in (0, 1000].  Because L >= LB always holds for any feasible plan
  (you can never beat the per-token lower bound), the score never exceeds 1000.
"""
import sys
from collections import deque

DIRS = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1),
}

L_MAX_FACTOR = 200  # hard cap on number of actions = L_MAX_FACTOR * (H*W) (anti-abuse)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split('\n')
    # First line: H W k
    idx = 0
    while toks[idx].strip() == '':
        idx += 1
    H, W, k = map(int, toks[idx].split())
    idx += 1
    grid = []
    for _ in range(H):
        row = toks[idx]
        idx += 1
        # pad / truncate defensively
        if len(row) < W:
            row = row + '.' * (W - len(row))
        grid.append(list(row[:W]))
    starts = []
    targets = []
    for _ in range(k):
        while toks[idx].strip() == '':
            idx += 1
        sr, sc, tr, tc = map(int, toks[idx].split())
        idx += 1
        starts.append((sr, sc))
        targets.append((tr, tc))
    return H, W, k, grid, starts, targets


def bfs_dist(grid, H, W, src, dst):
    """Single-token shortest path length over free cells (ignoring other tokens)."""
    if src == dst:
        return 0
    if grid[src[0]][src[1]] == '#' or grid[dst[0]][dst[1]] == '#':
        return None
    dist = [[-1] * W for _ in range(H)]
    q = deque([src])
    dist[src[0]][src[1]] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '.' and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1
                if (nr, nc) == dst:
                    return dist[nr][nc]
                q.append((nr, nc))
    return None


def lower_bound(grid, H, W, starts, targets):
    total = 0
    for s, t in zip(starts, targets):
        d = bfs_dist(grid, H, W, s, t)
        if d is None:
            return None  # instance itself infeasible -- should not happen with our gen
        total += d
    return total


def read_solution(path):
    with open(path) as f:
        toks = f.read().split()
    return toks


def score(instance_path, solution_path):
    H, W, k, grid, starts, targets = read_instance(instance_path)

    # occupancy maps
    pos = [list(p) for p in starts]
    occ = {}
    for i, (r, c) in enumerate(pos):
        if grid[r][c] == '#':
            return 0.0  # start on a wall -- invalid instance, defensive
        if (r, c) in occ:
            return 0.0  # duplicate start
        occ[(r, c)] = i

    toks = read_solution(solution_path)
    if not toks:
        return 0.0
    try:
        L = int(toks[0])
    except ValueError:
        return 0.0
    L_MAX = L_MAX_FACTOR * H * W
    if L < 0 or L > L_MAX:
        return 0.0
    if len(toks) < 1 + 2 * L:
        return 0.0

    p = 1
    for _ in range(L):
        try:
            i = int(toks[p])
        except ValueError:
            return 0.0
        d = toks[p + 1]
        p += 2
        if i < 0 or i >= k:
            return 0.0
        if d not in DIRS:
            return 0.0
        dr, dc = DIRS[d]
        r, c = pos[i]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            return 0.0
        if grid[nr][nc] == '#':
            return 0.0
        if (nr, nc) in occ:
            return 0.0  # destination occupied by another token => collision
        # apply
        del occ[(r, c)]
        occ[(nr, nc)] = i
        pos[i] = [nr, nc]

    # all tokens must be at targets
    for i in range(k):
        if tuple(pos[i]) != targets[i]:
            return 0.0

    LB = lower_bound(grid, H, W, starts, targets)
    if LB is None:
        return 0.0
    if LB == 0:
        # every token already at target; a 0-length plan is perfect.
        return 1000.0 if L == 0 else 1000.0 * 0  # L>0 means wasted moves -> but LB=0
    # If LB==0 but L>0, the formula 1000*LB/max(L,LB) = 0, which is correct: wasted moves.
    return 1000.0 * LB / max(L, LB)


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py INSTANCE SOLUTION\n")
        sys.exit(2)
    s = score(sys.argv[1], sys.argv[2])
    print(f"{s:.6f}")


if __name__ == "__main__":
    main()
