#!/usr/bin/env python3
"""Deterministic local scorer for "Sokoban-Style Box Pushing".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score. HIGHER is better. INFEASIBLE -> 0.

Scoring rule (see context.md "Evaluation settings"):

  * Instance (read from <instance_file>):
        H W S
        row_0 .. row_{H-1}          (each row W chars over '#','.','o','@','+','$','*')
    A single agent on a grid with PUSH-ONLY Sokoban mechanics, a budget of S
    moves, B boxes and exactly B target cells.

  * SOLUTION format (read from <solution_file>): a single move string -- a
    sequence of characters in {U,D,L,R} (whitespace/newlines between them are
    ignored; an EMPTY string is allowed and means "do nothing"). U=up (y-1),
    D=down (y+1), L=left (x-1), R=right (x+1).

  * REPLAY (exact push rules). Start from the board's agent/box/target layers.
    For each move in order, let (ny,nx) be the agent's destination:
      - if (ny,nx) is a wall -> ILLEGAL move;
      - else if (ny,nx) holds a box, let (by,bx) be the cell one further in the
        same direction; the push is legal only if (by,bx) is inside the grid,
        is NOT a wall, and does NOT already hold a box. If the push is legal the
        box moves to (by,bx) and the agent moves to (ny,nx). Otherwise -> ILLEGAL;
      - else (empty floor or target) the agent simply moves to (ny,nx).
    Any character outside {U,D,L,R} (after stripping whitespace) -> ILLEGAL.
    Using MORE than S moves -> ILLEGAL.

  * FEASIBILITY (any violation -> score 0):
      - the move string contains only {U,D,L,R} (whitespace ignored);
      - its length is <= S;
      - every move in the replay is LEGAL (no move into a wall, no illegal push).
    An illegal move floors the WHOLE solution to 0 (it is rejected outright, not
    truncated).

  * RAW VALUE of a feasible solution: replay the moves, then count the number of
    boxes resting on target cells at the end. Call this `parked`.

  * SCORE (higher better), normalized against a deterministic baseline the scorer
    recomputes itself -- a GREEDY nearest-box-to-nearest-target pusher (see
    greedy_baseline): for the same instance it parks `base` boxes (and starts
    with `start` boxes already parked). The score is
        score = round(1_000_000 * (1 + parked) / (1 + base))      (0 if INFEASIBLE)
    so an empty/feasible move string that matches the starting parked count scores
    on the order of the baseline, the greedy baseline itself scores ~1_000_000,
    and a solver that parks strictly more boxes than greedy scores strictly more.
    The (1 + .) offsets keep the ratio well-defined when base == 0 and reward the
    very first extra box. INFEASIBLE -> 0.
"""
import sys

DIRS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}


# ----------------------------------------------------------------------------- IO
def read_instance(path):
    with open(path) as f:
        lines = f.read().split('\n')
    first = lines[0].split()
    H = int(first[0]); W = int(first[1]); S = int(first[2])
    rows = []
    for i in range(1, H + 1):
        rows.append(lines[i])
    # decompose into layers
    wall = [[False] * W for _ in range(H)]
    target = [[False] * W for _ in range(H)]
    box = [[False] * W for _ in range(H)]
    ay = ax = -1
    for y in range(H):
        row = rows[y]
        for x in range(W):
            c = row[x]
            if c == '#':
                wall[y][x] = True
            elif c == 'o':
                target[y][x] = True
            elif c == '@':
                ay, ax = y, x
            elif c == '+':
                target[y][x] = True
                ay, ax = y, x
            elif c == '$':
                box[y][x] = True
            elif c == '*':
                target[y][x] = True
                box[y][x] = True
            # '.' -> nothing
    return H, W, S, wall, target, box, ay, ax


def read_solution(path):
    try:
        with open(path) as f:
            raw = f.read()
    except OSError:
        return None
    # strip all whitespace; remaining chars must be in {U,D,L,R}.
    moves = ''.join(raw.split())
    for c in moves:
        if c not in DIRS:
            return None
    return moves


# -------------------------------------------------------------------------- replay
def replay(H, W, S, wall, target, box, ay, ax, moves):
    """Return number of boxes on targets at the end, or None if any move illegal."""
    if len(moves) > S:
        return None
    # work on a mutable copy of the box layer
    box = [row[:] for row in box]
    for c in moves:
        dy, dx = DIRS[c]
        ny, nx = ay + dy, ax + dx
        if ny < 0 or ny >= H or nx < 0 or nx >= W:
            return None
        if wall[ny][nx]:
            return None
        if box[ny][nx]:
            by, bx = ny + dy, nx + dx
            if by < 0 or by >= H or bx < 0 or bx >= W:
                return None
            if wall[by][bx] or box[by][bx]:
                return None
            box[ny][nx] = False
            box[by][bx] = True
            ay, ax = ny, nx
        else:
            ay, ax = ny, nx
    parked = 0
    for y in range(H):
        for x in range(W):
            if box[y][x] and target[y][x]:
                parked += 1
    return parked


def count_parked(H, W, target, box):
    p = 0
    for y in range(H):
        for x in range(W):
            if box[y][x] and target[y][x]:
                p += 1
    return p


# ------------------------------------------------- baseline: greedy nearest pusher
def _bfs_dist(H, W, wall, blocked, src):
    """BFS distances from src over floor cells (wall or blocked cells impassable)."""
    from collections import deque
    INF = float('inf')
    dist = [[INF] * W for _ in range(H)]
    sy, sx = src
    if wall[sy][sx] or blocked[sy][sx]:
        return dist
    dist[sy][sx] = 0
    q = deque([(sy, sx)])
    while q:
        y, x = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not wall[ny][nx] and not blocked[ny][nx] and dist[ny][nx] == INF:
                dist[ny][nx] = dist[y][x] + 1
                q.append((ny, nx))
    return dist


def greedy_baseline(H, W, S, wall, target, box, ay, ax):
    """A greedy nearest-box-to-nearest-target pusher.

    Repeatedly: pick the (box, target) pair, over boxes not yet on a target and
    free targets, that is closest in Manhattan terms; push that box toward that
    target one axis at a time, repositioning the agent behind the box each push
    via BFS over floor cells (treating boxes as obstacles). A push is committed
    only if it is legal and the agent can actually reach the required standing
    cell within the remaining budget; otherwise that pair is abandoned. Returns
    the number of boxes parked when the move budget runs out or no progress is
    possible. This is the deterministic normalizer -- the solver must beat it.
    """
    box = [row[:] for row in box]
    targets = [(y, x) for y in range(H) for x in range(W) if target[y][x]]
    budget = S
    moved = True
    while moved and budget > 0:
        moved = False
        # boxes not currently on a target
        loose = [(y, x) for y in range(H) for x in range(W) if box[y][x] and not target[y][x]]
        free_t = [(ty, tx) for (ty, tx) in targets if not box[ty][tx]]
        if not loose or not free_t:
            break
        # choose the closest box->target pair by Manhattan distance
        best = None
        for (by, bx) in loose:
            for (ty, tx) in free_t:
                d = abs(by - ty) + abs(bx - tx)
                if best is None or d < best[0]:
                    best = (d, by, bx, ty, tx)
        _, by, bx, ty, tx = best
        # push this box toward target, one axis-aligned step at a time
        progressed = False
        guard = 0
        while (by, bx) != (ty, tx) and budget > 0 and guard < 4 * H * W:
            guard += 1
            # decide a direction that reduces distance and is push-legal
            cand = []
            if by < ty:
                cand.append((1, 0))
            elif by > ty:
                cand.append((-1, 0))
            if bx < tx:
                cand.append((0, 1))
            elif bx > tx:
                cand.append((0, -1))
            did = False
            for (dy, dx) in cand:
                nby, nbx = by + dy, bx + dx
                if not (0 <= nby < H and 0 <= nbx < W):
                    continue
                if wall[nby][nbx] or box[nby][nbx]:
                    continue
                # agent must stand on the cell opposite the push direction
                sy, sx = by - dy, bx - dx
                if not (0 <= sy < H and 0 <= sx < W) or wall[sy][sx] or box[sy][sx]:
                    continue
                # can the agent reach (sy,sx) over floor (boxes block)?
                blocked = [[box[yy][xx] for xx in range(W)] for yy in range(H)]
                dist = _bfs_dist(H, W, wall, blocked, (ay, ax))
                walk = dist[sy][sx]
                if walk == float('inf'):
                    continue
                if walk + 1 > budget:
                    continue
                # commit: walk (cost `walk`) + 1 push
                budget -= (int(walk) + 1)
                box[by][bx] = False
                box[nby][nbx] = True
                ay, ax = by, bx           # agent ends where the box was
                by, bx = nby, nbx
                did = True
                progressed = True
                moved = True
                break
            if not did:
                break
        if progressed:
            continue
    return count_parked(H, W, target, box)


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    H, W, S, wall, target, box, ay, ax = read_instance(sys.argv[1])

    moves = read_solution(sys.argv[2])
    if moves is None:
        print(0)
        return
    parked = replay(H, W, S, wall, target, box, ay, ax, moves)
    if parked is None:
        print(0)
        return

    base = greedy_baseline(H, W, S, wall, target, box, ay, ax)
    score = int(round(1_000_000.0 * (1 + parked) / (1 + base)))
    print(score)


if __name__ == "__main__":
    main()
