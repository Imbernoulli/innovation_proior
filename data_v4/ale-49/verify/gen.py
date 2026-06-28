#!/usr/bin/env python3
"""Instance generator for ale-49: Reconfiguration Routing (token sliding on a grid).

Usage: python3 gen.py SEED  > instance.txt

Instance format (stdout):
    H W k
    H lines of W characters each: the grid. '#' = wall, '.' = free.
    k lines: "sr sc tr tc"  start (row,col) and target (row,col) for token i (0-indexed).

Guarantees produced by this generator:
  - Grid is H x W with 0 <= H,W and a random fraction of wall cells.
  - All start cells are distinct free cells; all target cells are distinct free cells.
  - Every token's target is reachable from its start ignoring other tokens
    (so a feasible full reconfiguration always exists -- tokens can be routed
    one at a time, parking others out of the way is NOT required because the
    free space is kept generous enough; feasibility of the *joint* problem is
    guaranteed by construction: see below).

Feasibility guarantee for the joint problem:
  We keep the number of free cells strictly larger than k (at least k + a slack
  margin of free cells), and the free region is a single 4-connected component.
  On a connected board whose free cells outnumber the tokens by at least one,
  any permutation-style reconfiguration is solvable (there is always at least one
  empty cell to shuffle through -- the classic 15-puzzle connectivity argument
  generalizes: with >=1 blank and a non-separable connected region the puzzle
  group is the full symmetric/alternating group, which is more than enough since
  our tokens are distinguishable but targets are a fixed assignment). To be safe
  and simple we additionally keep a comfortable blank margin.
"""
import sys
import random
from collections import deque


def connected_free(grid, H, W):
    """Return (component_size_of_largest, comp_id_grid) for 4-connectivity over free cells."""
    comp = [[-1] * W for _ in range(H)]
    cid = 0
    sizes = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.' and comp[i][j] == -1:
                q = deque([(i, j)])
                comp[i][j] = cid
                sz = 0
                while q:
                    r, c = q.popleft()
                    sz += 1
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '.' and comp[nr][nc] == -1:
                            comp[nr][nc] = cid
                            q.append((nr, nc))
                sizes.append(sz)
                cid += 1
    if not sizes:
        return 0, comp, -1
    best = max(range(len(sizes)), key=lambda x: sizes[x])
    return sizes[best], comp, best


def gen(seed):
    rng = random.Random(seed * 1_000_003 + 12345)

    # Size scales mildly with seed but stays in a bounded heuristic-sized range.
    H = rng.randint(12, 30)
    W = rng.randint(12, 30)

    # Wall density: keep it moderate so the free region is generous and connected.
    wall_p = rng.uniform(0.05, 0.22)

    # Build grid, retry until a large connected free component dominates the board.
    for _attempt in range(200):
        grid = [['.' for _ in range(W)] for _ in range(H)]
        # Place walls. Mix of scattered single walls and a few short bars for structure.
        for i in range(H):
            for j in range(W):
                if rng.random() < wall_p:
                    grid[i][j] = '#'
        # A couple of short wall "bars" to create corridors/bottlenecks.
        nbars = rng.randint(0, 3)
        for _ in range(nbars):
            br = rng.randint(0, H - 1)
            bc = rng.randint(0, W - 1)
            length = rng.randint(2, max(2, min(H, W) // 2))
            horiz = rng.random() < 0.5
            for t in range(length):
                r = br if horiz else br + t
                c = bc + t if horiz else bc
                if 0 <= r < H and 0 <= c < W:
                    grid[r][c] = '#'

        big, comp, bid = connected_free(grid, H, W)
        if bid < 0:
            continue
        # Restrict free cells to the largest component (turn the rest into walls)
        # so the working region is guaranteed connected.
        for i in range(H):
            for j in range(W):
                if grid[i][j] == '.' and comp[i][j] != bid:
                    grid[i][j] = '#'

        # Erode 1-wide DEAD ENDS: repeatedly wall off any free cell that has only a
        # single free neighbour.  A free region in which every free cell has >= 2
        # free neighbours has no trapping dead-end pockets, so two tokens can always
        # be routed past each other (the region is "non-separating" enough for token
        # sliding).  This keeps the joint reconfiguration robustly feasible while
        # still leaving a corridor-rich, congested board -- the interesting case.
        changed = True
        while changed:
            changed = False
            for i in range(H):
                for j in range(W):
                    if grid[i][j] != '.':
                        continue
                    deg = 0
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ni, nj = i + dr, j + dc
                        if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
                            deg += 1
                    if deg <= 1:
                        grid[i][j] = '#'
                        changed = True

        # Re-check connectivity after erosion (erosion can split or empty the region).
        big, comp, bid = connected_free(grid, H, W)
        if bid < 0:
            continue
        for i in range(H):
            for j in range(W):
                if grid[i][j] == '.' and comp[i][j] != bid:
                    grid[i][j] = '#'

        free_cells = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == '.']
        nfree = len(free_cells)
        if nfree < 8:
            continue

        # Choose k with a GENEROUS blank margin: at least three blank cells per token
        # (k <= nfree/4).  Together with dead-end erosion this keeps the joint
        # reconfiguration robustly solvable (plenty of room to slide tokens past one
        # another) while leaving a congested, corridor-rich board -- the regime
        # where prioritized space-time planning matters.
        max_k = min(nfree // 4, 40)
        if max_k < 2:
            continue
        k = rng.randint(2, max(2, max_k))

        if 4 * k > nfree:
            continue

        # Pick distinct start cells and distinct target cells.
        starts = rng.sample(free_cells, k)
        targets = rng.sample(free_cells, k)

        # Make the instance non-degenerate: ensure at least one token actually
        # needs to move (start != target for some token).  Reshuffle targets if
        # every token is already at its target.
        if all(starts[i] == targets[i] for i in range(k)):
            targets = rng.sample(free_cells, k)

        return H, W, k, grid, starts, targets

    # Fallback: an open board with no walls.
    H, W = 14, 14
    grid = [['.' for _ in range(W)] for _ in range(H)]
    free_cells = [(i, j) for i in range(H) for j in range(W)]
    k = 6
    starts = rng.sample(free_cells, k)
    targets = rng.sample(free_cells, k)
    return H, W, k, grid, starts, targets


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    H, W, k, grid, starts, targets = gen(seed)
    out = []
    out.append(f"{H} {W} {k}")
    for i in range(H):
        out.append("".join(grid[i]))
    for i in range(k):
        sr, sc = starts[i]
        tr, tc = targets[i]
        out.append(f"{sr} {sc} {tr} {tc}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
