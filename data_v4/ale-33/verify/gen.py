#!/usr/bin/env python3
"""
Instance generator for ale-33: Grid Wire Routing (disjoint rectilinear paths).

Usage: python3 gen.py <seed>   ->  writes one instance to stdout.

An instance:
  - A grid of H rows x W columns.  Cells are (r, c) with 0 <= r < H, 0 <= c < W.
  - n terminal PAIRS.  Pair k has endpoints (r1,c1) and (r2,c2), all distinct
    cells (no cell is shared by two endpoints, and a pair's two endpoints differ).
  - The task (solver side): connect each pair by a 4-connected rectilinear path
    (a sequence of cells, each consecutive pair differing by exactly one in r or
    c) so that NO grid cell is used by more than one wire (the paths are
    vertex-disjoint, including endpoints), minimising the total number of edges
    (sum of path lengths).

GUARANTEED FEASIBILITY.  We build the instance by first carving n vertex-disjoint
rectilinear paths into an empty grid via randomized self-avoiding walks, then we
hand the solver ONLY the endpoints.  Because a disjoint routing demonstrably
exists (the one we carved), the instance is always routable and the
feasibility->0 floor is meaningful: a valid output always exists.

Output format (stdin of the solver), whitespace-separated tokens:
    H W n
    r1 c1 r2 c2          (pair 0)
    r1 c1 r2 c2          (pair 1)
    ...
    r1 c1 r2 c2          (pair n-1)
All coordinates are 0-based.
"""
import sys
import random


def carve_paths(H, W, n, rng):
    """Carve n vertex-disjoint rectilinear paths by randomized self-avoiding
    walks.  Returns a list of endpoint quadruples, or None if it failed to place
    all n (caller retries with a fresh attempt)."""
    occ = [[-1] * W for _ in range(H)]  # -1 free, else owner wire id
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    pairs = []

    def free_cells():
        return [(r, c) for r in range(H) for c in range(W) if occ[r][c] == -1]

    for wid in range(n):
        # pick a random free start
        fc = free_cells()
        if not fc:
            return None
        placed = False
        # several start attempts for this wire
        for _attempt in range(40):
            sr, sc = rng.choice(fc)
            if occ[sr][sc] != -1:
                continue
            # target walk length: a meandering walk so paths are non-trivial
            target = rng.randint(2, max(2, (H + W)))
            path = [(sr, sc)]
            occ[sr][sc] = wid
            cur = (sr, sc)
            for _step in range(target):
                r, c = cur
                cand = []
                order = DIRS[:]
                rng.shuffle(order)
                for dr, dc in order:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and occ[nr][nc] == -1:
                        cand.append((nr, nc))
                if not cand:
                    break
                nxt = cand[0]
                occ[nxt[0]][nxt[1]] = wid
                path.append(nxt)
                cur = nxt
            if len(path) >= 1:
                a = path[0]
                b = path[-1]
                if a == b:
                    # length-1 walk: endpoints coincide; release and retry start
                    for (r, c) in path:
                        occ[r][c] = -1
                    continue
                pairs.append((a[0], a[1], b[0], b[1]))
                placed = True
                break
            else:
                for (r, c) in path:
                    occ[r][c] = -1
        if not placed:
            return None
    return pairs


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(seed * 1000003 + 777)

    # Size varies a little with the seed so the seed set spans regimes, but is
    # bounded so the solver stays fast and the grid is congested enough to make
    # routing non-trivial.
    H = rng.randint(18, 30)
    W = rng.randint(18, 30)
    # number of pairs scaled to the grid: dense enough that naive routing
    # collides, sparse enough that a disjoint routing exists.
    cells = H * W
    n = rng.randint(max(6, cells // 14), max(8, cells // 9))

    pairs = None
    for _try in range(200):
        rng2 = random.Random(rng.randint(0, 10**9))
        pairs = carve_paths(H, W, n, rng2)
        if pairs is not None and len(pairs) == n:
            break
        pairs = None
    if pairs is None:
        # Fallback: fewer pairs that we are sure we can place (very rare).
        for nn in range(n, 0, -1):
            for _try in range(200):
                rng2 = random.Random(rng.randint(0, 10**9))
                pr = carve_paths(H, W, nn, rng2)
                if pr is not None and len(pr) == nn:
                    pairs = pr
                    n = nn
                    break
            if pairs is not None:
                break

    out = [f"{H} {W} {n}"]
    for (r1, c1, r2, c2) in pairs:
        out.append(f"{r1} {c1} {r2} {c2}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
