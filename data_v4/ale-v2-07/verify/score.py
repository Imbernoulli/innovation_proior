#!/usr/bin/env python3
"""
Deterministic local scorer for "Grid Wire Routing" (ale-v2-07).

Usage:  python3 score.py INSTANCE_FILE SOLUTION_FILE
        (or)  python3 score.py INSTANCE_FILE   < SOLUTION_FILE

Score = number of nets validly routed, with vertex-disjoint paths.
ANY feasibility violation in the submitted output floors the whole score to 0.

A submission is FEASIBLE iff:
  * The first token R is an integer with 0 <= R <= K.
  * It lists exactly R routed nets, each a distinct net index in [0, K).
  * For each listed net i with path length L and cells p_0..p_{L-1}:
      - L >= 1, and all cells are inside the grid and are '.' (not '#');
      - p_0 equals one terminal of net i and p_{L-1} equals the other terminal
        of net i (either orientation is accepted);
      - consecutive cells are orthogonally adjacent (4-neighbour);
      - the path visits no cell twice (a simple path);
  * Across ALL listed nets, no grid cell is used by more than one net
    (vertex-disjoint => paths neither overlap nor cross).

If every check passes, the score is R (the count of routed nets).
Otherwise the score is 0.

The script prints a single integer (the score) to stdout. With -v it also
prints a human-readable reason on stderr when it floors to 0.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split('\n')
    # First non-empty line: H W K
    it = iter(toks)
    header = next(it).split()
    H, W, K = int(header[0]), int(header[1]), int(header[2])
    grid = []
    for _ in range(H):
        grid.append(next(it))
    nets = []
    for _ in range(K):
        r1, c1, r2, c2 = map(int, next(it).split())
        nets.append(((r1, c1), (r2, c2)))
    return H, W, K, grid, nets


def score(instance_path, sol_text, verbose=False):
    H, W, K, grid, nets = read_instance(instance_path)

    def blocked(r, c):
        return not (0 <= r < H and 0 <= c < W) or grid[r][c] != '.'

    def fail(msg):
        if verbose:
            sys.stderr.write("INFEASIBLE: " + msg + "\n")
        return 0

    toks = sol_text.split()
    if not toks:
        return fail("empty output")
    pos = 0

    def nxt():
        nonlocal pos
        if pos >= len(toks):
            raise IndexError
        v = toks[pos]
        pos += 1
        return v

    try:
        R = int(nxt())
    except (IndexError, ValueError):
        return fail("missing/invalid R")
    if R < 0 or R > K:
        return fail(f"R={R} out of range [0,{K}]")

    used = {}            # cell -> net index that owns it
    seen_nets = set()

    for _ in range(R):
        try:
            i = int(nxt())
            L = int(nxt())
        except (IndexError, ValueError):
            return fail("truncated net header")
        if not (0 <= i < K):
            return fail(f"net index {i} out of range")
        if i in seen_nets:
            return fail(f"net {i} listed twice")
        seen_nets.add(i)
        if L < 1:
            return fail(f"net {i} path length {L} < 1")
        path = []
        try:
            for _k in range(L):
                r = int(nxt()); c = int(nxt())
                path.append((r, c))
        except (IndexError, ValueError):
            return fail(f"net {i} truncated path")

        # cells valid and free
        local = set()
        for (r, c) in path:
            if blocked(r, c):
                return fail(f"net {i} cell ({r},{c}) blocked/out-of-grid")
            if (r, c) in local:
                return fail(f"net {i} revisits cell ({r},{c})")
            local.add((r, c))

        # adjacency
        for k in range(1, L):
            (r0, c0), (r1, c1) = path[k - 1], path[k]
            if abs(r0 - r1) + abs(c0 - c1) != 1:
                return fail(f"net {i} non-adjacent step {path[k-1]}->{path[k]}")

        # endpoints match the net's terminals (either orientation)
        a, b = nets[i]
        ends = (path[0], path[-1])
        if not (ends == (a, b) or ends == (b, a)):
            return fail(f"net {i} endpoints {ends} != terminals {(a,b)}")

        # vertex-disjoint across nets
        for cell in path:
            if cell in used and used[cell] != i:
                return fail(f"net {i} shares cell {cell} with net {used[cell]}")
            used[cell] = i

    return R


if __name__ == '__main__':
    verbose = '-v' in sys.argv
    args = [a for a in sys.argv[1:] if a != '-v']
    instance_path = args[0]
    if len(args) >= 2:
        with open(args[1]) as f:
            sol_text = f.read()
    else:
        sol_text = sys.stdin.read()
    print(score(instance_path, sol_text, verbose=verbose))
