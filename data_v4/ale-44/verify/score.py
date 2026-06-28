#!/usr/bin/env python3
"""Deterministic local scorer for ale-44 "Conveyor / Belt Layout".

Usage:
    python3 score.py INSTANCE_FILE SOLUTION_FILE
prints a single integer: the score (number of delivered items), or 0 if the
solution is infeasible.

Solution format (stdout of the solver):
    K                   number of placed belt tiles
    K lines: r c d      a belt tile at (r,c) pointing in direction d
                        (0=R,1=D,2=L,3=U)

Feasibility -> 0 floor.  The output is INFEASIBLE (score forced to 0) if:
  * it is malformed / non-integer / truncated;
  * K < 0 or K > B (budget exceeded);
  * any tile cell is out of the grid;
  * any tile is placed on a source or a sink cell (illegal);
  * two tiles are placed on the same cell (overlap);
  * any direction code is not in {0,1,2,3}.

Score (when feasible): build the per-cell "next direction" map (sources use
their emission direction, belt tiles use their tile direction). Each source
emits one item; the item follows cell directions one step per tick. The item
is DELIVERED iff it reaches a sink cell within T ticks; it is LOST if it
leaves the grid, enters an empty cell with no tile, or fails to reach a sink
within T ticks (e.g. it loops). Score = number of delivered items.
"""
import sys

# direction code -> (dr, dc).  0=Right,1=Down,2=Left,3=Up
DR = [0, 1, 0, -1]
DC = [1, 0, -1, 0]


def read_ints(path):
    with open(path) as f:
        return f.read().split()


def parse_instance(tokens):
    it = iter(tokens)
    H = int(next(it)); W = int(next(it))
    nS = int(next(it)); nG = int(next(it)); B = int(next(it)); T = int(next(it))
    sources = []
    for _ in range(nS):
        r = int(next(it)); c = int(next(it)); d = int(next(it))
        sources.append((r, c, d))
    sinks = []
    for _ in range(nG):
        r = int(next(it)); c = int(next(it))
        sinks.append((r, c))
    return H, W, nS, nG, B, T, sources, sinks


def main():
    inst_tokens = read_ints(sys.argv[1])
    H, W, nS, nG, B, T, sources, sinks = parse_instance(inst_tokens)

    # Cell role maps.
    src_dir = {}                       # (r,c) -> emission direction
    for (r, c, d) in sources:
        src_dir[(r, c)] = d
    sink_set = set((r, c) for (r, c) in sinks)
    reserved = set(src_dir.keys()) | sink_set

    # Parse solution; any malformedness => infeasible => 0.
    try:
        sol_tokens = read_ints(sys.argv[2])
        it = iter(sol_tokens)
        K = int(next(it))
        if K < 0 or K > B:
            print(0); return
        tile_dir = {}
        for _ in range(K):
            r = int(next(it)); c = int(next(it)); d = int(next(it))
            if not (0 <= r < H and 0 <= c < W):
                print(0); return
            if d < 0 or d > 3:
                print(0); return
            if (r, c) in reserved:          # tile on a source/sink -> illegal
                print(0); return
            if (r, c) in tile_dir:          # overlap -> illegal
                print(0); return
            tile_dir[(r, c)] = d
        try:
            next(it)                         # extra output is malformed
            print(0); return
        except StopIteration:
            pass
    except (StopIteration, ValueError, IndexError):
        print(0); return

    # Build the per-cell next-direction map.
    def cell_dir(r, c):
        if (r, c) in src_dir:
            return src_dir[(r, c)]
        if (r, c) in tile_dir:
            return tile_dir[(r, c)]
        return None                          # sink handled separately; empty -> None

    # Simulate each source's item independently (no collisions).
    delivered = 0
    for (sr, sc, sd) in sources:
        r, c = sr, sc
        d = sd                               # leave the source in its emission dir
        ok = False
        for _ in range(T):
            nr, nc = r + DR[d], c + DC[d]
            if not (0 <= nr < H and 0 <= nc < W):
                break                        # left the grid -> lost
            if (nr, nc) in sink_set:
                ok = True; break             # delivered
            nd = cell_dir(nr, nc)
            if nd is None:
                break                        # empty cell, no tile -> lost
            r, c, d = nr, nc, nd
        if ok:
            delivered += 1

    print(delivered)


if __name__ == "__main__":
    main()
