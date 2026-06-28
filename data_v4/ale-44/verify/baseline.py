#!/usr/bin/env python3
"""Trivial baseline for ale-44: STRAIGHT-LINE belts.

For each source, lay a straight run of belt tiles in the source's emission
direction (each tile keeps the same direction) until the run would leave the
grid or hit a sink/source.  Tiles are shared if a cell is reused with the same
direction; conflicting cells (already tiled a different way, or reserved) stop
that source's run.  Budget B is respected by laying sources in input order and
stopping when the budget is exhausted.

Emits the solution on stdout in the scorer's format.
"""
import sys

DR = [0, 1, 0, -1]
DC = [1, 0, -1, 0]


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    H = int(next(it)); W = int(next(it))
    nS = int(next(it)); nG = int(next(it)); B = int(next(it)); T = int(next(it))
    sources = [(int(next(it)), int(next(it)), int(next(it))) for _ in range(nS)]
    sinks = [(int(next(it)), int(next(it))) for _ in range(nG)]
    reserved = set((r, c) for (r, c, _) in sources) | set(sinks)
    sink_set = set(sinks)

    tile = {}                       # (r,c) -> dir
    used = 0
    for (sr, sc, sd) in sources:
        r, c, d = sr, sc, sd
        while True:
            nr, nc = r + DR[d], c + DC[d]
            if not (0 <= nr < H and 0 <= nc < W):
                break
            if (nr, nc) in sink_set:
                break               # delivered, no tile needed on the sink
            if (nr, nc) in reserved:
                break               # blocked by another source
            if (nr, nc) in tile:
                if tile[(nr, nc)] != d:
                    break           # conflicting tile -> stop this run
                # reuse
            else:
                if used >= B:
                    break
                tile[(nr, nc)] = d
                used += 1
            r, c = nr, nc

    out = [str(len(tile))]
    for (r, c), d in tile.items():
        out.append(f"{r} {c} {d}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
