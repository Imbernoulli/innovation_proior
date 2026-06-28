#!/usr/bin/env python3
"""Instance generator for ale-44 "Conveyor / Belt Layout".

Usage: python3 gen.py SEED  ->  writes one instance to stdout.

Instance format (stdin of the solver):
    H W                 grid dimensions
    nS nG B T           #sources, #sinks, tile budget, simulation horizon
    nS lines: r c d     a source at (r,c) emitting in direction d (0=R,1=D,2=L,3=U)
    nG lines: r c       a sink at (r,c)

All source/sink cells are pairwise distinct. Grid is 0-indexed:
row r in [0,H), col c in [0,W). Direction codes: 0=Right(+c), 1=Down(+r),
2=Left(-c), 3=Up(-r).
"""
import sys, random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rng = random.Random(seed * 1_000_003 + 12345)

    # Grid size grows mildly with seed so the seed set spans small..medium.
    H = rng.randint(20, 30)
    W = rng.randint(20, 30)
    cells = H * W

    nS = rng.randint(16, 26)         # many sources -> routing is contested
    nG = rng.randint(2, 4)           # few sinks -> long, mergeable trunks
    # Budget is TIGHT: laying every source on its own private shortest path
    # would cost far more than B, so the only way to route many sources is to
    # MERGE their belts onto shared trunks heading to a sink.  This is what
    # makes orientation search (not per-source greedy) the lever.
    B = rng.randint(int(cells * 0.12), int(cells * 0.18))
    T = H * W * 2                    # generous horizon (long enough for any path)

    # Sample distinct cells for sources and sinks.
    all_cells = [(r, c) for r in range(H) for c in range(W)]
    rng.shuffle(all_cells)
    chosen = all_cells[:nS + nG]
    src_cells = chosen[:nS]
    sink_cells = chosen[nS:nS + nG]

    out = []
    out.append(f"{H} {W}")
    out.append(f"{nS} {nG} {B} {T}")
    for (r, c) in src_cells:
        d = rng.randint(0, 3)
        out.append(f"{r} {c} {d}")
    for (r, c) in sink_cells:
        out.append(f"{r} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
