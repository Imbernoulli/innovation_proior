#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE regatta-fleet instance to stdout.

testId 1..10 is a difficulty ladder (grid + fleet size grow with testId).
Everything is seeded by testId only, so generation is deterministic.

Instance format (stdout):
  line 1: N B G w CAP Smax maxMoves
  line 2: B integers -- start_row_1 .. start_row_B (all boats begin at column 0)
  next G lines: col_g rowLo_g rowHi_g   -- gate g's column and its row band,
                listed in an ARBITRARY (not column-sorted) order; a boat
                satisfies gate g the moment it occupies a cell (row,col_g)
                with rowLo_g <= row <= rowHi_g, in any order, at any tick.
"""
import sys, random


def gen_gates(rng, N, G, bw, center):
    lowc, highc = 2, N - 3
    cols = []
    for k in range(G):
        c = lowc if G == 1 else round(lowc + k * (highc - lowc) / (G - 1))
        cols.append(c)
    for k in range(1, len(cols)):
        if cols[k] <= cols[k - 1]:
            cols[k] = cols[k - 1] + 1
    cols[-1] = min(cols[-1], N - 2)
    for k in range(len(cols) - 2, -1, -1):
        if cols[k] >= cols[k + 1]:
            cols[k] = cols[k + 1] - 1

    gates = []
    for c in cols:
        shift = rng.randint(-1, 1)
        lo = center + shift - bw // 2
        lo = max(0, min(N - bw, lo))
        hi = lo + bw - 1
        gates.append((c, lo, hi))
    return gates


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rng = random.Random(9000 + t)

    N = 15 + 2 * t                 # 17 .. 35  grid side (rows == cols)
    B = 6 + (t % 3)                 # cycles 7,8,6
    G = 2 + (t % 3)                 # cycles 3,4,2
    w = 7 + (t % 2)                 # cycles 8,7  (wake decay window, ticks)
    CAP = 8                         # hard cap on the per-move wake multiplier
    bw = max(2, B - 2)               # width of every gate's row band (< B: a couple
                                      # of boats MUST always share a lane somewhere)
    center = N // 2
    Smax = B * (w + 2) + 5
    maxMoves = 8 * N

    # The whole fleet lines up on the SAME starting row (a real start line): no
    # boat has an accidental positional head start, so the only way to spread
    # the fleet apart is to actually reason about lanes and phase, not just
    # "go to wherever I already happen to be".
    starts = [center] * B

    gates_sorted = gen_gates(rng, N, G, bw, center)
    gates_listed = list(reversed(gates_sorted))  # deliberately NOT column-sorted

    lines = [f"{N} {B} {G} {w} {CAP} {Smax} {maxMoves}",
             " ".join(str(x) for x in starts)]
    for (c, lo, hi) in gates_listed:
        lines.append(f"{c} {lo} {hi}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
