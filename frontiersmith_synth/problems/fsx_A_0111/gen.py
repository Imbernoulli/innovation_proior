#!/usr/bin/env python3
"""gen.py <testId>  ->  prints ONE instance to stdout.

Difficulty ladder (small scale): testId 1..10 maps grid side m = 6,8,...,24.
Obstructed pads are seeded deterministically by testId only. Row 0 and column 0
are NEVER obstructed, so the baseline border line (and the pads (0,0),(1,0),(0,1))
are always available.
"""
import sys
import random


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    if t < 1:
        t = 1
    if t > 10:
        t = 10

    m = 4 + 2 * t            # t=1 -> 6, ..., t=10 -> 24
    rng = random.Random(1000 + 7 * t)

    # obstruction density grows mildly with the ladder
    frac = 0.06 + 0.010 * (t - 1)          # ~6% .. ~15%
    interior = (m - 1) * (m - 1)           # cells with r>=1 and c>=1
    b = int(round(frac * interior))
    b = max(0, min(b, interior))

    cells = [(r, c) for r in range(1, m) for c in range(1, m)]
    rng.shuffle(cells)
    blocked = sorted(cells[:b])

    out = [str(m), str(len(blocked))]
    for (r, c) in blocked:
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
