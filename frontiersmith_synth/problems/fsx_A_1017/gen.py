#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance (a single integer n) to stdout.

The instance asks for a Temperature-2 abstract Tile Assembly Model (aTAM) tile
set that self-assembles, from a fixed seed, the W x n rectangle (W = number of
bits needed for 0..n-1) whose cell (row=bit b, col=counter value c) must carry
"value" c>>b & 1 -- i.e. column c, read bottom-to-top, is the binary
representation of c. Difficulty ladder: n increases with testId (more counter
values to assemble => a wider gap between a per-cell tiling and a cooperative
binary-counter tile set). Fully deterministic: testId -> n via a fixed table.
"""
import sys

# n is chosen so that even a maximally tile-efficient (O(1)) cooperative
# construction does not saturate the checker's ratio cap on these instances,
# while a per-cell tiling already costs n*W tile types (fast-growing gap).
LADDER = {
    1: 6,
    2: 7,
    3: 8,
    4: 9,
    5: 10,
    6: 12,
    7: 13,
    8: 14,
    9: 15,
    10: 16,
}


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    tid = int(sys.argv[1])
    n = LADDER.get(tid)
    if n is None:
        # graceful fallback for any tid beyond the fixed ladder (kept
        # deterministic and inside the same safe, non-saturating range)
        n = 6 + ((tid - 1) * 2) % 11
    print(n)


if __name__ == "__main__":
    main()
