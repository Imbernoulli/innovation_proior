#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance ("n M") to stdout.

testId 1..10 is a difficulty ladder: the tower budget n grows from 64 to 200.
All sizes are multiples of 8 (so clean structured placements exist), and the
mile-post range M = 4*n*n leaves ample room for both Sidon-type and
more-sums-than-differences layouts. Fully deterministic in testId.
"""
import sys

SIZES = [64, 80, 96, 112, 128, 144, 160, 176, 192, 200]


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tid < 1:
        tid = 1
    if tid > len(SIZES):
        tid = len(SIZES)
    n = SIZES[tid - 1]
    M = 4 * n * n
    print(f"{n} {M}")


if __name__ == "__main__":
    main()
