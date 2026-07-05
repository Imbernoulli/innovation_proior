#!/usr/bin/env python3
"""Instance generator for the coral-reef survey low-discrepancy point-set problem.

`python3 gen.py <testId>` prints ONE instance to stdout.
testId 1..N is a difficulty ladder: larger testId -> more stations to place.
All randomness is seeded ONLY by testId, so instances are reproducible.

Instance format (stdout):
    line 1: "d M K"   with d=2 (the reef is mapped to the unit square [0,1]^2),
                       M = total number of survey stations desired,
                       K = number of stations already fixed (pre-installed buoys).
    next K lines: "x y" the coordinates of the K fixed stations (in [0,1]^2).
"""
import sys


def lcg(seed):
    """Deterministic 64-bit LCG yielding floats in [0,1)."""
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        yield ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t < 1:
        t = 1
    d = 2
    M = 12 + 4 * t              # 16, 20, ..., up to 52 at t=10
    K = max(2, M // 6)          # a handful of pre-installed buoys
    rng = lcg(1000003 * t + 7)

    out = ["%d %d %d" % (d, M, K)]
    for _ in range(K):
        # snap fixed stations to a fine grid so coordinates are clean decimals
        x = round(next(rng), 6)
        y = round(next(rng), 6)
        out.append("%.6f %.6f" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
