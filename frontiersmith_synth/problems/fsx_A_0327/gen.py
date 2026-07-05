#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of the "mountain rescue relay" set problem.

Instance format (stdout), a single line:
    n V
where n = number of relay beacons to place, and V = inclusive upper altitude bound.
Beacons must be placed at DISTINCT integers in [0, V].

testId 1..10 is a difficulty ladder (increasing beacon budget n). The window V is set
generously (V = 8*n*n) so that a fully difference-spread (Sidon) layout always fits; the
difficulty is intrinsic to the additive-combinatorics objective, not to the window.
All randomness (there is none here) would be seeded by testId only.
"""
import sys

# difficulty ladder: number of beacons grows with testId (large-scale family instance)
N_LADDER = [20, 24, 28, 32, 40, 48, 56, 64, 72, 80]


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(2)
    t = int(sys.argv[1])
    if t < 1:
        t = 1
    idx = (t - 1) % len(N_LADDER)
    n = N_LADDER[idx]
    V = 8 * n * n
    sys.stdout.write("%d %d\n" % (n, V))


if __name__ == "__main__":
    main()
