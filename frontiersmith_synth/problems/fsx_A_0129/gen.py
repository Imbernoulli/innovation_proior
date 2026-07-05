#!/usr/bin/env python3
"""Instance generator for the aquarium-floor probe placement problem.

Usage:  python3 gen.py <testId>
Prints ONE instance to stdout. testId 1..10 is a difficulty ladder:
the number M of probe fittings grows (larger tile => more fittings to place
uniformly, a bigger search space). The dimension is fixed at d=2 (the
rectangular aquarium floor). Everything is a deterministic function of testId.

Instance format (stdout):
    d M
where d is the number of coordinate axes (2) and M is the number of probe
fittings that MUST be placed.
"""
import sys

# probe counts per test id (difficulty ladder). Values are a mix of non-square
# sizes so no "perfect grid" is optimal; the min-discrepancy set is unknown.
M_LADDER = [5, 7, 9, 11, 14, 17, 20, 24, 28, 32]


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(2)
    t = int(sys.argv[1])
    if t < 1:
        t = 1
    # clamp / wrap into the ladder deterministically
    idx = (t - 1) % len(M_LADDER)
    M = M_LADDER[idx]
    d = 2
    sys.stdout.write("%d %d\n" % (d, M))


if __name__ == "__main__":
    main()
