#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance "n M seed" to stdout.

testId 1..10 is a difficulty ladder (increasing tide-pool count). The transect length
scales as M = 2n, keeping the shelf moderately dense at every scale so that sum/difference
overlaps are structurally possible. Seeded by testId only (deterministic)."""
import sys

# (n, M) difficulty ladder; M = 2n.
PLAN = [(10, 20), (12, 24), (14, 28), (16, 32), (18, 36),
        (20, 40), (22, 44), (24, 48), (26, 52), (28, 56)]


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t < 1:
        t = 1
    if t > len(PLAN):
        t = len(PLAN)
    n, M = PLAN[t - 1]
    seed = 100003 + t * 7919
    sys.stdout.write("%d %d %d\n" % (n, M, seed))


if __name__ == "__main__":
    main()
