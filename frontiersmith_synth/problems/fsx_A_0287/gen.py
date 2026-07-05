#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance "n M seed" to stdout.
testId 1..10 is a difficulty ladder (increasing sensor count). Seeded by testId only."""
import sys

# (n, M) difficulty ladder; M = 5n keeps the rail moderately roomy at every scale.
PLAN = [(12, 60), (15, 75), (18, 90), (22, 110), (26, 130),
        (31, 155), (36, 180), (41, 205), (46, 230), (50, 250)]


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
