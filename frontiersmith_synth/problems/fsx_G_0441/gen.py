#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance (the number of wires n) to stdout.

The instance of the 'minimum-comparator sorting network' problem is simply the
wire count n. testId 1..8 is a difficulty ladder over n. All n here are >= 13,
where the minimum comparator count is a genuine OPEN problem (proven optimal
only for n <= 12), so there is no reachable known optimum.
"""
import sys

# difficulty ladder: number of wires. Only n where a clean sub-quadratic network
# strictly beats the padded-bitonic construction, so the tiers stay separated.
LADDER = [13, 14, 15, 16, 18, 20, 21, 22]


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    tid = int(sys.argv[1])
    idx = (tid - 1) % len(LADDER)
    n = LADDER[idx]
    print(n)


if __name__ == "__main__":
    main()
