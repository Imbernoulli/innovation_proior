#!/usr/bin/env python3
"""gen.py <testId>  -> prints ONE instance (a single integer n) to stdout.

testId 1..10 is a fixed difficulty ladder (small -> large n). No randomness in the
scored artifact; the mapping is deterministic.
"""
import sys

# Fixed ladder of instance sizes. Chosen so the reference (base-5) construction,
# the base-4 construction, and the base-3 construction are all strictly different
# sizes (guarantees tier divergence), and the best simple construction stays below
# saturation.
SIZES = [2000, 3500, 5000, 6500, 8000, 10000, 12000, 18000, 24000, 30000]


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    if t < 1:
        t = 1
    if t > len(SIZES):
        # extend deterministically if the harness asks for more cases
        n = SIZES[-1]
    else:
        n = SIZES[t - 1]
    print(n)


if __name__ == "__main__":
    main()
