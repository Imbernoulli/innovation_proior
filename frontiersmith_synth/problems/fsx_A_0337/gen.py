#!/usr/bin/env python3
"""gen.py <testId> -- print ONE instance for the sum-frequency wiring problem.

testId 1..10 is a difficulty ladder: the number of couplers n grows, and the rail
length M = 9*n grows with it so the sum-frequency slots [0, 2M] stay linear in n
(a short rail -> forced collisions -> genuine packing optimization). The instance is
a pure function of testId (fully deterministic; no randomness).
"""
import sys


def instance(test_id: int):
    t = max(1, min(10, int(test_id)))
    n = 10 + 2 * t          # 12, 14, ..., 30
    M = 9 * n               # short rail: 2M+1 sum-slots, linear in n
    return n, M


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n, M = instance(tid)
    sys.stdout.write("%d %d\n" % (n, M))


if __name__ == "__main__":
    main()
