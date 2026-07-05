#!/usr/bin/env python3
"""gen.py <testId>  -- print ONE instance to stdout.

Instance = "n M": a budget n on the number of sensor time-offsets we may deploy,
and the maximum offset value M (offsets live in [0, M]).  testId 1..10 is a
difficulty ladder: n grows and the range multiplier M/n cycles through several
regimes (tight ranges force sum collisions; looser ranges reward spread-out
Sidon-like placements).  Fully deterministic in testId.
"""
import sys

def instance(t):
    # size budget grows with difficulty; stays "small" (<= 200)
    n = 12 + 3 * t                      # t=1..10 -> 15..42
    mults = [4, 5, 6, 7, 8, 5, 6, 7, 8, 9]
    a = mults[(t - 1) % len(mults)]
    M = a * n
    return n, M

def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t < 1:
        t = 1
    n, M = instance(t)
    sys.stdout.write("%d %d\n" % (n, M))

if __name__ == "__main__":
    main()
