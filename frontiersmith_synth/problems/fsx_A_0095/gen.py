#!/usr/bin/env python3
"""gen.py <testId>  -> prints ONE instance to stdout.

Instance line: "n V"
  n = number of consecutive baking shifts (schedule length)
  V = maximum number of loaves that may be baked in a single shift

Difficulty ladder: testId 1..10 -> increasing n (longer schedule = larger
combinatorial search space). Fully deterministic in testId (no RNG needed).
"""
import sys

def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t < 1:
        t = 1
    if t > 10:
        t = 10
    n = 12 + 4 * t          # 16, 20, 24, ..., 52
    V = 1000
    sys.stdout.write("%d %d\n" % (n, V))

if __name__ == "__main__":
    main()
