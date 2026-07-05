#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE instance (a single integer n = number of rivers).

Difficulty ladder (small = fast reward, large = eval):
  testId 1..10  ->  n = 4,4,5,5,6,6,7,7,8,8
The instance is fully determined by testId (deterministic; no randomness).
"""
import sys

LADDER = [4, 4, 5, 5, 6, 6, 7, 7, 8, 8]


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    idx = (t - 1) % len(LADDER)
    n = LADDER[idx]
    print(n)


if __name__ == "__main__":
    main()
