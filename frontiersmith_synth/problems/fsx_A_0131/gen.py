#!/usr/bin/env python3
"""Difficulty ladder for the telescope-array cap-set problem.

`python3 gen.py <testId>` prints one instance: the dimension n.
testId 1..6  ->  n = 3,4,5,6,7,8  (small = fast reward, large = eval).
The instance is fully determined by testId (deterministic).
"""
import sys


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tid < 1:
        tid = 1
    n = tid + 2          # tid 1 -> n=3, ... tid 6 -> n=8
    if n > 8:
        n = 8
    print(n)


if __name__ == "__main__":
    main()
