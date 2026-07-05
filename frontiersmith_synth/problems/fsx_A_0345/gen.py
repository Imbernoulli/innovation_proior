#!/usr/bin/env python3
"""gen.py <testId>  ->  prints ONE instance to stdout.

Difficulty ladder (testId 1..10): n = 30 * testId  (30 .. 300).
Larger n = higher-dimensional profile = harder to drive c1 below the flat value.
The instance is fully determined by testId (deterministic).
"""
import sys


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tid < 1:
        tid = 1
    n = 30 * tid
    M = 1_000_000
    sys.stdout.write("%d %d\n" % (n, M))


if __name__ == "__main__":
    main()
