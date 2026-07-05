#!/usr/bin/env python3
"""gen.py <testId>  ->  prints ONE instance to stdout.

Instance = "n U". testId 1..10 is a difficulty ladder in n (larger n = larger
search space). Fully deterministic (pure function of testId; no randomness).
"""
import sys

U = 1000000
LADDER = [24, 32, 40, 48, 56, 64, 80, 96, 112, 128]


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t < 1:
        t = 1
    if t > len(LADDER):
        # extend gracefully if the harness asks for more cases
        n = LADDER[-1] + 8 * (t - len(LADDER))
    else:
        n = LADDER[t - 1]
    print("%d %d" % (n, U))


if __name__ == "__main__":
    main()
