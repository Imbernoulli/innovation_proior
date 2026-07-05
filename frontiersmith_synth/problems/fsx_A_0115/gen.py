#!/usr/bin/env python3
# gen.py <testId>  -- prints ONE instance "n U" to stdout.
# testId 1..10 = difficulty ladder (small -> larger ridge).  Deterministic: depends only on testId.
import sys

NS = [8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
U = 1000

def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t < 1:
        t = 1
    if t > len(NS):
        # extrapolate deterministically beyond the ladder if asked
        n = NS[-1] + 4 * (t - len(NS))
    else:
        n = NS[t - 1]
    print("%d %d" % (n, U))

if __name__ == "__main__":
    main()
