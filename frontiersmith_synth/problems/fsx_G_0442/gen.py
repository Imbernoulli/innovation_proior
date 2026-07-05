#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE symmetric-boolean-function instance to stdout.

Instance format (stdin for the solver):
    line 1: n                          (number of input bits x0..x_{n-1})
    line 2: spectrum, a string of n+1  characters over {0,1};
            character k (0-indexed) is '1' iff the function ACCEPTS inputs
            whose popcount (number of set bits) equals k.

The target is the symmetric function  f(x) = spectrum[popcount(x)].
Difficulty ladder: testId 1..10 -> increasing n. Everything seeded by testId only.
"""
import sys, random

NS = {1: 8, 2: 9, 3: 10, 4: 11, 5: 12, 6: 13, 7: 14, 8: 15, 9: 16, 10: 18}


def main():
    tid = int(sys.argv[1])
    n = NS.get(tid, 8 + tid)
    rng = random.Random(1000 + tid * 7919)
    p = 0.42 + 0.06 * ((tid * 37) % 5) / 4.0  # 0.42..0.48, deterministic
    while True:
        bits = [1 if rng.random() < p else 0 for _ in range(n + 1)]
        s = sum(bits)
        has_run = any(bits[i] == 1 and bits[i + 1] == 1 for i in range(n))
        # need: not-all, not-none, some rejects, and at least one length>=2 run
        if 2 <= s <= n - 1 and has_run:
            break
    sys.stdout.write("%d\n%s\n" % (n, "".join(map(str, bits))))


if __name__ == "__main__":
    main()
