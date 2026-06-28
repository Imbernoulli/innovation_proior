#!/usr/bin/env python3
"""
Independent brute oracle for fcs-p3-05.

This intentionally avoids the closed-form Catalan formula, factorials, modular
inverses, and the existing convolution brute.  For each distinct n it counts
balanced bracket strings directly from the prefix-balance definition using exact
integer DP, then reduces the exact count modulo the query prime.
"""
from functools import lru_cache
import sys


@lru_cache(maxsize=None)
def count_balanced(n: int) -> int:
    # State is (opens_used, current_balance).  Closing is allowed only when the
    # current balance is positive, so every generated prefix is valid.
    states = {(0, 0): 1}
    for _ in range(2 * n):
        nxt = {}
        for (opens, balance), ways in states.items():
            if opens < n:
                key = (opens + 1, balance + 1)
                nxt[key] = nxt.get(key, 0) + ways
            if balance > 0:
                key = (opens, balance - 1)
                nxt[key] = nxt.get(key, 0) + ways
        states = nxt
    return states.get((n, 0), 0)


def main() -> None:
    data = sys.stdin.read().split()
    if not data:
        return
    q = int(data[0])
    pos = 1
    out = []
    for _ in range(q):
        n = int(data[pos])
        p = int(data[pos + 1])
        pos += 2
        out.append(str(count_balanced(n) % p))
    sys.stdout.write("\n".join(out))
    if out:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
