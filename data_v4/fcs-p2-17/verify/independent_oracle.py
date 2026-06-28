#!/usr/bin/env python3
"""Independent brute oracle for fcs-p2-17.

Counts unordered coin multisets by explicitly choosing how many copies of each
distinct denomination are used. This intentionally differs from sol.cpp's
forward sum-relaxation DP.
"""

from functools import lru_cache
import sys


def solve(inp: str) -> int | None:
    toks = inp.split()
    if not toks:
        return None

    it = iter(toks)
    n = int(next(it))
    target = int(next(it))
    mod = int(next(it))
    coins = [int(next(it)) for _ in range(n)]

    vals = sorted({x for x in coins if x <= target})

    @lru_cache(maxsize=None)
    def count(idx: int, rem: int) -> int:
        if rem == 0:
            return 1
        if idx == len(vals):
            return 0

        coin = vals[idx]
        total = 0
        max_take = rem // coin
        for used in range(max_take + 1):
            total += count(idx + 1, rem - used * coin)
        return total

    return count(0, target) % mod


def main() -> None:
    ans = solve(sys.stdin.read())
    if ans is not None:
        print(ans)


if __name__ == "__main__":
    main()
