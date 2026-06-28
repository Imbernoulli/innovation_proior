#!/usr/bin/env python3
# Independent fast checker using a DIFFERENT method than sol.cpp:
# For each length L (1..19), count L-digit numbers <= bound with digit sum % L == 0
# via memoized recursion over (pos, residue, tight, leading). This is an
# independent digit-DP implementation (recursive, per-length) for cross-checking
# large inputs that the range-scan brute cannot reach.
import sys
from functools import lru_cache

def count_len_le(length, bound_digits):
    # bound_digits: list of digits (MSB first), length == length.
    mod = length
    n = length
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def rec(pos, res, tight):
        if pos == n:
            return 1 if res % mod == 0 else 0
        total = 0
        lo = 1 if pos == 0 else 0
        hi = bound_digits[pos] if tight else 9
        for d in range(lo, hi + 1):
            total += rec(pos + 1, (res + d) % mod, tight and (d == hi))
        return total
    r = rec(0, 0, True)
    rec.cache_clear()
    return r

def count_up_to(N):
    if N <= 0:
        return 0
    s = str(N)
    D = len(s)
    ans = 0
    for length in range(1, D + 1):
        if length < D:
            bd = [9] * length
        else:
            bd = [int(c) for c in s]
        ans += count_len_le(length, bd)
    return ans

def main():
    data = sys.stdin.read().split()
    L = int(data[0]); R = int(data[1])
    print(count_up_to(R) - count_up_to(L - 1))

if __name__ == "__main__":
    main()
