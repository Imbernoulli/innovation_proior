#!/usr/bin/env python3
"""
Independent brute-force oracle for: number of binary strings of length N with no
run of k or more consecutive ones, modulo p.

Reads the SAME stdin format as sol.cpp:
    T
    N k p     (T lines)
and prints T answers, one per line.

This oracle uses a completely independent method from the shipped solution:
a direct O(N*k) dynamic programming over (position, length of trailing run of
ones), which makes no use of the order-k linear recurrence, matrix exponentiation
or Kitamasa. It is therefore a genuine cross-check (valid only for modest N).

dp[j] = number of valid strings of the current length whose trailing run of ones
        has length exactly j  (0 <= j <= k-1).
Initial (length 0): the empty string has trailing-run length 0 -> dp[0] = 1.
Transition appending one more character:
    - append a 0: goes to state 0 from any state.
    - append a 1: state j -> j+1, allowed only if j+1 <= k-1.
Answer for length N is sum(dp).
"""
import sys


def count(N, k, p):
    if k <= 0:
        return 0
    # states 0..k-1 (trailing run of ones), but if k==1 only state 0 exists
    dp = [0] * k
    dp[0] = 1 % p
    for _ in range(N):
        ndp = [0] * k
        # appending '0' collapses everything into state 0
        total = sum(dp) % p
        ndp[0] = total
        # appending '1' shifts state j -> j+1 if j+1 <= k-1
        for j in range(k - 1):
            ndp[j + 1] = (ndp[j + 1] + dp[j]) % p
        dp = ndp
    return sum(dp) % p


def main():
    data = sys.stdin.read().split()
    idx = 0
    T = int(data[idx]); idx += 1
    out = []
    for _ in range(T):
        N = int(data[idx]); k = int(data[idx + 1]); p = int(data[idx + 2])
        idx += 3
        out.append(str(count(N, k, p)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
