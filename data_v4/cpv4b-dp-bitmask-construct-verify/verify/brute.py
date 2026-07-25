#!/usr/bin/env python3
# INDEPENDENT brute force for the "alarm tape" problem.
#
# Problem (restated):
#   We must emit a binary tape t[0..n-1]. A blacklist gives m forbidden patterns,
#   each a binary string of EXACTLY length L. The tape is SAFE iff no contiguous
#   window of length L equals any blacklisted pattern. (If n < L there are zero
#   windows of length L, so every tape of length n is safe.)
#   Output the lexicographically smallest SAFE tape with '0' < '1', or "-1".
#
# Input format (stdin):
#   first line: n L m
#   next m lines: each a length-L binary string (the blacklist; may be empty if m=0)
#
# Brute force: enumerate all 2^n tapes in lexicographic order, return first safe.

import sys


def safe(t, n, L, forb):
    if n < L:
        return True
    for i in range(0, n - L + 1):
        if t[i:i + L] in forb:
            return False
    return True


def solve(n, L, forb):
    for mask in range(1 << n):
        # index 0 = MSB so that mask=0 (all '0') is lexicographically smallest
        t = ''.join('1' if (mask >> (n - 1 - i)) & 1 else '0' for i in range(n))
        if safe(t, n, L, forb):
            return t
    return "-1"


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    L = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    forb = set()
    for _ in range(m):
        forb.add(data[idx]); idx += 1
    print(solve(n, L, forb))


if __name__ == "__main__":
    main()
