#!/usr/bin/env python3
# Independent brute force for the "lexicographically smallest valid drum pattern" problem.
#
# Pattern: string of length n over {'H','R'} (Hit, Rest).
#   P1: no K consecutive 'R' (every maximal run of 'R' has length <= K-1)
#   P2: exactly h hits ('H')
# Among all valid patterns output the lexicographically smallest under the ordering H < R
# (i.e. plain string comparison since 'H' < 'R' in ASCII). If none exists, print -1.
#
# Method: enumerate every string of length n in lexicographic order (000... = HHH...),
# return the first that satisfies both properties. Obviously correct; exponential, small n only.
import sys
from itertools import product

def solve(n, K, h):
    # iterate in lexicographic order with 'H' as the smaller symbol (bit 0 -> 'H', 1 -> 'R')
    for bits in product("HR", repeat=n):
        s = "".join(bits)
        if s.count('H') != h:
            continue
        # check no run of K consecutive R
        run = 0
        ok = True
        for ch in s:
            if ch == 'R':
                run += 1
                if run >= K:
                    ok = False
                    break
            else:
                run = 0
        if ok:
            return s
    return "-1"

def main():
    data = sys.stdin.read().split()
    n, K, h = int(data[0]), int(data[1]), int(data[2])
    print(solve(n, K, h))

if __name__ == "__main__":
    main()
