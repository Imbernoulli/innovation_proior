#!/usr/bin/env python3
# Independent brute-force ORACLE + CHECKER for the min-count subset-sum construction.
#
# Because several minimum-size subsets may exist and the problem accepts ANY one of
# them, this script cannot just diff strings. Instead it:
#   (1) computes the TRUE minimum item count to reach S by exhaustive subset search,
#   (2) reads the candidate solver's output and validates the claimed construction:
#         - if true answer is "impossible", the solver must print -1, and vice versa;
#         - otherwise the printed indices must be distinct, in range, sum to exactly S,
#           and number exactly the true minimum count.
# It prints "OK" on success and "MISMATCH: <reason>" on failure (exit code signals it).
#
# Usage: python3 brute.py <input_file> <solver_output_file>
import sys
from itertools import combinations


def true_min_count(n, S, w):
    # Returns the minimum number of items summing to exactly S, or None if impossible.
    # Exhaustive over subset SIZES from small to large so the first hit is the minimum.
    if S == 0:
        return 0
    items = [(i, w[i]) for i in range(n) if 0 < w[i] <= S]
    # try increasing subset size
    for k in range(1, len(items) + 1):
        for comb in combinations(items, k):
            if sum(v for _, v in comb) == S:
                return k
    return None


def main():
    inp = open(sys.argv[1]).read().split()
    idx = 0
    n = int(inp[idx]); idx += 1
    S = int(inp[idx]); idx += 1
    w = [int(inp[idx + i]) for i in range(n)]

    truth = true_min_count(n, S, w)

    out = open(sys.argv[2]).read().split()
    if len(out) == 0:
        print("MISMATCH: empty output")
        sys.exit(1)

    # solver prints -1 for impossible
    if out[0] == "-1":
        if truth is None:
            print("OK")
            sys.exit(0)
        print(f"MISMATCH: solver says impossible but min count is {truth}")
        sys.exit(1)

    if truth is None:
        print("MISMATCH: solver printed a subset but no subset sums to S")
        sys.exit(1)

    k = int(out[0])
    indices = [int(x) for x in out[1:1 + k]]
    if len(indices) != k:
        print(f"MISMATCH: header says {k} items but {len(indices)} indices given")
        sys.exit(1)

    if len(set(indices)) != k:
        print("MISMATCH: repeated index")
        sys.exit(1)

    for ix in indices:
        if ix < 1 or ix > n:
            print(f"MISMATCH: index {ix} out of range [1,{n}]")
            sys.exit(1)

    tot = sum(w[ix - 1] for ix in indices)
    if tot != S:
        print(f"MISMATCH: subset sums to {tot}, not S={S}")
        sys.exit(1)

    if k != truth:
        print(f"MISMATCH: used {k} items but minimum is {truth}")
        sys.exit(1)

    print("OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
