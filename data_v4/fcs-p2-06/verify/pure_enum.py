#!/usr/bin/env python3
"""Second, fully independent oracle: pure parenthesization enumeration with
NO memoization at all (exponential, only for tiny n).  Used as an extra
cross-check that the memoized brute and the DP agree with brute-force truth.

Also exposes greedy('cheapest adjacent pair first') so we can confirm it is
genuinely suboptimal on the documented counterexample.
"""
import sys


def enum_cost(p, i, j):
    # Min cost to multiply matrices i..j (1-indexed), no memo.
    if i == j:
        return 0
    best = None
    for k in range(i, j):
        c = enum_cost(p, i, k) + enum_cost(p, k + 1, j) + p[i - 1] * p[k] * p[j]
        if best is None or c < best:
            best = c
    return best


def greedy_cost(p, n):
    # 'multiply the cheapest adjacent pair first' heuristic.
    # dims is the current list of dimension boundaries between live matrices.
    dims = list(p)
    total = 0
    while len(dims) > 2:  # more than one matrix remains
        # adjacent pair m and m+1 share boundary dims[m+1]; cost dims[m]*dims[m+1]*dims[m+2]
        best_m, best_c = None, None
        for m in range(len(dims) - 2):
            c = dims[m] * dims[m + 1] * dims[m + 2]
            if best_c is None or c < best_c:
                best_c, best_m = c, m
        total += best_c
        # merge: remove the shared interior boundary dims[best_m+1]
        del dims[best_m + 1]
    return total


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    p = [int(data[1 + t]) for t in range(n + 1)]
    if n <= 1:
        print(0)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "greedy":
        print(greedy_cost(p, n))
    else:
        print(enum_cost(p, 1, n))


if __name__ == "__main__":
    main()
