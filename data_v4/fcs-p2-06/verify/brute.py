#!/usr/bin/env python3
"""Independent brute oracle for the matrix-chain multiplication problem.

For small n it enumerates ALL full binary parenthesizations of the chain by
recursion (no DP table reuse beyond Python's own call structure) and takes the
true minimum cost.  This is intentionally a different mechanism from the
solution's interval DP so that a bug in the DP recurrence or its split bounds
would surface as a mismatch.

Input  (stdin): n, then n+1 integers p[0..n].
Output (stdout): minimum number of scalar multiplications.
"""
import sys
from functools import lru_cache


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    p = [int(data[idx + t]) for t in range(n + 1)]
    idx += n + 1

    if n <= 1:
        print(0)
        return

    # best(i, j): min cost to multiply matrices i..j (1-indexed inclusive).
    # The result of multiplying i..j is a p[i-1] x p[j] matrix.
    # We enumerate every place to split the chain and recurse on both halves,
    # i.e. every full parenthesization, then keep the minimum.
    @lru_cache(maxsize=None)
    def best(i, j):
        if i == j:
            return 0
        ans = None
        for k in range(i, j):
            cur = best(i, k) + best(k + 1, j) + p[i - 1] * p[k] * p[j]
            if ans is None or cur < ans:
                ans = cur
        return ans

    print(best(1, n))


if __name__ == "__main__":
    main()
