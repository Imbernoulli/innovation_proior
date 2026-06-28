import sys
from functools import lru_cache

# Independent oracle for the rod-cutting problem.
#
# For small n we ENUMERATE every integer composition of n explicitly (every
# ordered way to write n = c1 + c2 + ... + ck with ci >= 1) and take the max
# total price. This is the literal definition of "max revenue over all ways to
# cut the rod into integer pieces" with no algorithmic cleverness at all.
#
# For larger n (composition count 2^(n-1) explodes) we fall back to a plain
# memoized top-down recursion, structured differently from sol.cpp's bottom-up
# table so a transcription bug in one would not silently match the other.

def enumerate_compositions(n, p):
    if n == 0:
        return 0  # empty rod yields revenue 0

    best = -10**30  # will be overwritten by the first full composition

    def rec(remaining, acc):
        nonlocal best
        if remaining == 0:
            if acc > best:
                best = acc
            return
        for k in range(1, remaining + 1):
            rec(remaining - k, acc + p[k])

    rec(n, 0)
    return best


def memo_recursion(n, p):
    sys.setrecursionlimit(100000)

    @lru_cache(maxsize=None)
    def best(L):
        if L == 0:
            return 0
        return max(p[k] + best(L - k) for k in range(1, L + 1))

    return best(n)


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    p = [0] * (n + 1)
    for k in range(1, n + 1):
        p[k] = int(data[k])

    if n <= 18:
        print(enumerate_compositions(n, p))
    else:
        print(memo_recursion(n, p))


main()
