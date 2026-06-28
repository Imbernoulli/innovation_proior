#!/usr/bin/env python3
"""Independent oracle for fcs-p3-04.

The checked program uses 2x2 matrix exponentiation.  This oracle deliberately
uses different formulations:
  * for small N, recursively enumerate every binary string and count only
    strings that never place adjacent ones;
  * for medium N, run an exact arbitrary-precision DP over strings ending in
    zero/one and reduce only at the end;
  * for very large N, use a separate fast-doubling Fibonacci identity so edge
    cases near 1e18 can still be checked.
"""

import sys


ENUM_LIMIT = 24
DP_LIMIT = 10000


def enumerate_count(n):
    total = 0

    def visit(pos, prev_one):
        nonlocal total
        if pos == n:
            total += 1
            return
        visit(pos + 1, False)
        if not prev_one:
            visit(pos + 1, True)

    visit(0, False)
    return total


def exact_dp_count(n):
    # zero/one are counts of valid strings of the current length ending in 0/1.
    if n == 0:
        return 1
    zero, one = 1, 1
    for _ in range(2, n + 1):
        zero, one = zero + one, zero
    return zero + one


def fib_mod(k, mod):
    if mod == 1:
        return 0

    def rec(x):
        if x == 0:
            return 0, 1
        a, b = rec(x >> 1)
        c = (a * ((2 * b - a) % mod)) % mod
        d = (a * a + b * b) % mod
        if x & 1:
            return d, (c + d) % mod
        return c, d

    return rec(k)[0]


def oracle(n, p):
    if n <= ENUM_LIMIT:
        direct = enumerate_count(n)
        dp = exact_dp_count(n)
        assert direct == dp, (n, direct, dp)
        return direct % p
    if n <= DP_LIMIT:
        return exact_dp_count(n) % p
    # f(n) = Fib(n + 2), with Fib(0)=0, Fib(1)=1.
    return fib_mod(n + 2, p)


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    t = int(data[0])
    out = []
    at = 1
    for _ in range(t):
        n = int(data[at])
        p = int(data[at + 1])
        at += 2
        out.append(str(oracle(n, p)))
    sys.stdout.write("\n".join(out))
    if out:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
