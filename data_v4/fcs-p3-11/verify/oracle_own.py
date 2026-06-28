#!/usr/bin/env python3
"""Independent oracle for Pell modulo p.

This intentionally avoids the fast-doubling formulas used by sol.cpp.
It answers each query by binary exponentiation of the companion matrix
[[2, 1], [1, 0]] using Python integers.
"""

import sys


def mul2(a, b, mod):
    return (
        (
            (a[0][0] * b[0][0] + a[0][1] * b[1][0]) % mod,
            (a[0][0] * b[0][1] + a[0][1] * b[1][1]) % mod,
        ),
        (
            (a[1][0] * b[0][0] + a[1][1] * b[1][0]) % mod,
            (a[1][0] * b[0][1] + a[1][1] * b[1][1]) % mod,
        ),
    )


def pell_mod(n, mod):
    result = ((1 % mod, 0), (0, 1 % mod))
    base = ((2 % mod, 1 % mod), (1 % mod, 0))
    while n:
        if n & 1:
            result = mul2(result, base, mod)
        base = mul2(base, base, mod)
        n >>= 1
    return result[0][1] % mod


def main():
    tokens = sys.stdin.buffer.read().split()
    if not tokens:
        return
    t = int(tokens[0])
    out = []
    at = 1
    for _ in range(t):
        n = int(tokens[at])
        mod = int(tokens[at + 1])
        at += 2
        out.append(str(pell_mod(n, mod)))
    sys.stdout.write("\n".join(out))
    if out:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
