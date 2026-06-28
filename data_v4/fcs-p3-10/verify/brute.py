#!/usr/bin/env python3
"""Independent brute-force oracle.

Reads the same stdin format as sol.cpp and prints S(N) = sum of the first N
terms of the recurrence f(0)=a, f(1)=b, f(i)=c*f(i-1)+d*f(i-2), all mod p.

This brute simply iterates the recurrence term by term (O(N) per query), so it
is only usable for small N. It shares no code with the matrix solution.
"""
import sys


def solve_query(a, b, c, d, N, p):
    if p == 1:
        return 0
    if N == 0:
        return 0
    a %= p
    b %= p
    c %= p
    d %= p
    if N == 1:
        return a % p
    # iterate
    s = (a + b) % p          # sum of first 2 terms
    fprev2 = a               # f(i-2)
    fprev1 = b               # f(i-1)
    # we have summed f(0), f(1); generate f(2..N-1)
    for _ in range(2, N):
        f = (c * fprev1 + d * fprev2) % p
        s = (s + f) % p
        fprev2 = fprev1
        fprev1 = f
    return s % p


def main():
    data = sys.stdin.read().split()
    idx = 0
    q = int(data[idx]); idx += 1
    out = []
    for _ in range(q):
        a = int(data[idx]); b = int(data[idx + 1]); c = int(data[idx + 2])
        d = int(data[idx + 3]); N = int(data[idx + 4]); p = int(data[idx + 5])
        idx += 6
        out.append(str(solve_query(a, b, c, d, N, p)))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
