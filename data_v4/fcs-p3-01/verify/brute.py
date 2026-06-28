#!/usr/bin/env python3
"""Independent brute oracle for the tribonacci-mod problem.

Input format (stdin):
  T
  then T lines each: n p f0 f1 f2

Definition:
  f(0)=f0, f(1)=f1, f(2)=f2
  f(k) = f(k-1) + f(k-2) + f(k-3)  for k >= 3
  output f(n) mod p

This brute iterates the recurrence directly (O(n) per query), keeping
everything reduced mod p. Only usable for small n; used for differential
testing against the matrix-exponentiation solution.
"""
import sys


def solve_one(n, p, f0, f1, f2):
    a0 = f0 % p
    a1 = f1 % p
    a2 = f2 % p
    if n == 0:
        return a0
    if n == 1:
        return a1
    if n == 2:
        return a2
    # iterate; window holds (f(k-3), f(k-2), f(k-1))
    prev3, prev2, prev1 = a0, a1, a2
    cur = None
    for _ in range(3, n + 1):
        cur = (prev1 + prev2 + prev3) % p
        prev3, prev2, prev1 = prev2, prev1, cur
    return cur


def main():
    data = sys.stdin.read().split()
    idx = 0
    T = int(data[idx]); idx += 1
    out = []
    for _ in range(T):
        n = int(data[idx]); idx += 1
        p = int(data[idx]); idx += 1
        f0 = int(data[idx]); idx += 1
        f1 = int(data[idx]); idx += 1
        f2 = int(data[idx]); idx += 1
        out.append(str(solve_one(n, p, f0, f1, f2)))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
