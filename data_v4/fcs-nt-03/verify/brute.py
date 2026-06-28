#!/usr/bin/env python3
# Independent brute force for D(n) = sum_{i=1..n} d(i), where d(i) = number of
# divisors of i. We compute it the obvious slow way: a sieve of divisor counts,
# i.e. for every j, add 1 to every multiple of j. This is O(n log n) and is an
# entirely different code path from the hyperbola formula (no floor(n/i) at all).
import sys

def solve(n):
    if n <= 0:
        return 0
    d = [0] * (n + 1)
    for j in range(1, n + 1):
        for m in range(j, n + 1, j):
            d[m] += 1
    return sum(d[1:n + 1])

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    print(solve(n))

if __name__ == "__main__":
    main()
