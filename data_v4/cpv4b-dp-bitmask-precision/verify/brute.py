#!/usr/bin/env python3
# Independent brute force: enumerate every permutation (Hamiltonian path order)
# and compute the product exactly with Python big integers. No DP, no overflow,
# no floating point -- this is the trusted oracle.
import sys
from itertools import permutations

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    it = iter(data)
    n = int(next(it))
    b = [int(next(it)) for _ in range(n)]
    m = [[int(next(it)) for _ in range(n)] for _ in range(n)]

    if n == 0:
        print(0)
        return
    if n == 1:
        print(b[0])
        return

    best = None
    for perm in permutations(range(n)):
        prod = b[perm[0]]
        for k in range(1, n):
            prod *= m[perm[k - 1]][perm[k]]
        if best is None or prod > best:
            best = prod
    print(best)

if __name__ == "__main__":
    main()
