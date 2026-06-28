#!/usr/bin/env python3
"""Independent brute oracle for the lattice-path counting problem.

Number of monotone lattice paths from (0,0) to (a,b) (unit steps +x or +y)
equals C(a+b, a). We compute it a completely different way from sol.cpp:
build Pascal's triangle row-by-row using additive recurrence
  C(n,k) = C(n-1,k-1) + C(n-1,k)  (mod p),
which involves NO factorials and NO modular inverse. This is an independent
algorithm (additive DP table) so it cross-checks the multiplicative approach.

We answer each query by reading off C(a+b, a) from the table, building the
table up to the max a+b across queries. To keep memory/time sane for the
brute (it is O(N^2)), the generator only emits small a+b for differential
testing; this oracle is intentionally limited to small n.
"""
import sys

MOD = 1000000007


def solve(data):
    it = iter(data)
    q = int(next(it))
    aa = []
    bb = []
    maxn = 0
    for _ in range(q):
        a = int(next(it))
        b = int(next(it))
        aa.append(a)
        bb.append(b)
        maxn = max(maxn, a + b)

    # Pascal triangle up to row maxn, additive recurrence, mod p.
    # C[n][k] for 0<=k<=n.
    prev = [1]  # row 0
    rows = [prev]
    for n in range(1, maxn + 1):
        cur = [0] * (n + 1)
        cur[0] = 1
        cur[n] = 1
        for k in range(1, n):
            cur[k] = (prev[k - 1] + prev[k]) % MOD
        rows.append(cur)
        prev = cur

    out = []
    for a, b in zip(aa, bb):
        n = a + b
        out.append(str(rows[n][a] % MOD))
    return "\n".join(out) + ("\n" if out else "")


def main():
    data = sys.stdin.read().split()
    sys.stdout.write(solve(data))


if __name__ == "__main__":
    main()
