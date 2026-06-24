#!/usr/bin/env python3
import sys

MOD = 1_000_000_007

def solve(data):
    it = iter(data)
    n = int(next(it))
    a = [int(next(it)) for _ in range(n)]

    # For each subarray [l, r], the "owner" is the index of the minimum element,
    # breaking ties by choosing the LEFTMOST minimum index in the subarray.
    # c[i] = number of subarrays owned by index i.
    # Answer = sum_i (i * c[i]) mod 1e9+7, where i is the 0-based index.
    c = [0] * n
    for l in range(n):
        m = a[l]
        owner = l
        for r in range(l, n):
            if a[r] < m:        # STRICT: leftmost-min tie-break => only update on strictly smaller
                m = a[r]
                owner = r
            c[owner] += 1

    ans = 0
    for i in range(n):
        ans = (ans + (i % MOD) * (c[i] % MOD)) % MOD
    return str(ans)

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    sys.stdout.write(solve(data) + "\n")

if __name__ == "__main__":
    main()
