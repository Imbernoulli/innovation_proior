#!/usr/bin/env python3
"""Independent brute-force oracle for the stair-climbing composition counter.

Reads the same stdin format as sol.cpp:
    N k p
    s_1 s_2 ... s_k

Counts ordered compositions of N using parts from the set S = {s_i}, modulo p,
via a straightforward O(N * |S|) dynamic program (no matrix tricks).
f(0) = 1; f(n) = sum_{s in S, s <= n} f(n - s).

This is intentionally the slow, obviously-correct method; it is only viable for
small N, which is exactly what we feed it in differential testing.
"""
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    N = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    p = int(data[idx]); idx += 1
    S = []
    for _ in range(k):
        S.append(int(data[idx])); idx += 1
    # deduplicate set of distinct step sizes
    Sset = sorted(set(S))

    # f[n] mod p via the recurrence
    f = [0] * (N + 1)
    f[0] = 1 % p
    for n in range(1, N + 1):
        acc = 0
        for s in Sset:
            if s <= n:
                acc += f[n - s]
        f[n] = acc % p
    print(f[N] % p)


if __name__ == "__main__":
    main()
