#!/usr/bin/env python3
import sys

MOD = 998244353

def main():
    data = sys.stdin.read().split()
    idx = 0
    k = int(data[idx]); idx += 1
    c = [int(data[idx + i]) % MOD for i in range(k)]; idx += k
    a = [int(data[idx + i]) % MOD for i in range(k)]; idx += k
    N = int(data[idx]); idx += 1

    # Direct iteration of the recurrence:
    #   a[n] = c[0]*a[n-1] + c[1]*a[n-2] + ... + c[k-1]*a[n-k]  (mod MOD)
    # Obviously correct but O(N*k); only used for small N in differential testing.
    if N < k:
        print(a[N])
        return
    seq = list(a)  # seq[i] = a[i]
    for n in range(k, N + 1):
        v = 0
        for j in range(k):
            v += c[j] * seq[n - 1 - j]
        seq.append(v % MOD)
    print(seq[N] % MOD)

if __name__ == "__main__":
    main()
