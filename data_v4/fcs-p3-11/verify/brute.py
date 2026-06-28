#!/usr/bin/env python3
"""Independent brute oracle for the Pell-mod-p problem.

Reads the same stdin format as sol.cpp:
  T
  then T lines each: N p
Outputs P(N) mod p for each query, where
  P(0)=0, P(1)=1, P(n)=2 P(n-1)+P(n-2).

Strategy: this oracle is deliberately *different* in mechanism from the
fast-doubling C++. It uses matrix exponentiation of [[2,1],[1,0]] modulo p
done with Python big integers (no overflow concerns, no __int128, no
fast-doubling identities). For small N it can also just iterate; both paths
agree by construction, so we use matrix power as the single source of truth.
"""
import sys


def mat_mult(A, B, p):
    return [
        [(A[0][0] * B[0][0] + A[0][1] * B[1][0]) % p,
         (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % p],
        [(A[1][0] * B[0][0] + A[1][1] * B[1][0]) % p,
         (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % p],
    ]


def mat_pow(M, e, p):
    # identity
    R = [[1 % p, 0 % p], [0 % p, 1 % p]]
    while e > 0:
        if e & 1:
            R = mat_mult(R, M, p)
        M = mat_mult(M, M, p)
        e >>= 1
    return R


def pell_mod(N, p):
    # M^N = [[P(N+1), P(N)], [P(N), P(N-1)]]; we want P(N) = M^N[0][1].
    if N == 0:
        return 0 % p
    M = [[2 % p, 1 % p], [1 % p, 0 % p]]
    R = mat_pow(M, N, p)
    return R[0][1] % p


def main():
    data = sys.stdin.read().split()
    idx = 0
    T = int(data[idx]); idx += 1
    out = []
    for _ in range(T):
        N = int(data[idx]); p = int(data[idx + 1]); idx += 2
        out.append(str(pell_mod(N, p)))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
