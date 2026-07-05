#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  (ans ignored)

Deterministic scorer for the Ridge-Line Watchtower Codes problem.

Reads the instance (n, M) and the participant placement A. Validates strictly:
exactly n distinct integers, each in [0, M]. Computes the exact code-efficiency
ratio F = |A+A| / |A-A|, builds its own Sidon reference placement of n towers to
get a baseline ratio B, and prints

    Ratio: <sc/1000>      with  sc = min(1000, 100 * F / B)

Any feasibility violation prints "Ratio: 0.0". O(n^2) integer arithmetic only.
"""
import sys


def sumset_size(A):
    return len({a + b for a in A for b in A})


def diffset_size(A):
    return len({a - b for a in A for b in A})


def _is_prime(m):
    if m < 2:
        return False
    i = 2
    while i * i <= m:
        if m % i == 0:
            return False
        i += 1
    return True


def _next_prime(x):
    while not _is_prime(x):
        x += 1
    return x


def sidon_reference(n):
    """Erdos-Turan Sidon set of size n: elements 2*p*k + (k*k mod p), p >= n prime.
    Truncating a Sidon set keeps it Sidon. Its ratio is well below 1."""
    p = _next_prime(n)
    S = sorted({2 * p * k + (k * k) % p for k in range(p)})
    return S[:n]


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    with open(inf) as f:
        toks = f.read().split()
    n = int(toks[0])
    M = int(toks[1])

    # ---- parse participant placement ----
    try:
        with open(outf) as f:
            vals = [int(t) for t in f.read().split()]
    except Exception as e:
        print("Ratio: 0.0  (unparseable output: %s)" % e)
        return

    # ---- strict feasibility ----
    if len(vals) != n:
        print("Ratio: 0.0  (expected exactly %d integers, got %d)" % (n, len(vals)))
        return
    for v in vals:
        if v < 0 or v > M:
            print("Ratio: 0.0  (mile-post %d outside [0,%d])" % (v, M))
            return
    if len(set(vals)) != n:
        print("Ratio: 0.0  (mile-posts must be distinct)")
        return

    A = vals
    F = sumset_size(A) / diffset_size(A)

    # ---- internal baseline: Sidon reference of the same size ----
    ref = sidon_reference(n)
    B = sumset_size(ref) / diffset_size(ref)
    if B <= 0:
        B = 1e-9

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
