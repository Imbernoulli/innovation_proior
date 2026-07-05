#!/usr/bin/env python3
"""Deterministic checker for the wind-tunnel sensor-rail problem.

Usage: python3 verify.py <in> <out> <ans>   (ans ignored)

Reads instance (n M seed), reads the participant station set, strictly validates
feasibility, computes R = |A+A|/|A-A| by exact integer sumsets, and prints

    Ratio: <x in [0,1]>

on the LAST line. Any feasibility violation -> Ratio: 0.0. Exits 0 always."""
import sys

EXP = 3  # scoring exponent on (B / R)


def fail(reason):
    sys.stdout.write("reason: %s\nRatio: 0.0\n" % reason)
    sys.exit(0)


def sumset_ratio(A):
    """R = |A+A| / |A-A| with both cardinalities computed exactly."""
    P = set()
    D = set()
    for a in A:
        for b in A:
            P.add(a + b)
            D.add(a - b)
    return len(P) / len(D), len(P), len(D)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    n, M, seed = int(toks[0]), int(toks[1]), int(toks[2])
    return n, M, seed


def read_stations(path, n, M):
    # bounded read: legitimate outputs are tiny; guard against a flood
    with open(path, "rb") as f:
        raw = f.read(4_000_000)
    text = raw.decode("utf-8", "replace")
    toks = text.split()
    if len(toks) != n:
        fail("expected exactly %d station tokens, got %d" % (n, len(toks)))
    vals = []
    for t in toks:
        # strict base-10 integer: rejects floats, nan, inf, hex, signs+junk
        try:
            v = int(t)
        except ValueError:
            fail("non-integer token %r" % t)
        vals.append(v)
    s = set(vals)
    if len(s) != n:
        fail("stations not pairwise distinct")
    for v in vals:
        if v < 0 or v > M:
            fail("station %d out of range [0,%d]" % (v, M))
    return sorted(s)


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    inf, outf = sys.argv[1], sys.argv[2]
    n, M, seed = read_instance(inf)

    A = read_stations(outf, n, M)

    R, P, Dc = sumset_ratio(A)

    # internal baseline: an evenly spaced arithmetic progression of n stations.
    d = max(1, M // (n - 1))
    A0 = [i * d for i in range(n)]
    B, _, _ = sumset_ratio(A0)   # == 1.0 for an AP

    if R <= 0:
        fail("degenerate ratio")

    sc = 100.0 * (B / R) ** EXP
    if sc > 1000.0:
        sc = 1000.0
    ratio = sc / 1000.0
    sys.stdout.write("R=%.6f |A+A|=%d |A-A|=%d B=%.6f\n" % (R, P, Dc, B))
    sys.stdout.write("Ratio: %.6f\n" % ratio)


if __name__ == "__main__":
    main()
