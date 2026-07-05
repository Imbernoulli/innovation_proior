#!/usr/bin/env python3
"""Deterministic checker for the inverter-resonance autoconvolution problem.

CLI:  python3 verify.py <in> <out> <ans>   (ans ignored)

Reads:
  <in>  : "n V"
  <out> : n non-negative integers f_0..f_{n-1}  (the emission step vector)

Feasibility (any violation -> Ratio: 0.0):
  * exactly n integer tokens
  * 0 <= f_i <= V for every i
  * sum f_i > 0  (a non-zero profile)

Objective (MINIMIZE):
      c1(f) = 2 * n * max_k (f * f)[k]  /  (sum f)^2
  where (f * f)[k] = sum_{i+j=k} f_i f_j is the self-convolution.
  This is the discrete first-autocorrelation constant: the peak bus
  resonance normalised by total delivered power.

Scoring (exact rational objective, minimisation):
  Internal baseline B = c1(triangle profile) that the checker builds itself.
  sc = min(1000, 100 * B / F);  Ratio = sc / 1000.
  A construction matching the triangle baseline scores ~0.1; a 10x
  improvement caps at 1.0.
"""
import sys
from fractions import Fraction


def read_ints(path):
    with open(path) as f:
        return f.read().split()


def c1_fraction(f):
    """Exact c1 = 2 n * peak / S^2 as a Fraction."""
    n = len(f)
    S = sum(f)
    m = 2 * n - 1
    conv = [0] * m
    for i, fi in enumerate(f):
        if fi == 0:
            continue
        for j, fj in enumerate(f):
            if fj:
                conv[i + j] += fi * fj
    peak = max(conv)
    return Fraction(2 * n * peak, S * S)


def triangle(n):
    return [min(i + 1, n - i) for i in range(n)]


def fail(msg):
    print("infeasible: %s Ratio: 0.0" % msg)
    sys.exit(0)


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    tok = read_ints(inp)
    n = int(tok[0]); V = int(tok[1])

    toks = read_ints(outp)
    if len(toks) != n:
        fail("expected %d values, got %d" % (n, len(toks)))
    f = []
    for s in toks:
        try:
            x = int(s)
        except ValueError:
            fail("non-integer token '%s'" % s)
        if x < 0 or x > V:
            fail("value %d out of range [0,%d]" % (x, V))
        f.append(x)
    if sum(f) <= 0:
        fail("zero profile")

    F = c1_fraction(f)
    B = c1_fraction(triangle(n))

    sc = min(1000.0, 100.0 * float(B) / max(1e-9, float(F)))
    print("n=%d c1=%.6f baseline=%.6f Ratio: %.6f"
          % (n, float(F), float(B), sc / 1000.0))


if __name__ == "__main__":
    main()
