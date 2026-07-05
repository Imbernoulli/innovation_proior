#!/usr/bin/env python3
"""
Deterministic checker for fsx_A_0125 -- "Wind-Tunnel Sensor Leakage Energy".

Reads:
  <in>  : first token n = number of sensors.
  <out> : n non-negative real gain values f_0..f_{n-1}.

Objective (MINIMIZE) -- normalized leakage-energy index (squared L2 norm of the
autoconvolution / additive-energy functional):
    E(f) = n * ( sum_k c_k^2 )  /  ( sum_i f_i )^4
where  c_k = sum_{i+j=k} f_i f_j   (autoconvolution), k = 0 .. 2n-2.

Internal baseline B = E of the "half-block" construction the checker builds
itself: unit gain on the first floor(n/2) sensors, zero elsewhere (all gain
crammed into one contiguous half of the array). A trivially feasible but
deliberately concentrated allocation.

Score (minimization):
    sc  = min(1000.0, 100.0 * B / F)
    Ratio = sc / 1000.0
so reproducing the baseline -> 0.1 ; a 10x lower index caps at 1.0.
"""
import sys


def objective(f, n):
    s = sum(f)
    m = len(f)
    # autoconvolution c_k = sum_{i+j=k} f_i f_j , k = 0 .. 2m-2
    energy = 0.0
    for k in range(2 * m - 1):
        lo = max(0, k - (m - 1))
        hi = min(k, m - 1)
        acc = 0.0
        for i in range(lo, hi + 1):
            acc += f[i] * f[k - i]
        energy += acc * acc
    return n * energy / (s ** 4)


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as fh:
        n = int(fh.read().split()[0])

    with open(out_path) as fh:
        toks = fh.read().split()

    if len(toks) != n:
        fail("expected %d values, got %d" % (n, len(toks)))

    f = []
    for t in toks:
        try:
            v = float(t)
        except ValueError:
            fail("non-numeric token %r" % t)
        if v != v or v in (float("inf"), float("-inf")):
            fail("non-finite value")
        if v < 0.0:
            fail("negative gain")
        if v > 1e6:
            fail("gain exceeds 1e6")
        f.append(v)

    s = sum(f)
    if s <= 1e-9:
        fail("total gain must be positive")

    F = objective(f, n)

    # internal baseline: half-block
    base = [1.0] * (n // 2) + [0.0] * (n - n // 2)
    B = objective(base, n)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f  Ratio: %.6f" % (F, B, ratio))


if __name__ == "__main__":
    main()
