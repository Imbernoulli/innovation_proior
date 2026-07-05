#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic scorer for the orbital-debris
self-overlap (first autocorrelation inequality) problem.

Reads n from <in>. Reads a non-negative sweep-density profile f (n reals) from
<out>. Objective to MINIMISE:

    c1(f) = 2n * max_k conv(f,f)_k / (sum_i f_i)^2

where conv(f,f)_k = sum_i f_i f_{k-i} is the discrete self-convolution (the
relative-phase self-overlap distribution). Lower c1 = the fleet's own passes
pile up less at any single relative phase.

Feasibility: exactly n tokens, each finite and >= 0, not all zero, none absurdly
large. Any violation -> Ratio: 0.0.

Baseline B: c1 of a smooth centred Gaussian bump the checker builds itself
(a plausible naive "spread the mass smoothly" profile). Minimisation scoring:
    sc = min(1000, 100 * B / F);  print Ratio: sc/1000
so reproducing the baseline -> ~0.1 and a 10x-better profile caps at 1.0. The
exact optimum of this inequality is an open research constant (~1.5), so the
score never saturates.
"""
import sys
import math

MAXVAL = 1e9


def read_ratio_zero(reason):
    print("infeasible: " + reason)
    print("Ratio: 0.0")
    sys.exit(0)


def conv_max(f):
    n = len(f)
    out = [0.0] * (2 * n - 1)
    for i in range(n):
        fi = f[i]
        if fi == 0.0:
            continue
        base = i
        for j in range(n):
            out[base + j] += fi * f[j]
    return max(out)


def c1(f):
    s = sum(f)
    n = len(f)
    return 2.0 * n * conv_max(f) / (s * s)


def gauss_baseline(n):
    # centred bump on the phase axis, normalised x in [-1, 1]
    if n == 1:
        return [1.0]
    return [math.exp(-4.0 * ((2.0 * i / (n - 1) - 1.0) ** 2)) for i in range(n)]


def main():
    if len(sys.argv) < 3:
        read_ratio_zero("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    with open(inf) as fh:
        n = int(fh.read().split()[0])

    # bounded read of participant output
    try:
        with open(outf) as fh:
            raw = fh.read(1 << 20)  # 1 MB cap
    except Exception:
        read_ratio_zero("cannot read output")

    toks = raw.split()
    if len(toks) != n:
        read_ratio_zero("expected %d values, got %d" % (n, len(toks)))

    f = []
    for t in toks:
        try:
            v = float(t)
        except Exception:
            read_ratio_zero("non-numeric token")
        if not math.isfinite(v):
            read_ratio_zero("non-finite value")
        if v < 0.0:
            read_ratio_zero("negative density")
        if v > MAXVAL:
            read_ratio_zero("value exceeds cap")
        f.append(v)

    s = sum(f)
    if s <= 0.0:
        read_ratio_zero("total sweep mass is zero")

    F = c1(f)
    if not math.isfinite(F) or F <= 0.0:
        read_ratio_zero("objective not finite/positive")

    B = c1(gauss_baseline(n))

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("n=%d  F(c1)=%.6f  B(c1)=%.6f" % (n, F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
