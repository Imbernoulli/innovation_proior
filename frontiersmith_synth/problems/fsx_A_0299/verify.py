#!/usr/bin/env python3
"""Deterministic checker for the sentinel-grid low-discrepancy pointset problem.

Usage: python3 verify.py <in> <out> <ans>   (ans is an ignored placeholder)

Reads the instance (d M) from <in> and the participant's M*d coordinates from
<out>. Strictly validates feasibility; on ANY violation prints `Ratio: 0.0`.
Otherwise computes the exact L2 star discrepancy via Warnock's identity,
compares against an internally-built pseudo-random baseline, and prints
`Ratio: <r>`.  O(M^2 * d), bit-for-bit deterministic on reruns.
"""
import sys
import math


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    d = int(toks[0])
    M = int(toks[1])
    return d, M


def l2star_sq(pts, d, M):
    """Exact squared L2 star discrepancy (Warnock's identity)."""
    t1 = 3.0 ** (-d)
    s2 = 0.0
    for x in pts:
        p = 1.0
        for k in range(d):
            p *= (1.0 - x[k] * x[k])
        s2 += p
    t2 = (2.0 ** (1 - d) / M) * s2
    s3 = 0.0
    for i in range(M):
        xi = pts[i]
        for j in range(M):
            xj = pts[j]
            p = 1.0
            for k in range(d):
                a = xi[k]
                b = xj[k]
                p *= (1.0 - (a if a > b else b))
            s3 += p
    t3 = s3 / (M * M)
    return t1 - t2 + t3


def lcg_baseline(d, M):
    """Deterministic pseudo-random scatter, seeded only by (d, M)."""
    seed = (d * 1000003 + M * 97 + 12345) & 0x7FFFFFFF
    s = seed
    pts = []
    for _ in range(M):
        c = []
        for _k in range(d):
            s = (1103515245 * s + 12345) & 0x7FFFFFFF
            c.append(s / 0x7FFFFFFF)
        pts.append(tuple(c))
    return pts


def fail(reason):
    print("Infeasible (%s). Ratio: 0.0" % reason)
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    d, M = read_instance(in_path)

    # ---- read participant output ----
    try:
        with open(out_path) as f:
            raw = f.read().split()
    except Exception:
        fail("unreadable output")

    need = d * M
    if len(raw) != need:
        fail("expected %d numbers, got %d" % (need, len(raw)))

    vals = []
    for tok in raw:
        try:
            v = float(tok)
        except Exception:
            fail("non-numeric token")
        if not math.isfinite(v):
            fail("non-finite value")
        if v < 0.0 or v > 1.0:
            fail("coordinate outside [0,1]")
        vals.append(v)

    pts = [tuple(vals[i * d:(i + 1) * d]) for i in range(M)]

    # ---- objective (minimize) ----
    F = math.sqrt(max(0.0, l2star_sq(pts, d, M)))

    # ---- internal baseline ----
    B = math.sqrt(max(0.0, l2star_sq(lcg_baseline(d, M), d, M)))

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("d=%d M=%d F=%.8f B=%.8f Ratio: %.6f" % (d, M, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
