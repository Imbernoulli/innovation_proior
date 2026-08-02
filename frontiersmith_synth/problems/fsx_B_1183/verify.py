#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- score a wavefront-coefficient estimate by how
well it PREDICTS a held-out defocus image (defocus d3), which the participant
never observes directly. Prints exactly one final line: "... Ratio: <float>".
"""
import sys
import math
import random


def true_coeffs(test_id):
    rng = random.Random(1000003 * test_id + 7)
    trap = test_id not in (1, 2)
    mag0 = 0.60 + 0.35 * rng.random()
    t0 = -mag0 if trap else mag0
    t1 = rng.random() * 0.30 - 0.15
    t2 = rng.random() * 0.30 - 0.15
    t3 = rng.random() * 0.30 - 0.15
    return t0, t1, t2, t3


def intensity(g0, g1, g2, g3, t0, t1, t2, t3, d):
    R = g0 * (t0 + d) + g1 * t1 + g2 * t2 + g3 * t3
    return R * R


def fail(msg):
    print("INVALID: %s  Ratio: 0.0" % msg)
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        toks = f.read().split()
    idx = 0
    try:
        test_id = int(toks[idx]); idx += 1
        s_count = int(toks[idx]); idx += 1
        d1 = float(toks[idx]); idx += 1
        d2 = float(toks[idx]); idx += 1
        d3 = float(toks[idx]); idx += 1
        basis = []
        for _ in range(s_count):
            g0 = float(toks[idx]); idx += 1
            g1 = float(toks[idx]); idx += 1
            g2 = float(toks[idx]); idx += 1
            g3 = float(toks[idx]); idx += 1
            idx += 2  # skip observed I1, I2 (already used to build the artifact)
            basis.append((g0, g1, g2, g3))
    except (IndexError, ValueError):
        fail("malformed instance file")

    try:
        with open(out_path) as f:
            out_toks = f.read().split()
    except FileNotFoundError:
        fail("missing output")

    if len(out_toks) != 4:
        fail("expected exactly 4 numbers (t0 t1 t2 t3), got %d" % len(out_toks))

    try:
        vals = [float(x) for x in out_toks]
    except ValueError:
        fail("non-numeric token in output")

    for v in vals:
        if not math.isfinite(v):
            fail("non-finite coefficient")
        if abs(v) > 10.0:
            fail("coefficient out of the allowed [-10,10] range")

    t0h, t1h, t2h, t3h = vals
    t0, t1, t2, t3 = true_coeffs(test_id)

    err = 0.0
    errB = 0.0
    for (g0, g1, g2, g3) in basis:
        i_true = intensity(g0, g1, g2, g3, t0, t1, t2, t3, d3)
        i_hat = intensity(g0, g1, g2, g3, t0h, t1h, t2h, t3h, d3)
        i_zero = intensity(g0, g1, g2, g3, 0.0, 0.0, 0.0, 0.0, d3)
        err += (i_hat - i_true) ** 2
        errB += (i_zero - i_true) ** 2
    err /= s_count
    errB /= s_count

    eps = 1e-6  # keeps F finite; small relative to the natural scale of err/errB
    F = 1.0 / (err + eps)
    B = 1.0 / (errB + eps)
    raw = 100.0 * F / max(1e-9, B)   # = 100 * (errB+eps) / (err+eps): "how many times
                                      # smaller is my error than the zero-guess error"
    sc = min(900.0, raw)  # explicit cap < 1000 -> strong stays <= 0.90, headroom preserved
    print("F=%.6f B=%.6f raw=%.6f  Ratio: %.6f" % (F, B, raw, sc / 1000.0))


if __name__ == "__main__":
    main()
