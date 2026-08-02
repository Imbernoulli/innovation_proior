#!/usr/bin/env python3
"""gen.py <testId> -- one phase-diversity wavefront-sensing instance to stdout.

Deterministic: everything derives from testId only (random.Random seeded from
testId). testId 1..2 are the "easy" ladder rungs (true even-mode sign matches
the naive nonneg-sqrt convention). testId 3..10 are TRAP cases (true even-mode
sign is negative, defeating that convention) with growing pixel count.
"""
import sys
import random


def true_coeffs(test_id):
    rng = random.Random(1000003 * test_id + 7)
    trap = test_id not in (1, 2)
    mag0 = 0.60 + 0.35 * rng.random()      # |t0| in [0.60, 0.95]
    t0 = -mag0 if trap else mag0
    t1 = rng.random() * 0.30 - 0.15        # [-0.15, 0.15]
    t2 = rng.random() * 0.30 - 0.15        # [-0.15, 0.15]
    t3 = rng.random() * 0.30 - 0.15        # [-0.15, 0.15]
    return t0, t1, t2, t3


def defocus_settings(test_id):
    rng = random.Random(2000003 * test_id + 17)
    d1 = -0.30 + 0.05 * rng.random()       # near -0.30
    d2 = 0.22 + 0.05 * rng.random()        # near +0.22
    d3 = 0.06 + 0.04 * rng.random()        # held-out defocus, distinct from d1,d2
    return d1, d2, d3


def num_pixels(test_id):
    return 8 + 2 * (test_id - 1)           # 8, 10, ..., 26


def pixel_basis(test_id, s_count):
    rng = random.Random(3000003 * test_id + 29)
    rows = []
    for _ in range(s_count):
        g0 = 0.5 + 1.0 * rng.random()      # strictly positive: 0.5..1.5
        g1 = -1.0 + 2.0 * rng.random()     # -1..1
        g2 = -1.0 + 2.0 * rng.random()     # -1..1
        g3 = -1.0 + 2.0 * rng.random()     # -1..1
        rows.append((g0, g1, g2, g3))
    return rows


def response(g0, g1, g2, g3, t0, t1, t2, t3, d):
    """R_s(d): the pupil field's real response at defocus d. t0 is the only
    coefficient that couples to defocus (t0+d); t1,t2,t3 are fixed modes."""
    return g0 * (t0 + d) + g1 * t1 + g2 * t2 + g3 * t3


def intensity(g0, g1, g2, g3, t0, t1, t2, t3, d):
    R = response(g0, g1, g2, g3, t0, t1, t2, t3, d)
    return R * R


def main():
    test_id = int(sys.argv[1])
    s_count = num_pixels(test_id)
    t0, t1, t2, t3 = true_coeffs(test_id)
    d1, d2, d3 = defocus_settings(test_id)
    basis = pixel_basis(test_id, s_count)

    out = [f"{test_id} {s_count}", f"{d1:.9f} {d2:.9f} {d3:.9f}"]
    for (g0, g1, g2, g3) in basis:
        I1 = intensity(g0, g1, g2, g3, t0, t1, t2, t3, d1)
        I2 = intensity(g0, g1, g2, g3, t0, t1, t2, t3, d2)
        out.append(f"{g0:.9f} {g1:.9f} {g2:.9f} {g3:.9f} {I1:.9f} {I2:.9f}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
