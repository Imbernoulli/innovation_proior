#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of the "Two-Plane Phase Plate" problem to stdout.
Deterministic: all randomness is seeded from testId only (no wall-clock, no OS entropy).
Difficulty ladder: testId 1..10, small/benign -> large/adversarial (see make_instance()).
"""
import sys
import random
import numpy as np


def _blob_field(N, rnd, n, sig_lo, sig_hi, w_lo, w_hi, margin=2.5):
    ii, jj = np.indices((N, N))
    T = np.zeros((N, N))
    for _ in range(n):
        cy = rnd.uniform(margin, N - margin)
        cx = rnd.uniform(margin, N - margin)
        sig = rnd.uniform(sig_lo, sig_hi)
        w = rnd.uniform(w_lo, w_hi)
        T += w * np.exp(-((ii - cy) ** 2 + (jj - cx) ** 2) / (2.0 * sig * sig))
    return T


def _box_blur(T, radius=1):
    """Small separable box blur using only numpy (avoid an external dependency)."""
    N = T.shape[0]
    out = np.zeros_like(T)
    cnt = np.zeros_like(T)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ys, ye = max(0, dy), N + min(0, dy)
            xs, xe = max(0, dx), N + min(0, dx)
            oys, oye = max(0, -dy), N + min(0, -dy)
            oxs, oxe = max(0, -dx), N + min(0, -dx)
            out[oys:oye, oxs:oxe] += T[ys:ye, xs:xe]
            cnt[oys:oye, oxs:oxe] += 1.0
    return out / np.maximum(cnt, 1.0)


def make_instance(test_id):
    rnd = random.Random(9001 * test_id + 17)

    if test_id <= 2:
        N = 16
        aA = 0.006 + 0.002 * test_id
        aB = -0.010 - 0.003 * test_id
        TA = _blob_field(N, rnd, 3, 1.3, 1.8, 0.4, 1.0)
        TB = _blob_field(N, rnd, 3, 1.3, 1.8, 0.4, 1.0)
        lam = 0.02
    elif test_id <= 4:
        N = 20
        aA = 0.012 + 0.002 * test_id
        aB = -0.018 - 0.002 * test_id
        TA = _blob_field(N, rnd, 4, 1.2, 1.8, 0.3, 1.0)
        TB = _blob_field(N, rnd, 4, 1.2, 1.8, 0.3, 1.0)
        lam = 0.03
    elif test_id == 5:
        N = 24
        aA, aB = 0.020, -0.026
        ii, jj = np.indices((N, N))
        r = np.sqrt((ii - N / 2.0) ** 2 + (jj - N / 2.0) ** 2)
        TA = ((r > 5.5) & (r < 8.0)).astype(float) + 0.05
        TB = _blob_field(N, rnd, 3, 1.2, 2.0, 0.4, 1.0)
        lam = 0.04
    elif test_id == 6:
        N = 24
        aA, aB = 0.022, -0.030
        ii, jj = np.indices((N, N))
        TA = np.zeros((N, N))
        for cy, cx, w in [(5, 5, 1.6), (5, 19, 1.2), (19, 5, 0.8), (19, 19, 0.4)]:
            TA += w * np.exp(-((ii - cy) ** 2 + (jj - cx) ** 2) / (2.0 * 1.8 ** 2))
        TB = _blob_field(N, rnd, 4, 1.2, 2.0, 0.3, 1.0)
        lam = 0.04
    elif test_id == 7:
        N = 28
        aA, aB = 0.016, -0.020
        ii, jj = np.indices((N, N))
        block = 4
        TA = (((ii // block + jj // block) % 2) == 0).astype(float) + 0.05
        TB = _blob_field(N, rnd, 3, 1.2, 2.0, 0.4, 1.0)
        lam = 0.05
    elif test_id == 8:
        N = 28
        aA, aB = 0.024, -0.030
        ii, jj = np.indices((N, N))
        TA = np.full((N, N), 0.03)
        TA += 3.0 * np.exp(-((ii - 6) ** 2 + (jj - 22) ** 2) / (2.0 * 0.9 ** 2))
        TB = _blob_field(N, rnd, 5, 1.0, 1.8, 0.3, 1.0)
        lam = 0.05
    elif test_id == 9:
        N = 32
        aA, aB = 0.030, -0.036
        TA = _blob_field(N, rnd, 6, 1.0, 2.0, 0.3, 1.0)
        TB = _blob_field(N, rnd, 6, 1.0, 2.0, 0.3, 1.0)
        lam = 0.06
    else:
        N = 32
        aA, aB = 0.034, -0.040
        rnd2 = random.Random(31337 + test_id)
        TA = np.zeros((N, N))
        for _ in range(14):
            cy = rnd.randint(1, N - 2)
            cx = rnd.randint(1, N - 2)
            TA[cy, cx] += rnd.uniform(0.5, 1.0)
        TA = _box_blur(TA, radius=1) + 0.01
        TB = _blob_field(N, rnd2, 7, 1.0, 1.8, 0.3, 1.0)
        lam = 0.07

    TA = np.maximum(TA, 0.0)
    TB = np.maximum(TB, 0.0)
    return N, aA, aB, lam, TA, TB


def main():
    test_id = int(sys.argv[1])
    N, aA, aB, lam, TA, TB = make_instance(test_id)
    out = []
    out.append(f"{N} {aA:.10f} {aB:.10f} {lam:.10f}")
    for i in range(N):
        out.append(" ".join(f"{v:.10f}" for v in TA[i]))
    for i in range(N):
        out.append(" ".join(f"{v:.10f}" for v in TB[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
