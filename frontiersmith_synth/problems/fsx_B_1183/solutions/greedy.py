# TIER: greedy
"""Obvious first approach: fit each image on its own.

Assumes the response R(d) = g0*(t0+d)+g1*t1+g2*t2+g3*t3 is nonnegative and
takes +sqrt(I) of the raw intensity, fits all four coefficients by ordinary
least squares from EACH image independently, then AVERAGES the two per-image
estimates.

This never uses the two images STRUCTURALLY (their difference) -- it treats
them as two separate curve-fitting problems and reconciles by averaging.
Whenever the true response is actually negative for most pixels, both
per-image fits lock onto the wrong (sign-flipped) root and the average stays
wrong.
"""
import sys
import math


def solve4(rows):
    """Ordinary least squares for 4 unknowns via normal equations (Gaussian
    elimination with partial pivoting on the small 4x4 system)."""
    A = [[0.0] * 4 for _ in range(4)]
    b = [0.0] * 4
    for coeffs, y in rows:
        for i in range(4):
            b[i] += coeffs[i] * y
            for j in range(4):
                A[i][j] += coeffs[i] * coeffs[j]
    for i in range(4):
        A[i][i] += 1e-9  # tiny ridge for numerical safety
    # Gaussian elimination with partial pivoting
    for col in range(4):
        piv = max(range(col, 4), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-14:
            continue
        A[col], A[piv] = A[piv], A[col]
        b[col], b[piv] = b[piv], b[col]
        for r in range(col + 1, 4):
            factor = A[r][col] / A[col][col]
            for c in range(col, 4):
                A[r][c] -= factor * A[col][c]
            b[r] -= factor * b[col]
    x = [0.0] * 4
    for i in range(3, -1, -1):
        s = b[i] - sum(A[i][j] * x[j] for j in range(i + 1, 4))
        x[i] = s / A[i][i] if abs(A[i][i]) > 1e-14 else 0.0
    return x


def main():
    data = sys.stdin.read().split()
    idx = 0
    idx += 1  # test_id (unused)
    s_count = int(data[idx]); idx += 1
    d1 = float(data[idx]); idx += 1
    d2 = float(data[idx]); idx += 1
    idx += 1  # d3 (held-out defocus, not needed here)
    basis = []
    I1 = []
    I2 = []
    for _ in range(s_count):
        g0 = float(data[idx]); idx += 1
        g1 = float(data[idx]); idx += 1
        g2 = float(data[idx]); idx += 1
        g3 = float(data[idx]); idx += 1
        i1 = float(data[idx]); idx += 1
        i2 = float(data[idx]); idx += 1
        basis.append((g0, g1, g2, g3))
        I1.append(i1)
        I2.append(i2)

    rows1 = []
    rows2 = []
    for s in range(s_count):
        g0, g1, g2, g3 = basis[s]
        rows1.append(((g0, g1, g2, g3), math.sqrt(max(0.0, I1[s]))))
        rows2.append(((g0, g1, g2, g3), math.sqrt(max(0.0, I2[s]))))
    est1 = solve4(rows1)   # estimate of (t0+d1, t1, t2, t3)
    est2 = solve4(rows2)   # estimate of (t0+d2, t1, t2, t3)

    t0_1 = est1[0] - d1
    t0_2 = est2[0] - d2
    t0h = 0.5 * (t0_1 + t0_2)
    t1h = 0.5 * (est1[1] + est2[1])
    t2h = 0.5 * (est1[2] + est2[2])
    t3h = 0.5 * (est1[3] + est2[3])

    print("%.9f %.9f %.9f %.9f" % (t0h, t1h, t2h, t3h))


if __name__ == "__main__":
    main()
