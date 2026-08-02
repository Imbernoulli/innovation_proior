# TIER: strong
"""Insight: use the DIFFERENCE of the two defocus images structurally.

R_s(d) = g0[s]*(t0+d) + g1[s]*t1 + g2[s]*t2 + g3[s]*t3 is LINEAR in d (only
through the g0[s]*d term), so

    I1 - I2 = R(d1)^2 - R(d2)^2 = (R(d1)-R(d2)) * (R(d1)+R(d2))
            = g0*(d1-d2) * (R(d1)+R(d2))

Dividing by the KNOWN g0*(d1-d2) recovers R(d1)+R(d2) directly -- a purely
LINEAR quantity in (t0,t1,t2,t3) with NO sign ambiguity whatsoever (no square
root is ever taken to get it). This gives one clean linear equation per pixel;
solving the resulting 4x4 least squares recovers all four coefficients
exactly (up to floating-point noise). The sign is never guessed -- it falls
straight out of the algebra of the two-image difference.
"""
import sys


def solve4(rows):
    A = [[0.0] * 4 for _ in range(4)]
    b = [0.0] * 4
    for coeffs, y in rows:
        for i in range(4):
            b[i] += coeffs[i] * y
            for j in range(4):
                A[i][j] += coeffs[i] * coeffs[j]
    for i in range(4):
        A[i][i] += 1e-9
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

    rows = []
    for s in range(s_count):
        g0, g1, g2, g3 = basis[s]
        denom = g0 * (d1 - d2)
        Z = (I1[s] - I2[s]) / denom          # = R(d1) + R(d2), no sign ambiguity
        rhs = Z - g0 * (d1 + d2)             # = 2*g0*t0 + 2*g1*t1 + 2*g2*t2 + 2*g3*t3
        rows.append(((2 * g0, 2 * g1, 2 * g2, 2 * g3), rhs))
    t0h, t1h, t2h, t3h = solve4(rows)

    print("%.9f %.9f %.9f %.9f" % (t0h, t1h, t2h, t3h))


if __name__ == "__main__":
    main()
