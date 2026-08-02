# TIER: greedy
# The obvious recipe: "band gap vs composition" is a textbook Vegard's-law +
# bowing-parameter curve, y = a + b*x + c*x^2, fit by ordinary least squares
# to the training rows.  This IGNORES the dopant descriptors (dEN, dR)
# entirely, silently averaging over the whole visible chemistry family.  It
# is an honest, accurate INTERPOLATOR across the visible composition range
# (it even beats the plain straight line, because the bowing curvature is a
# real, chemistry-independent effect) -- but it has zero mechanism to react
# to a dopant whose electronegativity/radius mismatch lies outside the
# training family, so it fails hard on held-out chemistry.
import sys


def solve3(A, b):
    # tiny 3x3 Gaussian elimination with partial pivoting
    m = 3
    M = [A[i][:] + [b[i]] for i in range(m)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12:
            continue
        for r in range(m):
            if r != col:
                f = M[r][col] / M[col][col]
                for c in range(col, m + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][m] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(m)]


def main():
    data = sys.stdin.read().split()
    n = int(data[1])
    xs = []
    ys = []
    for i in range(n):
        base = 2 + 5 * i
        xs.append(float(data[base + 1]))
        ys.append(float(data[base + 4]))

    # normal equations for y = c0 + c1*x + c2*x^2
    feats = [[1.0, x, x * x] for x in xs]
    XtX = [[0.0] * 3 for _ in range(3)]
    Xty = [0.0] * 3
    for f, y in zip(feats, ys):
        for i in range(3):
            Xty[i] += f[i] * y
            for j in range(3):
                XtX[i][j] += f[i] * f[j]
    c = solve3(XtX, Xty)

    print("(%.6f) + (%.6f) * x + (%.6f) * x ** 2" % (c[0], c[1], c[2]))


if __name__ == "__main__":
    main()
