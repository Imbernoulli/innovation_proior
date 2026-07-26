# TIER: greedy
# The obvious recipe: fit a flexible black-box polynomial surface of (t,p) to
# the observed size S by ordinary least squares -- S ~ [1, t, t^2, p, t*p].
# It interpolates the narrow interior-band logbook well (a quadratic in t
# comfortably tracks a saturating curve over 10 points, and a linear term in p
# tracks the local trend across the 5 logged nutrient levels), but it never
# assumes -- or recovers -- the underlying multiplicative power-law/saturating
# shape.  The moment p leaves [0.42, 0.58] its straight-line dependence on p
# diverges from the true p^alpha / (1-p)^beta curvature, so it extrapolates
# badly on the extreme-p held-out split.
import sys


def lstsq(A, y):
    m = len(A)
    n = len(A[0])
    M = [[0.0] * (n + 1) for _ in range(n)]
    for i in range(m):
        for j in range(n):
            M[j][n] += A[i][j] * y[i]
            for k in range(n):
                M[j][k] += A[i][j] * A[i][k]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col] or 1e-12
        for k in range(col, n + 1):
            M[col][k] /= pv
        for r in range(n):
            if r != col:
                f = M[r][col]
                for k in range(col, n + 1):
                    M[r][k] -= f * M[col][k]
    return [M[j][n] for j in range(n)]


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0].split()[0])
    A = []
    y = []
    for ln in data[1:1 + n]:
        parts = ln.split()
        if len(parts) >= 3:
            t, p, s = float(parts[0]), float(parts[1]), float(parts[2])
            A.append([1.0, t, t * t, p, t * p])
            y.append(s)
    b = lstsq(A, y)
    print("%r + %r*t + %r*t**2 + %r*p + %r*t*p"
          % (b[0], b[1], b[2], b[3], b[4]))


if __name__ == "__main__":
    main()
