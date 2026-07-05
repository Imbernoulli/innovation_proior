# TIER: greedy
# Log-linear fit:  L ~ p + q*log(C) + r*log(D).
# Captures the decreasing trend in log-compute/log-data and extrapolates far
# better than a constant, but it has NO irreducible floor, so it keeps sliding
# below the true entropy floor in the large-scale held-out region.
import sys, math


def solve3(M, y):
    # Gaussian elimination on a 3x3 normal-equation system.
    A = [row[:] + [y[i]] for i, row in enumerate(M)]
    for c in range(3):
        piv = max(range(c, 3), key=lambda r: abs(A[r][c]))
        A[c], A[piv] = A[piv], A[c]
        if abs(A[c][c]) < 1e-12:
            A[c][c] = 1e-12
        pv = A[c][c]
        for j in range(c, 4):
            A[c][j] /= pv
        for r in range(3):
            if r != c:
                f = A[r][c]
                for j in range(c, 4):
                    A[r][j] -= f * A[c][j]
    return [A[0][3], A[1][3], A[2][3]]


def main():
    data = sys.stdin.read().split("\n")
    header = data[0].split()
    n = int(header[0])
    rows = []
    for line in data[1:1 + n]:
        p = line.split()
        if len(p) < 3:
            continue
        rows.append((float(p[0]), float(p[1]), float(p[2])))

    # design columns f = [1, log C, log D]
    M = [[0.0] * 3 for _ in range(3)]
    y = [0.0] * 3
    for C, D, L in rows:
        f = [1.0, math.log(C), math.log(D)]
        for i in range(3):
            for j in range(3):
                M[i][j] += f[i] * f[j]
            y[i] += f[i] * L
    p, q, r = solve3(M, y)
    sys.stdout.write("%r + %r*log(C) + %r*log(D)\n" % (p, q, r))


if __name__ == "__main__":
    main()
