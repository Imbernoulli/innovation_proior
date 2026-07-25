# TIER: greedy
# The obvious recipe: pooled black-box regression.  Fit each post-impact
# velocity as a linear combination of polynomial features (including the
# classical momenta m1*v1, m2*v2 and damping interactions) by ridge least
# squares over ALL rigs pooled together.  This interpolates well inside the
# training mass range [1,8], but the true law's mass scaling m**alpha with
# alpha != 1 and its exponential damping/speed dependence are the WRONG
# functional family -- so on the held-out rig (masses 10-30, hard ids also
# extrapolated damping) the fit extrapolates catastrophically.
import sys


def solve(A, b):
    """Gaussian elimination with partial pivoting (A is n x n, b length n)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            M[col][col] += 1e-8
            piv = col
        M[col], M[piv] = M[piv], M[col]
        d = M[col][col]
        for r in range(col + 1, n):
            f = M[r][col] / d
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n] - sum(M[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / M[r][r]
    return x


def feats(m1, m2, g, v1, v2):
    w = v2 - v1
    return [1.0, v1, v2, m1 * v1, m2 * v2, m1 * v2, m2 * v1, g * w, g * v1, g * v2]


def main():
    data = sys.stdin.read().split()
    if not data:
        print("V1 v1\nV2 v2")
        return
    n = int(data[0])
    rigs = []
    rows = []
    i = 3
    while i < len(data):
        if data[i] == "RIG":
            rigs.append(tuple(float(data[i + j]) for j in (1, 2, 3, 4)))
            i += 5
        elif data[i] == "ROW":
            rows.append((int(data[i + 1]),) + tuple(float(data[i + j]) for j in (2, 3, 4, 5)))
            i += 6
        else:
            i += 1

    p = len(feats(1, 1, 0, 0, 0))
    A = [[0.0] * p for _ in range(p)]
    b1 = [0.0] * p
    b2 = [0.0] * p
    for (ri, v1, v2, y1, y2) in rows:
        m1, m2, g, dt = rigs[ri]
        f = feats(m1, m2, g, v1, v2)
        for a in range(p):
            b1[a] += f[a] * y1
            b2[a] += f[a] * y2
            for c in range(p):
                A[a][c] += f[a] * f[c]
    for a in range(p):
        A[a][a] += 1e-6 * n
    c1 = solve(A, b1)
    c2 = solve(A, b2)

    def expr(c):
        terms = ["%.10f" % c[0], "%.10f * v1" % c[1], "%.10f * v2" % c[2],
                 "%.10f * m1 * v1" % c[3], "%.10f * m2 * v2" % c[4],
                 "%.10f * m1 * v2" % c[5], "%.10f * m2 * v1" % c[6],
                 "%.10f * g * ( v2 - v1 )" % c[7],
                 "%.10f * g * v1" % c[8], "%.10f * g * v2" % c[9]]
        return "( " + " ) + ( ".join(terms) + " )"

    print("V1 " + expr(c1))
    print("V2 " + expr(c2))


if __name__ == "__main__":
    main()
