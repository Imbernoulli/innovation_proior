# TIER: greedy
# The obvious first move: the sub-critical census (p, S) rises smoothly, so fit a
# flexible smooth curve -- a cubic least-squares polynomial -- through it and just
# extrapolate that SAME polynomial to the held-out region.  It tracks the training
# census closely (a practitioner would stop here), but a polynomial has no built-in
# threshold and no singular exponent: it has no way to represent a kink at some
# p_c, so extrapolated across and beyond the (unknown) transition it either
# flattens out or runs away, landing far from the true near-critical law.
import sys


def read_rows():
    data = sys.stdin.read().split()
    n = int(data[1])
    rows = []
    for i in range(n):
        p = float(data[3 + 2 * i])
        s = float(data[3 + 2 * i + 1])
        rows.append((p, s))
    return rows


def fit_poly(rows, deg=3):
    n = len(rows)
    X = [[p ** k for k in range(deg + 1)] for p, s in rows]
    y = [s for p, s in rows]
    A = [[0.0] * (deg + 1) for _ in range(deg + 1)]
    b = [0.0] * (deg + 1)
    for i in range(n):
        xi = X[i]
        for a in range(deg + 1):
            for c in range(deg + 1):
                A[a][c] += xi[a] * xi[c]
            b[a] += xi[a] * y[i]
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    m = len(M)
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-14:
            continue
        for r in range(m):
            if r != col:
                f = M[r][col] / M[col][col]
                for c in range(col, m + 1):
                    M[r][c] -= f * M[col][c]
    coef = [M[i][m] / M[i][i] if abs(M[i][i]) > 1e-14 else 0.0 for i in range(m)]
    return coef


def main():
    rows = read_rows()
    coef = fit_poly(rows, deg=3)
    # each numeric coefficient is emitted as its OWN whitespace-separated token
    # (sign folded in) so it reads as a clean float literal, not glued to
    # parentheses/operators -- keeps the expression easy to sanity-check.
    terms = []
    for k, c in enumerate(coef):
        if k == 0:
            terms.append("%.8f" % c)
        else:
            powterm = " * ".join(["p"] * k)
            terms.append("%.8f * %s" % (c, powterm))
    print(" + ".join(terms))


if __name__ == "__main__":
    main()
