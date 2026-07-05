# TIER: strong
"""Regime-aware fit: detect the break location on x0 by scanning candidate
thresholds, then least-squares fit on the hinge basis
{1, x0, x1, x2*x3, x3, max(0, x0-C)^2}. Recovering the piecewise FORM lets the
fit extrapolate across the break into the far tipping regime. Irreducible
held-out noise (and finite noisy far-side train data) keeps it below saturation."""
import sys


def solve_ls(A, b):
    n = len(A[0])
    M = [[0.0] * n for _ in range(n)]
    v = [0.0] * n
    for row, y in zip(A, b):
        for i in range(n):
            v[i] += row[i] * y
            for j in range(n):
                M[i][j] += row[i] * row[j]
    for i in range(n):
        M[i][i] += 1e-9
    for col in range(n):
        p = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[p] = M[p], M[col]
        v[col], v[p] = v[p], v[col]
        piv = M[col][col]
        if abs(piv) < 1e-12:
            continue
        for r in range(n):
            if r == col:
                continue
            f = M[r][col] / piv
            if f == 0.0:
                continue
            for k in range(col, n):
                M[r][k] -= f * M[col][k]
            v[r] -= f * v[col]
    return [v[i] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


def design(rows, C):
    A, b = [], []
    for (x0, x1, x2, x3, y) in rows:
        h = x0 - C if x0 > C else 0.0
        A.append([1.0, x0, x1, x2 * x3, x3, h * h])
        b.append(y)
    return A, b


def sse(rows, C, c):
    s = 0.0
    for (x0, x1, x2, x3, y) in rows:
        h = x0 - C if x0 > C else 0.0
        pred = c[0] + c[1] * x0 + c[2] * x1 + c[3] * (x2 * x3) + c[4] * x3 + c[5] * h * h
        s += (pred - y) ** 2
    return s


def main():
    toks = sys.stdin.read().split()
    vals = [float(t) for t in toks]
    rows = []
    for i in range(0, len(vals) - 4, 5):
        rows.append(tuple(vals[i:i + 5]))

    # scan candidate break locations; pick the one minimizing train SSE
    best = None
    Ccand = [x / 100.0 for x in range(-20, 90, 5)]   # -0.2 .. 0.85
    for C in Ccand:
        A, b = design(rows, C)
        c = solve_ls(A, b)
        s = sse(rows, C, c)
        if best is None or s < best[0]:
            best = (s, C, c)
    _, C, c = best

    expr = ("%.8f + (%.8f)*x0 + (%.8f)*x1 + (%.8f)*(x2*x3) + (%.8f)*x3 "
            "+ (%.8f)*(((x0-(%.6f)) + abs(x0-(%.6f)))/2.0)**2") % (
        c[0], c[1], c[2], c[3], c[4], c[5], C, C)
    sys.stdout.write(expr + "\n")


if __name__ == "__main__":
    main()
