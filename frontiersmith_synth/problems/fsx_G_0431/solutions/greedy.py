# TIER: greedy
"""Global smooth fit that IGNORES the regime break: ordinary least squares on the
basis {1, x0, x1, x2*x3, x3, x0^2}. It captures the non-breaking drivers and even
a global curvature term, but a single unbroken quadratic fit on the (mostly
near-side) train data underestimates the far-side tipping feedback, so it
extrapolates poorly across the break."""
import sys


def solve_ls(A, b):
    n = len(A[0])
    # normal equations M c = v
    M = [[0.0] * n for _ in range(n)]
    v = [0.0] * n
    for row, y in zip(A, b):
        for i in range(n):
            v[i] += row[i] * y
            for j in range(n):
                M[i][j] += row[i] * row[j]
    # Gaussian elimination with partial pivoting + tiny ridge for stability
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


def main():
    toks = sys.stdin.read().split()
    vals = [float(t) for t in toks]
    A, b = [], []
    for i in range(0, len(vals) - 4, 5):
        x0, x1, x2, x3, y = vals[i:i + 5]
        A.append([1.0, x0, x1, x2 * x3, x3, x0 * x0])
        b.append(y)
    c = solve_ls(A, b)
    expr = ("%.8f + (%.8f)*x0 + (%.8f)*x1 + (%.8f)*(x2*x3) + (%.8f)*x3 "
            "+ (%.8f)*(x0**2)") % (c[0], c[1], c[2], c[3], c[4], c[5])
    sys.stdout.write(expr + "\n")


if __name__ == "__main__":
    main()
