# TIER: greedy
# The obvious recipe: treat each class's wait as a smooth function of the
# total utilization and fit a per-class cubic polynomial w_c ~ b0 + b1*rho +
# b2*rho^2 + b3*rho^3 by least squares on the calm-season rows.  In-sample
# this looks excellent (the calm regime is gentle), but a polynomial stays
# finite as rho -> 1: it cannot represent the capacity pole, and a single
# curve in rho cannot see the berth-priority structure hidden in the mix.
# On storm rows (rho 0.85..0.97) it therefore underpredicts the lower-priority
# classes by orders of magnitude.  relu(...) + 0.001 keeps predictions
# positive when the cubic dives below zero outside the training range.
import sys


def solve_linear(A, b):
    """Gaussian elimination with partial pivoting (small dense system)."""
    m = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(m):
        piv = max(range(col, m), key=lambda rr: abs(M[rr][col]))
        if abs(M[piv][col]) < 1e-12:
            M[piv][col] = 1e-12
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for rr in range(col + 1, m):
            f = M[rr][col] / pv
            for cc in range(col, m + 1):
                M[rr][cc] -= f * M[col][cc]
    x = [0.0] * m
    for i in range(m - 1, -1, -1):
        s = M[i][m] - sum(M[i][j] * x[j] for j in range(i + 1, m))
        x[i] = s / M[i][i]
    return x


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    K = int(toks[1])
    vals = toks[3:]
    rhos = []
    ws = []
    idx = 0
    for _ in range(n):
        rho = float(vals[idx])
        idx += 1 + K
        rhos.append(rho)
        row = []
        for c in range(K):
            row.append(float(vals[idx]))
            idx += 1
        ws.append(row)

    # normal equations for the cubic, shared design matrix
    pw = [[rho ** p for rho in rhos] for p in range(7)]  # powers 0..6
    A = [[sum(pw[j][i] * pw[k][i] for i in range(n)) for k in range(4)]
         for j in range(4)]
    for j in range(4):
        A[j][j] += 1e-9
    out = []
    for c in range(K):
        b = [sum(pw[j][i] * ws[i][c] for i in range(n)) for j in range(4)]
        b0, b1, b2, b3 = solve_linear(A, b)
        out.append(
            "W%d = relu ( %.8f + %.8f * rho + %.8f * rho * rho + %.8f * rho * rho * rho ) + 0.001"
            % (c + 1, b0, b1, b2, b3))
    print("\n".join(out))


if __name__ == "__main__":
    main()
