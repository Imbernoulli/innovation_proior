# TIER: greedy
# Cobb-Douglas log-linear fit: log y = b0 + b1*log K + b2*log L  (OLS).
# Captures the dominant power-law trend but assumes unit elasticity of
# substitution (rho=0); biased when the true CES rho != 0, and extrapolates
# only moderately well.
import sys, math


def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        if abs(pv) < 1e-15:
            pv = 1e-15
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / pv
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][n] / (M[i][i] if abs(M[i][i]) > 1e-15 else 1e-15) for i in range(n)]


def ols(rows, feats):
    m = len(feats(rows[0][0], rows[0][1]))
    ATA = [[0.0] * m for _ in range(m)]
    ATy = [0.0] * m
    for K, L, y in rows:
        f = feats(K, L)
        ly = math.log(y)
        for i in range(m):
            ATy[i] += f[i] * ly
            for j in range(m):
                ATA[i][j] += f[i] * f[j]
    return solve(ATA, ATy)


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    vals = data[2:]
    rows = [(float(vals[3 * i]), float(vals[3 * i + 1]), float(vals[3 * i + 2]))
            for i in range(n)]

    def feats(K, L):
        return [1.0, math.log(K), math.log(L)]

    b = ols(rows, feats)
    expr = "exp( %r + %r * log(K) + %r * log(L) )" % (b[0], b[1], b[2])
    sys.stdout.write(expr + "\n")


if __name__ == "__main__":
    main()
