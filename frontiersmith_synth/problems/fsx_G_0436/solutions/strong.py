# TIER: strong
"""Recover the true M/G/1 FORM: OLS on the physically-correct basis
{1, hop, q, cv*q} where q = rho/(1-rho). This recovers the queueing pole from
low-load data, so it extrapolates into the unobserved high-load regime; only the
irreducible held-out noise limits the score."""
import sys


def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            continue
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for j in range(c, n + 1):
            M[c][j] /= piv
        for r in range(n):
            if r != c and abs(M[r][c]) > 0:
                f = M[r][c]
                for j in range(c, n + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][n] for i in range(n)]


def main():
    vals = []
    for tk in sys.stdin.read().split():
        try:
            vals.append(float(tk))
        except ValueError:
            pass
    rows = [vals[i:i + 4] for i in range(0, len(vals) - (len(vals) % 4), 4)]
    feats = []
    ys = []
    for rho, cv, hop, y in rows:
        q = rho / (1.0 - rho)
        feats.append([1.0, hop, q, cv * q])
        ys.append(y)
    k = 4
    AtA = [[0.0] * k for _ in range(k)]
    Atb = [0.0] * k
    for f, y in zip(feats, ys):
        for a in range(k):
            Atb[a] += f[a] * y
            for c in range(k):
                AtA[a][c] += f[a] * f[c]
    w = solve(AtA, Atb)
    # Coefficients as standalone tokens so a nan/inf token-flood breaks the line.
    print("%r + %r * hop + %r * ( rho / ( 1.0 - rho ) ) + %r * ( cv * ( rho / ( 1.0 - rho ) ) )"
          % (w[0], w[1], w[2], w[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
