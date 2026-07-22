# TIER: strong
# Insight: decompose the composite beauty score into its three named pieces
# and, for each, identify the invariant normalisation that keeps it
# meaningful once the grid gets bigger and the fold order changes --
#   defect term:  D per ORBIT      -> sqrt(D / M)   (not per area, not per motif)
#   motif term:   K per AREA       -> K / A
#   entropy term: already scale-free -> H
# Then fit B ~ 1 + sqrt(D/M) + H + K/A by ordinary least squares. Because
# these three engineered features are themselves scale-invariant, weights
# recovered on small g=8 rosettes still make sense on bigger, finer-order
# kaleidoscopes.
import sys, math


def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        if abs(d) < 1e-18:
            d = 1e-18
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / d
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][n] / (M[i][i] if abs(M[i][i]) > 1e-18 else 1e-18) for i in range(n)]


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    n = int(data[0])
    vals = data[2:]
    feats = []  # [1, sqrt(D/M), H, K/A]
    y = []
    for i in range(n):
        D = float(vals[8 * i + 2])
        Mo = float(vals[8 * i + 3])
        Ar = float(vals[8 * i + 4])
        K = float(vals[8 * i + 5])
        H = float(vals[8 * i + 6])
        B = float(vals[8 * i + 7])
        x1 = math.sqrt(D / Mo) if Mo > 0 else 0.0
        x3 = K / Ar if Ar > 0 else 0.0
        feats.append([1.0, x1, H, x3])
        y.append(B)
    m = 4
    Amat = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for x, yy in zip(feats, y):
        for r in range(m):
            b[r] += x[r] * yy
            for c in range(m):
                Amat[r][c] += x[r] * x[c]
    c0, c1, c2, c3 = solve(Amat, b)
    print("%.10g + %.10g*sqrt(D/M) + %.10g*H + %.10g*(K/A)" % (c0, c1, c2, c3))


if __name__ == "__main__":
    main()
