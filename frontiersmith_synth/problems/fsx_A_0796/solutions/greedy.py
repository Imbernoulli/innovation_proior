# TIER: greedy
# The obvious recipe: notice the reading LEVELS OFF rather than growing
# without bound, so fit a rational (Mobius) curve y = (n+B)/(C*n+D) instead
# of a raw polynomial -- a real step up from "just interpolate". But it
# pools ALL rows into ONE cross-multiplied linear least-squares system,
# completely ignoring the gear column s.
#
# Cross-multiplying y = (n+B)/(C*n+D) gives the LINEAR equation (fixing the
# numerator's leading coefficient to 1, since the fit is scale-invariant):
#     n + B - C*(n*y) - D*y = 0   =>   B - C*(n*y) - D*y = -n
# which is solved by ordinary least squares over every training row pooled
# together. The recovered curve is bounded and shaped like the truth, so it
# does not explode numerically -- but because the three gears interact with
# n through DIFFERENT coefficients, one pooled curve is a compromise that
# systematically misses whichever gears its asymptote doesn't happen to be
# close to, and training's modest ship counts never expose that interaction.
import sys


def solve(A, b):
    m = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(m):
        piv = max(range(c, m), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        if abs(d) < 1e-15:
            d = 1e-15
        for r in range(m):
            if r == c:
                continue
            f = M[r][c] / d
            for k in range(c, m + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][m] / (M[i][i] if abs(M[i][i]) > 1e-15 else 1e-15) for i in range(m)]


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    n_rows = int(data[0])
    vals = data[2:]
    rows = []
    for i in range(n_rows):
        n = float(vals[3 * i])
        y = float(vals[3 * i + 2])
        rows.append((n, y))

    m = 3
    Amat = [[0.0] * m for _ in range(m)]
    bb = [0.0] * m
    for n, y in rows:
        x = [1.0, -(n * y), -y]
        target = -n
        for r in range(m):
            bb[r] += x[r] * target
            for c in range(m):
                Amat[r][c] += x[r] * x[c]
    B, C, D = solve(Amat, bb)
    print("( n + %.10g ) / ( %.10g * n + %.10g )" % (B, C, D))


if __name__ == "__main__":
    main()
