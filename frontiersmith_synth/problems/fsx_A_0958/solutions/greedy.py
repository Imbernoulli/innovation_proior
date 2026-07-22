# TIER: greedy
# Textbook recipe: ordinary-least-squares QUADRATIC SURFACE regression
# y = c0 + c1*f + c2*a + c3*f^2 + c4*a^2 + c5*f*a.
# This is the "obvious" approach once a scatter plot shows curvature -- add
# quadratic terms and fit. Over the small-signal / sub-resonant training box
# a quadratic surface can locally track BOTH the mild saturation curvature
# in `a` and the mild rising curvature in `f` well enough to look plausible.
# But it has no notion of a BOUNDED saturation plateau or a resonance PEAK
# followed by rolloff -- extrapolated into the large-signal / near-and-past
# -resonance held-out region it either explodes (quadratic growth has no
# ceiling) or keeps climbing monotonically right through the peak instead of
# turning over, landing far from the true bounded, peaked response.
import sys


def solve_linear(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[piv] = M[piv], M[i]
        if abs(M[i][i]) < 1e-12:
            continue
        for r in range(i + 1, n):
            fac = M[r][i] / M[i][i]
            for c in range(i, n + 1):
                M[r][c] -= fac * M[i][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0
    return x


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    rows = []
    idx = 1
    for _ in range(n):
        f = float(toks[idx]); a = float(toks[idx + 1]); y = float(toks[idx + 2])
        idx += 3
        rows.append((f, a, y))

    A = [[0.0] * 6 for _ in range(6)]
    bvec = [0.0] * 6
    for f, a, y in rows:
        feats = [1.0, f, a, f * f, a * a, f * a]
        for i in range(6):
            bvec[i] += feats[i] * y
            for j in range(6):
                A[i][j] += feats[i] * feats[j]
    c = solve_linear(A, bvec)

    print("%.10g + %.10g*f + %.10g*a + %.10g*f**2 + %.10g*a**2 + %.10g*f*a"
          % (c[0], c[1], c[2], c[3], c[4], c[5]))


if __name__ == "__main__":
    main()
