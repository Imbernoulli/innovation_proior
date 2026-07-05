# TIER: trivial
"""
Trivial baseline: reproduce the checker's internal reference model exactly --
a two-term virial fit  P = k*T/V + c/V**2  by ordinary least squares on the
training log. Emits that expression. Scores ~0.1 by construction.
"""
import sys


def gelim(S, Sy):
    m = len(Sy)
    M = [S[i][:] + [Sy[i]] for i in range(m)]
    for c in range(m):
        piv = max(range(c, m), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-18:
            return None
        for r in range(m):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, m + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][m] / M[i][i] for i in range(m)]


def main():
    data = sys.stdin.read().split()
    rows = []
    it = iter(data)
    for a in it:
        b = next(it)
        c = next(it)
        rows.append((float(a), float(b), float(c)))
    S = [[0.0, 0.0], [0.0, 0.0]]
    Sy = [0.0, 0.0]
    for V, T, y in rows:
        g = [T / V, 1.0 / V ** 2]
        for i in range(2):
            Sy[i] += g[i] * y
            for j in range(2):
                S[i][j] += g[i] * g[j]
    k, c = gelim(S, Sy)
    print("%.12g * T/V + %.12g / V**2" % (k, c))


if __name__ == "__main__":
    main()
