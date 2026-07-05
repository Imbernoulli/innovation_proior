# TIER: strong
"""
Strong: Redlich-Kwong recovery
        P = k*T/(V-b) - a/(sqrt(T)*V*(V+b)).

Matches BOTH structural features of the hidden law: the excluded-volume pole at
V=b (repulsion) AND the temperature-dependent attraction with the V*(V+b) volume
shape. The co-volume b is found by a 1-D grid search; for each b the two linear
coefficients (k, a) come from ordinary least squares. Extrapolates markedly
better than the van-der-Waals form, but the fit is still limited by the
irreducible measurement noise -- so it does not reach a perfect score.
"""
import sys
import math


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


def fit_b(rows, bg):
    S = [[0.0, 0.0], [0.0, 0.0]]
    Sy = [0.0, 0.0]
    for V, T, y in rows:
        if V - bg <= 1e-6:
            return None
        att = 1.0 / (math.sqrt(T) * V * (V + bg))
        g = [T / (V - bg), -att]
        for i in range(2):
            Sy[i] += g[i] * y
            for j in range(2):
                S[i][j] += g[i] * g[j]
    coef = gelim(S, Sy)
    if coef is None:
        return None
    res = 0.0
    for V, T, y in rows:
        att = 1.0 / (math.sqrt(T) * V * (V + bg))
        pr = coef[0] * T / (V - bg) - coef[1] * att
        res += (pr - y) ** 2
    return res, coef[0], coef[1]


def main():
    data = sys.stdin.read().split()
    rows = []
    it = iter(data)
    for a in it:
        b = next(it)
        c = next(it)
        rows.append((float(a), float(b), float(c)))
    best = None
    for i in range(0, 180):
        bg = 0.002 + 0.0005 * i
        r = fit_b(rows, bg)
        if r is None:
            continue
        res, k, a = r
        if best is None or res < best[0]:
            best = (res, bg, k, a)
    _, bg, k, a = best
    print("%.12g * T/(V - %.12g) - %.12g / (sqrt(T)*V*(V + %.12g))"
          % (k, bg, a, bg))


if __name__ == "__main__":
    main()
