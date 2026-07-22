# TIER: strong
import math
import sys


def softplus(z):
    if z > 45.0:
        return z
    if z < -45.0:
        return math.exp(z)
    return math.log1p(math.exp(z))


def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-10:
            M[c][c] = 1e-10
        pv = M[c][c]
        for j in range(c, n + 1):
            M[c][j] /= pv
        for r in range(n):
            if r == c:
                continue
            f = M[r][c]
            if f:
                for j in range(c, n + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][n] for i in range(n)]


def features(row, alpha, bt, bo, gamma, bh, knee):
    N, T, D, R, _ = row
    u = N / 100.0
    th = (T - 20.0) / 12.0
    stress = D * R
    act = 0.55 * softplus((u - knee) / 0.55)
    return [
        1.0,
        (u ** alpha) * math.exp(bt * th) * (D ** 1.05),
        u * math.exp(bo * th) * stress,
        (act ** gamma) * math.exp(bh * th) * (stress ** 1.15),
        (u ** 0.80) * math.exp(0.35 * th) * D * (0.25 + R) * (0.8 + 0.4 * th),
    ]


def fit_active(rows, params, active):
    m = len(active)
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    ridge = 1e-6
    feats = []
    for row in rows:
        f = features(row, *params)
        feats.append(f)
        y = row[4]
        for ii, i in enumerate(active):
            b[ii] += f[i] * y
            for jj, j in enumerate(active):
                A[ii][jj] += f[i] * f[j]
    for i in range(m):
        A[i][i] += ridge
    subc = solve(A, b)
    c = [0.0] * 5
    for ii, i in enumerate(active):
        c[i] = subc[ii]
    # The pilot data are intentionally cold and early-cycle, so the activated
    # mechanism is weakly identified.  Clamp amplitudes to broad physical
    # ranges before extrapolating instead of letting collinearity invent a huge
    # high-temperature knee term.
    lo = [-0.02, 0.0, 0.0, 0.0, 0.0]
    hi = [0.08, 0.25, 0.09, 0.028, 0.025]
    for i in range(5):
        if c[i] < lo[i]:
            c[i] = lo[i]
        if c[i] > hi[i]:
            c[i] = hi[i]
    sse = 0.0
    for row, f in zip(rows, feats):
        pred = sum(c[i] * f[i] for i in range(5))
        sse += (pred - row[4]) ** 2
    return sse, c


def fit_nonnegative(rows, params):
    active = [0, 1, 2, 3, 4]
    best_sse, best_c = fit_active(rows, params, active)
    for _ in range(4):
        negatives = [(best_c[i], i) for i in active if i != 0 and best_c[i] < -1e-7]
        if not negatives:
            return best_sse, best_c
        _, drop = min(negatives)
        active.remove(drop)
        best_sse, best_c = fit_active(rows, params, active)
    return best_sse, best_c


def main():
    lines = sys.stdin.read().strip().splitlines()
    n = int(lines[0].split()[0])
    rows = []
    for line in lines[1:1 + n]:
        p = line.split()
        rows.append(tuple(float(x) for x in p[:5]))

    alphas = [0.46, 0.50, 0.54, 0.58, 0.62]
    bts = [0.20, 0.28, 0.36]
    bos = [0.38, 0.48, 0.58]
    gammas = [1.30, 1.45]
    bhs = [0.50, 0.62, 0.74]
    knees = [1.45, 1.70, 1.95, 2.20]

    best = None
    for alpha in alphas:
        for bt in bts:
            for bo in bos:
                for gamma in gammas:
                    for bh in bhs:
                        for knee in knees:
                            params = (alpha, bt, bo, gamma, bh, knee)
                            sse, c = fit_nonnegative(rows, params)
                            # Prefer simpler, smoother extrapolations when train SSE ties.
                            key = sse * (1.0 + 0.0002 * (abs(c[3]) + abs(c[4])))
                            if best is None or key < best[0]:
                                best = (key, params, c)

    _, params, c = best
    alpha, bt, bo, gamma, bh, knee = params
    act = "(0.55*log(1.0+exp(((N/100.0)-%r)/0.55)))" % knee
    expr = (
        "%r + %r*(N/100.0)**%r*exp(%r*((T-20.0)/12.0))*D**1.05 "
        "+ %r*(N/100.0)*exp(%r*((T-20.0)/12.0))*D*R "
        "+ %r*(%s)**%r*exp(%r*((T-20.0)/12.0))*(D*R)**1.15 "
        "+ %r*(N/100.0)**0.8*exp(0.35*((T-20.0)/12.0))*D*(0.25+R)*(0.8+0.4*((T-20.0)/12.0))"
    ) % (c[0], c[1], alpha, bt, c[2], bo, c[3], act, gamma, bh, c[4])
    sys.stdout.write(expr + "\n")


if __name__ == "__main__":
    main()
