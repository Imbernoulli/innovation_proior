# TIER: strong
# Model-based coefficient search.  Because the whole quadratic saddle (P, Q, B,
# a, c, x0, y0, T) is public, we can SIMULATE the fixed-form update ourselves and
# grid-search the four coefficients, keeping whichever minimizes the exact final
# duality gap.  We explore per-player step sizes (eta_x != eta_y), optimism
# (theta) and negative/positive momentum (alpha) -- the combination that damps
# the bilinear rotation.  The coefficients are still CONSTANT over all T steps,
# so convergence is only geometric: we land well below the naive baseline but
# strictly above the (unreachable) ideal gap -> a strong, non-perfect score.
import sys, json, math

inst = json.load(sys.stdin)
n = inst["n"]
P, Q, B, a, c = inst["P"], inst["Q"], inst["B"], inst["a"], inst["c"]
x0, y0, T = inst["x0"], inst["y0"], inst["T"]


def matvec(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def matTvec(M, v):
    nn = len(M)
    return [sum(M[i][j] * v[i] for i in range(nn)) for j in range(len(M[0]))]


def solve(A, b):
    m = len(b)
    Mx = [row[:] for row in A]
    x = b[:]
    for k in range(m):
        p = max(range(k, m), key=lambda i: abs(Mx[i][k]))
        if abs(Mx[p][k]) < 1e-15:
            return None
        Mx[k], Mx[p] = Mx[p], Mx[k]
        x[k], x[p] = x[p], x[k]
        for i in range(k + 1, m):
            f = Mx[i][k] / Mx[k][k]
            for j in range(k, m):
                Mx[i][j] -= f * Mx[k][j]
            x[i] -= f * x[k]
    for i in range(m - 1, -1, -1):
        s = x[i] - sum(Mx[i][j] * x[j] for j in range(i + 1, m))
        x[i] = s / Mx[i][i]
    return x


def Lval(x, y):
    Px = matvec(P, x)
    Qy = matvec(Q, y)
    By = matvec(B, y)
    t = 0.5 * sum(x[i] * Px[i] for i in range(n))
    t += sum(x[i] * By[i] for i in range(n))
    t -= 0.5 * sum(y[i] * Qy[i] for i in range(n))
    t += sum(a[i] * x[i] for i in range(n))
    t -= sum(c[i] * y[i] for i in range(n))
    return t


def gap(x, y):
    BTx = matTvec(B, x)
    ys = solve(Q, [BTx[i] - c[i] for i in range(n)])
    By = matvec(B, y)
    xs = solve(P, [-(By[i] + a[i]) for i in range(n)])
    if ys is None or xs is None:
        return 1e12
    g = Lval(x, ys) - Lval(xs, y)
    if not math.isfinite(g):
        return 1e12
    return abs(g)


def run(ex, ey, th, al):
    x = x0[:]
    y = y0[:]
    xp = x[:]
    yp = y[:]
    gxp = None
    gyp = None
    for _ in range(T):
        Px = matvec(P, x)
        By = matvec(B, y)
        BTx = matTvec(B, x)
        Qy = matvec(Q, y)
        gx = [Px[i] + By[i] + a[i] for i in range(n)]
        gy = [BTx[i] - Qy[i] - c[i] for i in range(n)]
        if gxp is None:
            gxp = gx[:]
            gyp = gy[:]
        dx = [(1 + th) * gx[i] - th * gxp[i] for i in range(n)]
        dy = [(1 + th) * gy[i] - th * gyp[i] for i in range(n)]
        xn = [x[i] - ex * dx[i] + al * (x[i] - xp[i]) for i in range(n)]
        yn = [y[i] + ey * dy[i] + al * (y[i] - yp[i]) for i in range(n)]
        for v in xn + yn:
            if not math.isfinite(v) or abs(v) > 1e12:
                return 1e12
        xp, yp = x, y
        gxp, gyp = gx, gy
        x, y = xn, yn
    return gap(x, y)


egrid = [0.006, 0.012, 0.02, 0.035, 0.06, 0.1]
tgrid = [0.0, 0.5, 1.0]
agrid = [-0.4, 0.0, 0.2]

best = 1e18
bc = (0.05, 0.05, 0.0, 0.0)
for ex in egrid:
    for ey in egrid:
        for th in tgrid:
            for al in agrid:
                g = run(ex, ey, th, al)
                if g < best:
                    best = g
                    bc = (ex, ey, th, al)

print(json.dumps({"eta_x": bc[0], "eta_y": bc[1], "theta": bc[2], "alpha": bc[3]}))
