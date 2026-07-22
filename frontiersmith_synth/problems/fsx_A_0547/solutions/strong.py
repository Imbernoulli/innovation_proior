# TIER: strong
# Insight: the achievable leaf-output set is the zonotope {A x : 0<=x<=cap} — a
# convex cone-like body — so we must PROJECT the target onto it under the box, not
# invert-then-clip. We solve the box-constrained regularized least squares
#     min_x  ||A x - t_s||^2 + cost * sum(x)   s.t. 0 <= x <= cap
# with accelerated projected gradient (FISTA). Because trunk inlets feed many
# leaves (shared ancestors), the box couples them; the projected solve trades the
# few high-influence inlets off against the many, which naive clipping cannot do.
import sys, math


def parse():
    tok = sys.stdin.read().split()
    it = iter(tok)
    nx = lambda: next(it)
    S = int(nx()); I = int(nx()); L = int(nx()); M = int(nx())
    cost = float(nx()); cap = float(nx())
    V = S + I + L
    nodedefs = []
    for _ in range(I + L):
        d = int(nx()); defs = []
        for _ in range(d):
            p = int(nx()); w = float(nx()); defs.append((p, w))
        nodedefs.append(defs)
    targets = [[float(nx()) for _ in range(M)] for _ in range(L)]
    return dict(S=S, I=I, L=L, M=M, cost=cost, cap=cap, V=V,
               nodedefs=nodedefs, targets=targets)


def build_A(inst):
    S, I, L, V = inst["S"], inst["I"], inst["L"], inst["V"]
    coef = [[0.0] * S for _ in range(V)]
    for j in range(S):
        coef[j][j] = 1.0
    for idx, defs in enumerate(inst["nodedefs"]):
        row = coef[S + idx]
        for (p, w) in defs:
            cp = coef[p]
            for s in range(S):
                row[s] += w * cp[s]
    return [coef[S + I + l] for l in range(L)]


def objective(inst, X):
    S, I, L, M, V = inst["S"], inst["I"], inst["L"], inst["M"], inst["V"]
    nodedefs, targets, cost = inst["nodedefs"], inst["targets"], inst["cost"]
    D = 0.0
    for s in range(M):
        val = [0.0] * V
        for j in range(S):
            val[j] = X[j][s]
        for idx, defs in enumerate(nodedefs):
            acc = 0.0
            for (p, w) in defs:
                acc += w * val[p]
            val[S + idx] = acc
        for l in range(L):
            diff = val[S + I + l] - targets[l][s]
            D += diff * diff
    G = sum(X[j][s] for j in range(S) for s in range(M))
    return D + cost * G


def matTvec(A, r):  # A^T r  (S,)  ; A is L x S, r is L
    L = len(A); S = len(A[0])
    out = [0.0] * S
    for l in range(L):
        rl = r[l]; row = A[l]
        for j in range(S):
            out[j] += row[j] * rl
    return out


def matvec(A, x):  # A x  (L,)
    return [sum(A[l][j] * x[j] for j in range(len(x))) for l in range(len(A))]


def lambda_max(ATA, S):
    v = [1.0] * S
    lam = 1.0
    for _ in range(40):
        w = [sum(ATA[a][b] * v[b] for b in range(S)) for a in range(S)]
        nw = math.sqrt(sum(c * c for c in w))
        if nw < 1e-18:
            return 0.0
        lam = nw
        v = [c / nw for c in w]
    return lam


def fista(A, t, cap, cost, S, Lstep, iters=500):
    x = [0.0] * S
    y = x[:]
    tk = 1.0
    step = 1.0 / max(1e-12, Lstep)
    for _ in range(iters):
        Ay = matvec(A, y)
        r = [Ay[l] - t[l] for l in range(len(A))]
        g = matTvec(A, r)
        xnew = [0.0] * S
        for j in range(S):
            v = y[j] - step * (2.0 * g[j] + cost)
            if v < 0.0:
                v = 0.0
            elif v > cap:
                v = cap
            xnew[j] = v
        tnext = (1.0 + math.sqrt(1.0 + 4.0 * tk * tk)) / 2.0
        b = (tk - 1.0) / tnext
        y = [xnew[j] + b * (xnew[j] - x[j]) for j in range(S)]
        x = xnew
        tk = tnext
    return x


def main():
    inst = parse()
    A = build_A(inst)
    S, L, M, cap, cost = inst["S"], inst["L"], inst["M"], inst["cap"], inst["cost"]
    ATA = [[sum(A[l][a] * A[l][b] for l in range(L)) for b in range(S)] for a in range(S)]
    Lstep = 2.0 * lambda_max(ATA, S) + cost + 1e-9

    X = [[0.0] * M for _ in range(S)]
    for s in range(M):
        t = [inst["targets"][l][s] for l in range(L)]
        xs = fista(A, t, cap, cost, S, Lstep, iters=500)
        for j in range(S):
            X[j][s] = xs[j]

    # keep the projected solution unless a trivial fallback is better (safety net)
    fX = objective(inst, X)
    Z = [[0.0] * M for _ in range(S)]
    if objective(inst, Z) < fX:
        X = Z

    lines = [" ".join(repr(X[j][s]) for s in range(M)) for j in range(S)]
    sys.stdout.write("\n".join(lines) + "\n")


main()
