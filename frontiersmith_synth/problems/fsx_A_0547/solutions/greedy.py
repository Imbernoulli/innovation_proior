# TIER: greedy
# The obvious approach: the leaf<-inlet map is linear, so solve the UNCONSTRAINED
# least-squares inverse per species (normal equations) and clip the result into
# [0,cap]. Ignores the box during the solve and ignores the coupling of shared
# ancestors -> the clip wrecks the match whenever the inverse leaves the box.
import sys


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


def solve_lin(Amat, b):
    # Gaussian elimination for a small symmetric SxS system.
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(Amat)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-15:
            continue
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / pv
            if f != 0.0:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-15 else 0.0 for i in range(n)]


def main():
    inst = parse()
    A = build_A(inst)
    S, L, M, cap = inst["S"], inst["L"], inst["M"], inst["cap"]
    # normal equations ATA x = AT t  (ridge for stability)
    ATA = [[sum(A[l][a] * A[l][b] for l in range(L)) for b in range(S)] for a in range(S)]
    tr = sum(ATA[i][i] for i in range(S))
    ridge = 1e-9 * (tr / max(1, S) + 1.0)
    for i in range(S):
        ATA[i][i] += ridge
    X = [[0.0] * M for _ in range(S)]
    for s in range(M):
        ATt = [sum(A[l][a] * inst["targets"][l][s] for l in range(L)) for a in range(S)]
        xs = solve_lin(ATA, ATt)
        for j in range(S):
            v = xs[j]
            if v < 0.0:
                v = 0.0
            if v > cap:
                v = cap
            X[j][s] = v
    lines = [" ".join(repr(X[j][s]) for s in range(M)) for j in range(S)]
    sys.stdout.write("\n".join(lines) + "\n")


main()
