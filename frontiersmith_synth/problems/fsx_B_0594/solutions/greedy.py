# TIER: greedy
# The obvious energy-audit heuristic: an auditor walks the outside walls and pours
# insulation onto whichever EXTERIOR wall is currently bleeding the most heat,
# one layer at a time, until the budget runs out. It never touches interior walls
# (the audit does not flag doorways) and cannot touch stained-glass windows.
# This is a *recomputing* greedy over exterior walls -- as good as the audit gets.
import sys
from fractions import Fraction as Fr

def read():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); K = int(next(it))
    Tstar = Fr(next(it)); Tout = Fr(next(it))
    occ = [int(next(it)) for _ in range(N)]
    walls = []
    for _ in range(M):
        a = int(next(it)); b = int(next(it))
        R0 = Fr(next(it)); rho = Fr(next(it)); kmax = int(next(it))
        walls.append((a, b, R0, rho, kmax))
    return N, M, K, Tstar, Tout, occ, walls

def solve(A, bvec, n):
    if n == 0:
        return []
    Mx = [row[:] + [bvec[r]] for r, row in enumerate(A)]
    for c in range(n):
        piv = -1
        for r in range(c, n):
            if Mx[r][c] != 0:
                piv = r; break
        if piv < 0:
            continue
        Mx[c], Mx[piv] = Mx[piv], Mx[c]
        pv = Mx[c][c]
        Mx[c] = [x / pv for x in Mx[c]]
        for r in range(n):
            if r != c and Mx[r][c] != 0:
                f = Mx[r][c]
                Mx[r] = [xr - f * xc for xr, xc in zip(Mx[r], Mx[c])]
    return [Mx[r][n] for r in range(n)]

def temps(N, Tstar, Tout, occ, walls, klist):
    fixedT = {0: Tout}
    for rid in range(1, N + 1):
        if occ[rid - 1]:
            fixedT[rid] = Tstar
    floaters = [rid for rid in range(1, N + 1) if rid not in fixedT]
    idx = {rid: p for p, rid in enumerate(floaters)}
    nf = len(floaters)
    g = [Fr(1) / (R0 + Fr(k) * rho) for (a, b, R0, rho, km), k in zip(walls, klist)]
    A = [[Fr(0)] * nf for _ in range(nf)]
    bvec = [Fr(0)] * nf
    for (a, b, R0, rho, km), ge in zip(walls, g):
        for (i, j) in ((a, b), (b, a)):
            if i in idx:
                p = idx[i]
                A[p][p] += ge
                if j in idx:
                    A[p][idx[j]] -= ge
                else:
                    bvec[p] += ge * fixedT[j]
    T = solve(A, bvec, nf)
    Tof = dict(fixedT)
    for rid in floaters:
        Tof[rid] = T[idx[rid]]
    return Tof, g

def main():
    N, M, K, Tstar, Tout, occ, walls = read()
    klist = [0] * M
    # exterior insulatable walls only
    ext = [i for i, (a, b, R0, rho, km) in enumerate(walls) if (a == 0 or b == 0) and km > 0]
    for _ in range(K):
        Tof, g = temps(N, Tstar, Tout, occ, walls, klist)
        best = None; bestloss = None
        for i in ext:
            a, b, R0, rho, km = walls[i]
            if klist[i] >= km:
                continue
            room = b if a == 0 else a
            loss = g[i] * (Tof[room] - Tout)   # current heat bleeding through this wall
            if bestloss is None or loss > bestloss:
                bestloss = loss; best = i
        if best is None:
            break
        klist[best] += 1
    print(" ".join(str(x) for x in klist))

if __name__ == "__main__":
    main()
