# TIER: strong
# INSIGHT: the monastery is a resistor network from the heat source (occupied rooms
# at T*) to ground (outside). Total loss = (T* - Tout) * effective conductance of the
# min-cut. What matters is the MARGINAL effect of a layer on the WHOLE network, not a
# wall's standalone heat flow. A buffer room with an uninsulatable stained-glass window
# is a lost cause: its exterior walls are shorted, so insulating them barely helps.
# The real min-cut is often the INTERIOR doorway feeding that leaky buffer -- sealing it
# thermally isolates the occupied room from the short-circuit. We therefore do steepest
# descent on TOTAL loss over ALL insulatable walls (interior included), recomputing the
# marginal each layer so the budget equalizes marginal resistance along the cut.
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

def heat_loss(N, Tstar, Tout, occ, walls, klist):
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
    H = Fr(0)
    for (a, b, R0, rho, km), ge in zip(walls, g):
        if a == 0 or b == 0:
            room = b if a == 0 else a
            H += ge * (Tof[room] - Tout)
    return H

def main():
    N, M, K, Tstar, Tout, occ, walls = read()
    klist = [0] * M
    cand = [i for i, (a, b, R0, rho, km) in enumerate(walls) if km > 0]
    cur = heat_loss(N, Tstar, Tout, occ, walls, klist)
    for _ in range(K):
        best = None; bestH = cur
        for i in cand:
            a, b, R0, rho, km = walls[i]
            if klist[i] >= km:
                continue
            klist[i] += 1
            H = heat_loss(N, Tstar, Tout, occ, walls, klist)
            klist[i] -= 1
            if H < bestH:
                bestH = H; best = i
        if best is None:
            break
        klist[best] += 1
        cur = bestH
    print(" ".join(str(x) for x in klist))

if __name__ == "__main__":
    main()
