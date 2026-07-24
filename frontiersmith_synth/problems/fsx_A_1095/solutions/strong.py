# TIER: strong
# Insight: throughflow needs ONE maintained pressure imbalance, not many
# openings. Seed with the best single stack loop (a low inlet paired with a
# high outlet), then run a restraint-aware local search that may ADD a vent
# only when it strictly helps, and crucially may REMOVE or SWAP vents --
# sealing leaks that short-circuit the loop. Exact rational scoring identical
# to the checker is used for every evaluation.
import sys
from fractions import Fraction

LAMBDA = Fraction(7, 8)
EVAL_CAP = 9000


def solve_linear(A, b):
    n = len(A)
    Mx = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = -1
        best = None
        for r in range(col, n):
            v = Mx[r][col]
            if v != 0:
                a = abs(v)
                if best is None or a > best:
                    best = a
                    piv = r
        if piv < 0:
            return None
        if piv != col:
            Mx[col], Mx[piv] = Mx[piv], Mx[col]
        pv = Mx[col][col]
        for r in range(col + 1, n):
            f = Mx[r][col]
            if f == 0:
                continue
            qq = f / pv
            for c in range(col, n + 1):
                if Mx[col][c] != 0:
                    Mx[r][c] -= qq * Mx[col][c]
    x = [Fraction(0)] * n
    for r in range(n - 1, -1, -1):
        s = Mx[r][n]
        for c in range(r + 1, n):
            if Mx[r][c] != 0:
                s -= Mx[r][c] * x[c]
        if Mx[r][r] == 0:
            return None
        x[r] = s / Mx[r][r]
    return x


def evaluate(inst, cut):
    W = inst['W']
    k0 = inst['k0']
    q = inst['q']
    occ = inst['occ']
    doorways = inst['doorways']
    cands = inst['cands']
    openj = list(cut)
    if not openj:
        return Fraction(0)
    G = [int(k0)] * W
    for (c, d, z, g) in doorways:
        G[c] += g
        G[d] += g
    for j in openj:
        c, z, g = cands[j]
        G[c] += g
    t = [Fraction(q[c], G[c]) for c in range(W)]
    A = [[Fraction(0)] * W for _ in range(W)]
    b = [Fraction(0)] * W
    for (c, d, z, g) in doorways:
        A[c][c] += g
        A[d][d] += g
        A[c][d] -= g
        A[d][c] -= g
        rhs = g * (t[c] - t[d]) * z
        b[c] -= rhs
        b[d] += rhs
    for j in openj:
        c, z, g = cands[j]
        A[c][c] += g
        b[c] -= g * t[c] * z
    pi = solve_linear(A, b)
    if pi is None:
        return Fraction(0)
    dflow = [(c, d, g * (pi[c] - pi[d] + (t[c] - t[d]) * z))
             for (c, d, z, g) in doorways]
    oflow = [(cands[j][0],
              cands[j][2] * (pi[cands[j][0]] + t[cands[j][0]] * cands[j][1]))
             for j in openj]
    In = [Fraction(0)] * W
    A2 = [[Fraction(0)] * W for _ in range(W)]
    b2 = [Fraction(0)] * W
    for (c, d, f) in dflow:
        if f > 0:
            In[d] += f
            A2[d][c] -= LAMBDA * f
        elif f < 0:
            In[c] += -f
            A2[c][d] -= LAMBDA * (-f)
    for (c, f) in oflow:
        if f < 0:
            In[c] += -f
            b2[c] += -f
    for c in range(W):
        if In[c] == 0:
            A2[c][c] = Fraction(1)
        else:
            A2[c][c] += In[c]
    phi = solve_linear(A2, b2)
    if phi is None:
        return Fraction(0)
    F = Fraction(0)
    for c in range(W):
        if occ[c]:
            F += phi[c] * In[c]
    return F


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    W = int(next(it))
    B = int(next(it))
    k0 = int(next(it))
    h = [int(next(it)) for _ in range(W)]
    q = [int(next(it)) for _ in range(W)]
    occ = [int(next(it)) for _ in range(W)]
    D = int(next(it))
    doorways = []
    for _ in range(D):
        doorways.append((int(next(it)) - 1, int(next(it)) - 1,
                         int(next(it)), int(next(it))))
    M = int(next(it))
    cands = []
    for _ in range(M):
        cands.append((int(next(it)) - 1, int(next(it)), int(next(it))))
    inst = dict(W=W, B=B, k0=k0, h=h, q=q, occ=occ,
                doorways=doorways, cands=cands)

    cache = {}

    def ev(s):
        key = tuple(s)
        if key not in cache:
            cache[key] = evaluate(inst, key)
        return cache[key]

    # 1) seed: best single stack loop = (low inlet, high outlet) pair
    lows = [j for j in range(M) if cands[j][1] <= 2]
    highs = [j for j in range(M) if cands[j][1] >= h[cands[j][0]] - 1]
    best, bestF = None, Fraction(-1)
    for o in highs:
        for i in lows:
            if i == o:
                continue
            F = ev((i, o))
            if F > bestF:
                bestF, best = F, (i, o)
    if best is None:
        best, bestF = (0, 1), ev((0, 1))
    cur, curF = list(best), bestF

    # 2) restraint-aware local search: add / remove / swap, first improvement
    improved = True
    while improved and len(cache) < EVAL_CAP:
        improved = False
        if len(cur) < B:
            for j in range(M):
                if j in cur:
                    continue
                F = ev(tuple(sorted(cur + [j])))
                if F > curF:
                    cur = sorted(cur + [j])
                    curF = F
                    improved = True
                    break
        if improved:
            continue
        if len(cur) > 2:
            for j in list(cur):
                F = ev(tuple(x for x in cur if x != j))
                if F > curF:
                    cur = [x for x in cur if x != j]
                    curF = F
                    improved = True
                    break
        if improved:
            continue
        curset = set(cur)
        for j in list(cur):
            for kk in range(M):
                if kk in curset:
                    continue
                F = ev(tuple(sorted((curset - {j}) | {kk})))
                if F > curF:
                    cur = sorted((curset - {j}) | {kk})
                    curF = F
                    improved = True
                    break
            if improved:
                break

    sys.stdout.write("%d\n%s\n" % (len(cur), " ".join(str(j + 1) for j in cur)))


main()
