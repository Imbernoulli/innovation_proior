#!/usr/bin/env python3
"""gen.py <testId> -- generator for fsx_A_1095 (The Sighing Caravanserai).

Deterministic: the rng is seeded by testId only. Each candidate instance is
scored internally with the EXACT checker model (embedded copy) and rejected
unless it provably shows the required structure:
  * the minimal hall loop (vents 1,2) gives baseline F0 > 0,
  * some (low inlet, high outlet) pair reaches >= 3x baseline (headroom),
  * trap stages (>=5): the top-conductance budget-fill reaches <= 0.75 of the
    best pair (the obvious greedy is punished),
  * gentle stages (<=4): the top-conductance fill still beats 1.2x baseline
    (the ladder is sane).
"""
import sys
import random
from fractions import Fraction

LAMBDA = Fraction(7, 8)


# ---------------- exact scorer (identical logic to verify.py) ----------------
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


# ----------------------------- instance builder -----------------------------
def build_candidate(rng, stage):
    W = min(9, 4 + (stage + 1) // 2)          # 5..9 bays
    k0 = rng.randint(2, 6)
    hall = rng.randrange(W)
    h, q, occ = [], [], []
    for c in range(W):
        if c == hall:
            h.append(rng.randint(34, 56))     # the tall great hall (stack shaft)
            q.append(rng.randint(110, 220))
            occ.append(1)
        else:
            h.append(rng.randint(8, 16))
            if rng.random() < 0.75:
                occ.append(1)
                q.append(rng.randint(15, 60))
            else:
                occ.append(0)
                q.append(0 if rng.random() < 0.6 else rng.randint(10, 30))
    if not any(occ[c] for c in range(W) if c != hall):
        c = rng.choice([c for c in range(W) if c != hall])
        occ[c] = 1
        q[c] = rng.randint(15, 60)
    gh = rng.randint(6, 12)
    doorways = [(c, c + 1, 0, gh) for c in range(W - 1)]   # the connected row
    if stage >= 4:
        for _ in range(rng.randint(1, 2)):               # upper corridors
            a, bb = rng.sample(range(W), 2)
            z = rng.randint(3, min(h[a], h[bb]) - 1)
            doorways.append((a, bb, z, rng.randint(2, 5)))

    room_idxs = [c for c in range(W) if occ[c] and c != hall]

    # positions 0,1: the minimal hall loop = the checker's baseline anchor
    cands = [(hall, 0, rng.randint(2, 4)), (hall, h[hall], rng.randint(2, 4))]
    rest = []
    for c in range(W):
        if c != hall and occ[c]:
            rest.append((c, 0, rng.randint(2, 5)))       # room floor inlet
            rest.append((c, h[c], rng.randint(2, 5)))    # room roof vent

    if stage <= 4:
        # gentle: tight budget, some genuinely good vents, a few weak leaks
        for c in room_idxs[:2]:
            rest.append((c, 0, rng.randint(8, 12)))      # big room inlet
        rest.append((hall, h[hall], rng.randint(7, 10)))  # big hall lantern
        for _ in range(1 + stage // 2):
            z = rng.randint(h[hall] // 4, h[hall] // 2)
            rest.append((hall, z, rng.randint(3, 6)))    # weak-ish leaks
        B = max(3, (len(rest) + 2) // 3)
    elif stage <= 7:
        # big-leak trap: the largest-g candidates are mid-height hall leaks
        # and mid-height room windows; filling the budget short-circuits flow
        for _ in range(3 + stage // 2):
            z = rng.randint(h[hall] // 4, h[hall] // 2)
            rest.append((hall, z, rng.randint(12, 20)))
        for c in room_idxs:
            rest.append((c, h[c] // 2, rng.randint(8, 14)))
        B = max(3, (2 * (len(rest) + 2)) // 3)
    else:
        # open-everything trap: budget covers ALL vents; cutting everything
        # equalizes pressures and drains the hall's warmth
        for _ in range(3 + stage // 2):
            z = rng.randint(h[hall] // 4, h[hall] // 2)
            rest.append((hall, z, rng.randint(12, 20)))
        unheated = [c for c in range(W) if q[c] == 0 and c != hall] or \
                   [c for c in range(W) if c != hall]
        for _ in range(stage // 2):
            c = rng.choice(unheated)
            rest.append((c, h[c], rng.randint(8, 14)))   # short-shaft top vents
        for c in room_idxs:
            rest.append((c, h[c] // 2, rng.randint(8, 14)))
        B = len(rest) + 2

    rng.shuffle(rest)
    cands += rest
    return dict(W=W, B=B, k0=k0, h=h, q=q, occ=occ,
                doorways=doorways, cands=cands)


def greedy_set(inst):
    M = len(inst['cands'])
    order = sorted(range(M), key=lambda j: (-inst['cands'][j][2], j))
    return sorted(order[:inst['B']])


def accept(inst, stage):
    M = len(inst['cands'])
    Ftriv = evaluate(inst, [0, 1])
    if Ftriv <= 0:
        return False
    lows = [j for j in range(M) if inst['cands'][j][1] <= 2]
    highs = [j for j in range(M)
             if inst['cands'][j][1] >= inst['h'][inst['cands'][j][0]] - 1]
    best = Fraction(0)
    for o in highs:
        for i in lows:
            if i != o:
                F = evaluate(inst, [i, o])
                if F > best:
                    best = F
    if best < 3 * Ftriv:
        return False
    Fg = evaluate(inst, greedy_set(inst))
    if stage >= 5:
        return 4 * Fg <= 3 * best        # greedy punished: <= 0.75 of best pair
    return 5 * Fg >= 6 * Ftriv           # ladder sane: greedy >= 1.2 baseline


def main():
    test_id = int(sys.argv[1])
    if not (1 <= test_id <= 10):
        raise SystemExit("testId must be 1..10")
    stage = test_id
    rng = random.Random(910000 + test_id * 7919)
    for _ in range(6000):
        inst = build_candidate(rng, stage)
        if accept(inst, stage):
            W = inst['W']
            print(W, inst['B'], inst['k0'])
            print(" ".join(map(str, inst['h'])))
            print(" ".join(map(str, inst['q'])))
            print(" ".join(map(str, inst['occ'])))
            print(len(inst['doorways']))
            for (c, d, z, g) in inst['doorways']:
                print(c + 1, d + 1, z, g)
            print(len(inst['cands']))
            for (c, z, g) in inst['cands']:
                print(c + 1, z, g)
            return
    raise SystemExit("gen %d: no acceptable instance found" % test_id)


if __name__ == "__main__":
    main()
