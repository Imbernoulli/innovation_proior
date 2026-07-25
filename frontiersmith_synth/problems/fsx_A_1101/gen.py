#!/usr/bin/env python3
"""gen.py -- generator for fsx_A_1101 (Night Depot LIFO).

Usage: python3 gen.py <testId>   (testId 1..10)  prints ONE instance to stdout.
Seeded by testId only; fully deterministic.  Each candidate instance is
screened (deterministic retry loop) so the reference ladder constructions
are feasible and trap cases really punish arrival-order / bucket parking.
"""
import sys
import random

T = 36  # planning horizon in hours (evening + night + morning)

# testId -> (R, W, n, P, SH, inv, tight)
#   inv   = anti-correlation strength between arrival and departure order
#           (block-reversal block size ~ 1 + inv*min(n,R)); large inv = trap
#   tight = valley transformer capacity as a fraction of R*P
SPECS = {
    1:  (3, 2, 5,  6, 15, 0.05, 0.55),
    2:  (3, 3, 8,  6, 18, 0.10, 0.55),
    3:  (4, 3, 11, 7, 20, 0.15, 0.52),
    4:  (4, 3, 12, 7, 28, 0.55, 0.50),   # trap
    5:  (5, 3, 14, 8, 25, 0.25, 0.50),
    6:  (6, 3, 16, 8, 30, 0.35, 0.48),
    7:  (6, 3, 18, 8, 38, 0.70, 0.46),   # trap
    8:  (7, 3, 20, 9, 40, 0.75, 0.45),   # trap
    9:  (8, 3, 22, 9, 45, 0.80, 0.44),   # trap
    10: (8, 3, 24, 10, 50, 0.85, 0.42),  # trap, hardest
}

# ---------------------------------------------------------------------------
# reference ladder constructions (duplicated in verify.py / solutions/*.py)
# ---------------------------------------------------------------------------

def trivial_construct(n, R, W, T_, P, cap, a, d, E):
    order = sorted(range(n), key=lambda i: (a[i], i))
    rows = [[] for _ in range(R)]
    row_of = [-1] * n
    for i in order:
        r = min(range(R), key=lambda r: (len(rows[r]), r))
        rows[r].append(i)
        row_of[i] = r
    need = E[:]
    rowhour = [[0] * T_ for _ in range(R)]
    capleft = cap[:]
    for h in range(T_):
        for r in range(R):
            for i in rows[r]:
                if need[i] > 0 and a[i] <= h < d[i]:
                    give = min(need[i], P - rowhour[r][h], capleft[h])
                    if give > 0:
                        need[i] -= give
                        rowhour[r][h] += give
                        capleft[h] -= give
    for i in sorted(range(n), key=lambda i: (d[i], i)):
        if need[i] == 0:
            continue
        r = row_of[i]
        for h in range(a[i], d[i]):
            give = min(need[i], P - rowhour[r][h], capleft[h])
            if give > 0:
                need[i] -= give
                rowhour[r][h] += give
                capleft[h] -= give
            if need[i] == 0:
                break
    return all(x == 0 for x in need)


def strong_park(n, R, W, a, d):
    rows = [[] for _ in range(R)]
    for i in sorted(range(n), key=lambda i: (a[i], d[i], i)):
        cand = -1
        bestback = -1
        for r in range(R):
            if len(rows[r]) >= W:
                continue
            if rows[r] and d[rows[r][-1]] <= d[i]:
                if d[rows[r][-1]] > bestback:
                    bestback = d[rows[r][-1]]
                    cand = r
        if cand < 0:
            for r in range(R):
                if len(rows[r]) == 0:
                    cand = r
                    break
        if cand < 0:
            bestkey = None
            for r in range(R):
                if len(rows[r]) >= W:
                    continue
                sh = sum(1 for j in rows[r] if d[j] > d[i])
                key = (sh, len(rows[r]), r)
                if bestkey is None or key < bestkey:
                    bestkey = key
                    cand = r
        rows[cand].append(i)
    return rows


def smart_charge(n, R, T_, P, cap, prc, a, d, E, rows):
    row_of = [-1] * n
    for r in range(R):
        for i in rows[r]:
            row_of[i] = r
    need = E[:]
    rowhour = [[0] * T_ for _ in range(R)]
    capleft = cap[:]
    price_order = sorted(range(T_), key=lambda h: (prc[h], h))
    for i in sorted(range(n), key=lambda i: (d[i], i)):
        r = row_of[i]
        for h in price_order:
            if need[i] == 0:
                break
            if h < a[i] or h >= d[i]:
                continue
            give = min(need[i], P - rowhour[r][h], capleft[h])
            if give > 0:
                need[i] -= give
                rowhour[r][h] += give
                capleft[h] -= give
    for i in sorted(range(n), key=lambda i: (d[i], i)):
        if need[i] == 0:
            continue
        r = row_of[i]
        for h in range(a[i], d[i]):
            give = min(need[i], P - rowhour[r][h], capleft[h])
            if give > 0:
                need[i] -= give
                rowhour[r][h] += give
                capleft[h] -= give
            if need[i] == 0:
                break
    return all(x == 0 for x in need)


def row_shunts(rows, d):
    """total shunt moves = #pairs (i ahead of j in a row) with d[i] > d[j]."""
    s = 0
    for row in rows:
        for x in range(len(row)):
            for y in range(x + 1, len(row)):
                if d[row[x]] > d[row[y]]:
                    s += 1
    return s


# ---------------------------------------------------------------------------
# instance construction
# ---------------------------------------------------------------------------

def build(rng, R, W, n, P, SH, inv, tight):
    a = [rng.randint(0, 4) for _ in range(n)]
    E = [rng.randint(P, 2 * P) for _ in range(n)]
    # departure ranks: block-reversed by arrival order (anti-correlated blocks)
    order = sorted(range(n), key=lambda i: (a[i], i))
    Bsz = max(1, min(n, 1 + int(round(inv * min(n, R)))))
    perm = [0] * n
    rank = 0
    for s in range(0, n, Bsz):
        chunk = order[s:s + Bsz]
        for bus in reversed(chunk):
            perm[bus] = rank
            rank += 1
    dvals = sorted(rng.sample(range(12, T), n))
    d = [dvals[perm[i]] for i in range(n)]
    prc = []
    cap = []
    base = R * P
    for h in range(T):
        if h <= 5:
            p, f = 10, 0.90
        elif h <= 10:
            p, f = 6, 0.70
        elif h <= 26:
            p, f = 2 + rng.randint(0, 1), tight
        elif h <= 31:
            p, f = 5, 0.70
        else:
            p, f = 8, 0.60
        prc.append(p)
        cap.append(max(1, int(f * base) + rng.randint(0, max(1, base // 10))))
    return a, d, E, prc, cap


def gen_case(tid):
    rng = random.Random(7919 + tid * 1013)
    R, W, n, P, SH, inv, tight = SPECS[tid]
    for _attempt in range(500):
        a, d, E, prc, cap = build(rng, R, W, n, P, SH, inv, tight)
        if not trivial_construct(n, R, W, T, P, cap, a, d, E):
            continue
        rows = strong_park(n, R, W, a, d)
        if not smart_charge(n, R, T, P, cap, prc, a, d, E, rows):
            continue
        sh = row_shunts(rows, d)
        if inv >= 0.5 and sh != 0:        # trap cases: mirror parking must be clean
            continue
        if inv < 0.5 and sh > 2:
            continue
        # trap cases: the obvious recipe (group buses with similar departure
        # hours into the same row) must really suffer shunts
        if inv >= 0.5:
            S = sorted(range(n), key=lambda i: (d[i], i))
            grows = [[] for _ in range(R)]
            for k, i in enumerate(S):
                grows[k * R // n].append(i)
            # parking sequence is arrival order, so within a row buses keep
            # arrival order: rebuild rows in arrival order
            for r in range(R):
                grows[r].sort(key=lambda i: (a[i], i))
            if row_shunts(grows, d) < max(3, n // 3):
                continue
        return n, R, W, P, SH, prc, cap, a, d, E
    raise SystemExit("gen %d: no feasible instance in 500 tries" % tid)


def main():
    tid = int(sys.argv[1])
    n, R, W, P, SH, prc, cap, a, d, E = gen_case(tid)
    out = []
    out.append("%d %d %d %d %d %d" % (n, R, W, T, P, SH))
    out.append(" ".join(map(str, prc)))
    out.append(" ".join(map(str, cap)))
    for i in range(n):
        out.append("%d %d %d" % (a[i], d[i], E[i]))
    print("\n".join(out))


if __name__ == "__main__":
    main()
