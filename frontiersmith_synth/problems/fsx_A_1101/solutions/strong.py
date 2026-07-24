# TIER: strong
# The insight: the parking layout is a data structure.  A row exits in
# exactly the order it was filled, so if every row is filled in
# departure-nondecreasing order the FIFO physics itself reproduces the
# required morning departure permutation and shunting vanishes entirely.
# That is just patience sorting: process buses in arrival order (ties broken
# by earliest departure, which is free since same-hour parking order is ours)
# and put each bus behind the row whose back bus has the largest departure
# hour not exceeding its own (depth-capped at the cable reach W, with a
# minimum-blocking fallback when all R rows are "inverted" at the top).
# Charging: cheapest-feasible-hours EDF, same scheduler as the greedy tier,
# so the entire improvement comes from the layout insight.
import sys


def main():
    data = list(map(int, sys.stdin.read().split()))
    pos = [0]

    def nxt():
        v = data[pos[0]]
        pos[0] += 1
        return v

    n = nxt(); R = nxt(); W = nxt(); T = nxt(); P = nxt(); SH = nxt()
    prc = [nxt() for _ in range(T)]
    cap = [nxt() for _ in range(T)]
    a = []; d = []; E = []
    for _ in range(n):
        a.append(nxt()); d.append(nxt()); E.append(nxt())

    rows = [[] for _ in range(R)]
    row_of = [-1] * n
    park_order = sorted(range(n), key=lambda i: (a[i], d[i], i))
    for i in park_order:
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
        row_of[i] = cand

    exit_order = sorted(range(n), key=lambda i: (d[i], i))

    need = E[:]
    rowhour = [[0] * T for _ in range(R)]
    capleft = cap[:]
    recs = []
    price_order = sorted(range(T), key=lambda h: (prc[h], h))
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
                recs.append((i, h, give))
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
                recs.append((i, h, give))
            if need[i] == 0:
                break

    out = []
    for i in park_order:
        out.append("%d %d" % (i, row_of[i]))
    out.append(" ".join(map(str, exit_order)))
    out.append(str(len(recs)))
    for (i, h, k) in recs:
        out.append("%d %d %d" % (i, h, k))
    print("\n".join(out))


main()
