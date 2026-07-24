# TIER: greedy
# The obvious recipe: group buses with similar departure hours into the same
# row (sorted-by-departure contiguous buckets), park in arrival order, and
# charge each bus in its cheapest hours before its deadline (EDF pass).
# Ignores that exit physics needs arrival order == departure order per row,
# so on anti-correlated instances the buckets are full of blocking pairs.
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

    # bucket rows by departure rank
    S = sorted(range(n), key=lambda i: (d[i], i))
    row_of = [0] * n
    for k, i in enumerate(S):
        row_of[i] = k * R // n
    park_order = sorted(range(n), key=lambda i: (a[i], i))
    exit_order = S[:]

    # per-bus cheapest-hours charging, earliest-deadline-first
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
