# TIER: trivial
# Balanced parking in arrival order (row with fewest buses), exit by departure
# hour, charge as early as possible at full rate.  Mirrors the checker's
# internal reference construction exactly.
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

    order = sorted(range(n), key=lambda i: (a[i], i))
    rows = [[] for _ in range(R)]
    row_of = [-1] * n
    for i in order:
        r = min(range(R), key=lambda r: (len(rows[r]), r))
        rows[r].append(i)
        row_of[i] = r
    exit_order = sorted(range(n), key=lambda i: (d[i], i))

    need = E[:]
    rowhour = [[0] * T for _ in range(R)]
    capleft = cap[:]
    recs = []
    for h in range(T):
        for r in range(R):
            for i in rows[r]:
                if need[i] > 0 and a[i] <= h < d[i]:
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
    for i in order:
        out.append("%d %d" % (i, row_of[i]))
    out.append(" ".join(map(str, exit_order)))
    out.append(str(len(recs)))
    for (i, h, k) in recs:
        out.append("%d %d %d" % (i, h, k))
    print("\n".join(out))


main()
