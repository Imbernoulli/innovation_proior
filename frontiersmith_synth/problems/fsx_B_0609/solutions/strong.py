# TIER: strong
# Insight: the insert stream is FULLY VISIBLE, so gap placement is a transport
# problem -- move slack to exactly where future demand lands -- not a uniform
# worst-case hedge. We scan the trace, count how many inserts fall into each
# gap BETWEEN consecutive initial keys, and carve a reservoir there sized to that
# demand. Cold intervals (no future insert) are left PACKED SOLID so scans over
# them touch no gaps at all. Trailing capacity is parked far to the right where
# no scan reaches. Result: inserts land in their reservoirs near-free while the
# scan-heavy cold regions stay dense.
import sys, bisect


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it)); M = int(next(it)); Q = int(next(it))
    C = int(next(it)); VMAX = int(next(it))
    keys = [int(next(it)) for _ in range(N)]
    demand = [0] * (N + 1)
    for _ in range(Q):
        typ = next(it)
        if typ == 'I':
            v = int(next(it))
            j = bisect.bisect_right(keys, v)   # interval before key j
            demand[j] += 1
        else:
            next(it); next(it)

    # reservoir before key j = demand[j]; interval N (past last key) uses the
    # free tail automatically. A small cushion keeps drift-scattered inserts free.
    pos = []
    cur = 0
    total_res = sum(demand[:N])
    # cushion budget: whatever capacity is left after keys + exact reservoirs,
    # but keep it modest so we don't dilute; add 1 extra slot to hot intervals.
    slack = C - N - total_res
    cushion = 1 if slack > N else 0
    for j in range(N):
        res = demand[j]
        if res > 0:
            res += cushion
        cur += res
        if cur >= C:
            cur = C - (N - j)   # clamp so remaining keys still fit
        pos.append(cur)
        cur += 1
    # safety: strictly increasing & in range
    prev = -1
    for i in range(N):
        if pos[i] <= prev:
            pos[i] = prev + 1
        if pos[i] >= C:
            pos[i] = C - (N - i)
        prev = pos[i]

    out = ["%d 0" % N, " ".join(map(str, pos))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
