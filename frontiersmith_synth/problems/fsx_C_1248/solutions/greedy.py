# TIER: greedy
# "Peak clock frequency" recipe: split into the finest possible stages (S=N,
# one logic block per stage) to minimize cycle time, then buy forwarding paths
# by best value/cost ratio until the budget runs out. Never reconsiders S.
import sys


def main():
    data = sys.stdin.read().split()
    idx = [0]

    def nxt():
        v = data[idx[0]]
        idx[0] += 1
        return v

    N = int(nxt()); K = int(nxt()); L = int(nxt())
    Br = int(nxt()); Mb = int(nxt()); resolve_block = int(nxt())
    Budget = int(nxt()); I = int(nxt())
    c = [int(nxt()) for _ in range(N)]
    haz = []
    for _ in range(K):
        need_b = int(nxt()); res_b = int(nxt()); dist = int(nxt()); freq = int(nxt())
        haz.append((need_b, res_b, dist, freq))

    S = N
    cuts = list(range(1, N))  # one block per stage -> stage_of(i) == i

    items = []
    for k in range(1, K + 1):
        need_b, res_b, dist, freq = haz[k - 1]
        gap = res_b - need_b
        stall = max(0, gap - dist)
        value = freq * stall
        cost = gap
        if cost > 0 and value > 0:
            items.append((value / cost, cost, k))
    items.sort(key=lambda t: -t[0])

    chosen = []
    remaining = Budget
    for _, cost, k in items:
        if cost <= remaining:
            chosen.append(k)
            remaining -= cost

    print(S)
    print(" ".join(map(str, cuts)))
    print(len(chosen))
    print(" ".join(map(str, chosen)))


if __name__ == "__main__":
    main()
