# TIER: strong
# The insight: reformulate the move space. Instead of only scoring single
# moves, score every 2-move "expand THEN collapse" combo as one atomic
# opportunity (its NET benefit, even though its first half raises the cost),
# alongside plain 1-move collapses. Rank all opportunities by benefit PER
# MOVE and take them greedily -- this is the only way to reach the canyon
# collapses, which a step-by-step "never get worse right now" recipe cannot
# find because their first half is uphill.
import sys


def main():
    toks = sys.stdin.read().split()
    idx = [0]

    def nxt():
        v = toks[idx[0]]
        idx[0] += 1
        return v

    n = int(nxt()); k = int(nxt()); r = int(nxt())
    s = [int(nxt()) for _ in range(n)]
    cost = [0] * (k + 1)
    for v in range(1, k + 1):
        cost[v] = int(nxt())
    me = int(nxt())
    expand = {}
    for _ in range(me):
        v = int(nxt()); x = int(nxt()); y = int(nxt())
        expand[v] = (x, y)
    mc = int(nxt())
    collapse = {}
    for _ in range(mc):
        x = int(nxt()); y = int(nxt()); z = int(nxt())
        collapse[(x, y)] = z

    cur = list(s)
    moves = []
    budget = r

    def find_best_opportunity(budget):
        best = None  # (efficiency, benefit, moves_needed, kind, i)
        L = len(cur)
        for i in range(L - 1):
            x0, y0 = cur[i], cur[i + 1]
            if (x0, y0) in collapse:
                z = collapse[(x0, y0)]
                benefit = cost[x0] + cost[y0] - cost[z]
                if benefit > 0 and (best is None or benefit > best[0]):
                    best = (benefit, benefit, 1, "C", i)
        if budget >= 2:
            for i in range(L):
                v = cur[i]
                if v not in expand:
                    continue
                x, y = expand[v]
                if i + 1 < L:
                    w = cur[i + 1]
                    if (y, w) in collapse:
                        z = collapse[(y, w)]
                        benefit = (cost[v] + cost[w]) - (cost[x] + cost[z])
                        eff = benefit / 2.0
                        if benefit > 0 and (best is None or eff > best[0]):
                            best = (eff, benefit, 2, "ER", i)
                if i - 1 >= 0:
                    u = cur[i - 1]
                    if (u, x) in collapse:
                        z = collapse[(u, x)]
                        benefit = (cost[u] + cost[v]) - (cost[z] + cost[y])
                        eff = benefit / 2.0
                        if benefit > 0 and (best is None or eff > best[0]):
                            best = (eff, benefit, 2, "EL", i)
        return best

    while budget > 0:
        opp = find_best_opportunity(budget)
        if opp is None:
            break
        eff, benefit, need, kind, i = opp
        if need > budget:
            break
        if kind == "C":
            x, y = cur[i], cur[i + 1]
            z = collapse[(x, y)]
            cur[i:i + 2] = [z]
            moves.append(("C", i + 1))
        elif kind == "ER":
            v = cur[i]
            x, y = expand[v]
            cur[i:i + 1] = [x, y]
            moves.append(("E", i + 1))
            xx, yy = cur[i + 1], cur[i + 2]
            z = collapse[(xx, yy)]
            cur[i + 1:i + 3] = [z]
            moves.append(("C", i + 2))
        elif kind == "EL":
            v = cur[i]
            x, y = expand[v]
            cur[i:i + 1] = [x, y]
            moves.append(("E", i + 1))
            xx, yy = cur[i - 1], cur[i]
            z = collapse[(xx, yy)]
            cur[i - 1:i + 1] = [z]
            moves.append(("C", i))
        budget -= need

    print(len(moves))
    out = [f"{op} {pos}" for op, pos in moves]
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
