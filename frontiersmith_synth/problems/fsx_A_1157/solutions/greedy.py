# TIER: greedy
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); KINDS = int(next(it)); budget = int(next(it))
    cost = []; price = []
    for _ in range(T):
        c = [int(next(it)) for _ in range(KINDS)]
        p = int(next(it))
        cost.append(c); price.append(p)
    K = int(next(it))

    # obvious recipe: pool every scenario into one big "average mix" of per-kind work
    # D_k, then run a textbook bottleneck-minimizing waterfill over the per-kind
    # SPECIALISTS ONLY (buy +1 unit, repeatedly, for whichever kind currently has the
    # worst D_k / (units bought) ratio) so the fleet is sized proportionally to average
    # demand. This never considers the "wasteful" generalist -- a specialist is always
    # a strictly better bang-per-buck for any single kind in isolation -- and it treats
    # every scenario as if it were the same averaged mix, blind to the fact that
    # scenarios don't overlap and only the 2 worst of 9 can be sacrificed.
    demand = [0] * KINDS
    for _ in range(K):
        n_jobs = int(next(it)); oracle = int(next(it))
        for _ in range(n_jobs):
            L = int(next(it))
            for _ in range(L):
                k = int(next(it)); w = int(next(it))
                demand[k] += w

    # specialist type for kind k = the type with the lowest cost at kind k
    spec = [min(range(T), key=lambda t: (cost[t][k], price[t], t)) for k in range(KINDS)]
    spec_price = [price[spec[k]] for k in range(KINDS)]

    counts = [0] * T
    spent = 0
    units = [0] * KINDS
    while True:
        # cheapest affordable kind to buy next
        afford = [k for k in range(KINDS) if spent + spec_price[k] <= budget]
        if not afford:
            break
        # worst-served kind: infinite ratio if not yet bought, else demand/units
        def ratio(k):
            return float('inf') if units[k] == 0 and demand[k] > 0 else demand[k] / max(1, units[k])
        kbest = max(afford, key=ratio)
        units[kbest] += 1
        spent += spec_price[kbest]
        counts[spec[kbest]] += 1

    if sum(counts) == 0:
        cheapest = min(range(T), key=lambda t: (price[t], t))
        counts[cheapest] = 1

    print(" ".join(map(str, counts)))


if __name__ == "__main__":
    main()
