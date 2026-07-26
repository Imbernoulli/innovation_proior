# TIER: strong
# Insight: the objective is the p90 (worst-tail) scenario cost, not the average
# scenario cost, so the layout only needs to serve the co-purchase structure of
# the FEW scenarios that actually define that tail -- global SKU velocity is
# the wrong signal to slot by.
#
# 1) Bootstrap a rough velocity-based layout (classic ABC slotting) purely to
#    ESTIMATE which scenarios are the costly ones under a naive layout.
# 2) Recompute every scenario's total picker distance under that rough layout
#    and take the top slice (by cost) as the "tail set" -- a generous superset
#    of the true p90-defining scenarios.
# 3) Build the co-purchase AFFINITY GRAPH pooled ONLY over orders in the tail
#    set: every pair of SKUs that appears together in a tail-set order gets an
#    edge, weight += 1. A SKU's weighted degree in this graph (its total edge
#    weight) measures how central it is to the "communities" that actually
#    drive the worst-case scenarios -- exactly the tail-pooled affinity signal
#    the innovation hook calls for, as opposed to context-free global velocity.
# 4) Re-slot using a composite priority = global frequency + a BOOST on that
#    weighted degree. SKUs that are individually rare (low global velocity)
#    but sit at the center of a tight tail-scenario co-purchase community get
#    pulled forward toward the depot, displacing only the SKUs that were
#    barely more popular than them to begin with -- not the genuinely
#    high-velocity assortment that every scenario relies on.
import sys


def read_instance(d):
    p = 0
    def nxt():
        nonlocal p
        v = d[p]; p += 1
        return v
    N = int(nxt()); A = int(nxt()); L = int(nxt()); K = int(nxt()); W = int(nxt())
    scenarios = []
    for _ in range(K):
        q = int(nxt())
        orders = []
        for _ in range(q):
            s = int(nxt())
            orders.append([int(nxt()) for _ in range(s)])
        scenarios.append(orders)
    return N, A, L, K, W, scenarios


def scenario_cost(perm, L, W, orders):
    tot = 0
    for skus in orders:
        aisles = set()
        max_depth = {}
        for sku in skus:
            slot = perm[sku]
            aisle = slot // L + 1
            depth = slot % L + 1
            aisles.add(aisle)
            if depth > max_depth.get(aisle, 0):
                max_depth[aisle] = depth
        if not aisles:
            continue
        tot += 2 * W * max(aisles) + 2 * sum(max_depth.values())
    return tot


TAIL_FRAC = 0.15
BOOST = 6


def main():
    d = sys.stdin.read().split()
    N, A, L, K, W, scenarios = read_instance(d)

    freq = [0] * N
    for orders in scenarios:
        for skus in orders:
            for sku in skus:
                freq[sku] += 1

    # --- 1) rough bootstrap layout, just to rank scenarios ---
    rough_order = sorted(range(N), key=lambda sku: (-freq[sku], sku))
    rough_perm = [0] * N
    for slot, sku in enumerate(rough_order):
        rough_perm[sku] = slot

    costs = [(s, scenario_cost(rough_perm, L, W, scenarios[s])) for s in range(K)]
    costs.sort(key=lambda t: -t[1])

    # --- 2) tail set: top slice of costliest scenarios (generous superset) ---
    import math
    tail_n = max(1, math.ceil(TAIL_FRAC * K))
    tail_idx = set(s for s, _ in costs[:tail_n])

    # --- 3) co-purchase affinity graph, pooled ONLY over the tail scenarios:
    #         edge weight += 1 for every SKU pair sharing a tail-set order.
    #         A SKU's weighted degree = total edge weight touching it. ---
    weighted_degree = [0] * N
    for s in tail_idx:
        for skus in scenarios[s]:
            uniq = list(dict.fromkeys(skus))
            m = len(uniq)
            if m < 2:
                continue
            for sku in uniq:
                weighted_degree[sku] += (m - 1)

    # --- 4) composite priority slotting ---
    score = [freq[i] + BOOST * weighted_degree[i] for i in range(N)]
    order = sorted(range(N), key=lambda sku: (-score[sku], -freq[sku], sku))
    perm = [0] * N
    for slot, sku in enumerate(order):
        perm[sku] = slot

    print(N)
    print("\n".join(str(x) for x in perm))


main()
