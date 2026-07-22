# TIER: strong
# The reformulation: don't search for a revenue curve, search directly for the
# price vector that maximizes the MIN cohort average profit (a leximin /
# "self-selection ladder" objective). Repeatedly find the currently worst-off
# cohort, find the product its own types are choosing, and raise exactly that
# product's price up to (but not past) the point where that cohort would
# defect -- a water-filling pass on the participation constraints -- then
# polish with ordinary coordinate ascent directly on the min-objective, seeded
# both from a do-nothing baseline AND from the naive revenue-curve menu (so the
# search can never do worse than the naive recipe it improves on).
import sys


def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)
    M = int(next(it))
    K = int(next(it))
    costs = []
    markups = []
    for _ in range(M):
        costs.append(int(next(it)))
        markups.append(int(next(it)))
    cohorts = []
    for _ in range(K):
        T = int(next(it))
        types = []
        for _ in range(T):
            n = int(next(it))
            o = int(next(it))
            v = [int(next(it)) for _ in range(M)]
            types.append((n, o, v))
        cohorts.append(types)
    return M, K, costs, markups, cohorts


def choice(prices, v, M):
    best_j = -1
    best_u = None
    for j in range(M):
        u = v[j] - prices[j]
        if best_u is None or u > best_u:
            best_u = u
            best_j = j
    return best_j, best_u


def eval_min(M, K, costs, markups, cohorts, prices):
    avgs = []
    for k in range(K):
        total_profit = 0
        total_pop = 0
        for (n, o, v) in cohorts[k]:
            j, u = choice(prices, v, M)
            profit = (prices[j] - costs[j]) if (u is not None and u > o) else 0
            total_profit += n * profit
            total_pop += n
        avgs.append(total_profit / total_pop)
    return min(avgs), avgs


def baseline_prices(M, costs, markups):
    return [costs[j] + max(1, markups[j] // 5) for j in range(M)]


def revenue_curve_greedy(M, K, costs, markups, cohorts):
    prices = []
    for j in range(M):
        pts = []
        for k in range(K):
            for (n, o, v) in cohorts[k]:
                pts.append((v[j], n))
        lo, hi = costs[j], costs[j] + markups[j]
        cand = sorted(set(val for val, n in pts if lo <= val <= hi))
        if not cand:
            cand = [lo]
        best_p = lo
        best_rev = -1
        for p in cand:
            rev = sum(n * (p - costs[j]) for val, n in pts if val >= p)
            if rev > best_rev:
                best_rev = rev
                best_p = p
        prices.append(best_p)
    return prices


def leximin_polish(M, K, costs, markups, cohorts, prices, rounds):
    best_prices = list(prices)
    best_val, _ = eval_min(M, K, costs, markups, cohorts, best_prices)
    cur = list(prices)
    for _ in range(rounds):
        val, avgs = eval_min(M, K, costs, markups, cohorts, cur)
        w = min(range(K), key=lambda k: avgs[k])
        counts = {}
        for (n, o, v) in cohorts[w]:
            j, u = choice(cur, v, M)
            counts[j] = counts.get(j, 0) + n
        if not counts:
            break
        jt = max(counts, key=lambda kk: counts[kk])
        best_p = cur[jt]
        bv = None
        for p in range(costs[jt], costs[jt] + markups[jt] + 1):
            cur[jt] = p
            v2, _ = eval_min(M, K, costs, markups, cohorts, cur)
            if bv is None or v2 >= bv:
                bv = v2
                best_p = p
        cur[jt] = best_p
        v2, _ = eval_min(M, K, costs, markups, cohorts, cur)
        if v2 > best_val:
            best_val = v2
            best_prices = list(cur)
    return best_prices


def coord_polish(M, K, costs, markups, cohorts, prices, sweeps):
    cur = list(prices)
    for _ in range(sweeps):
        for j in range(M):
            best_p = cur[j]
            best_val = None
            for p in range(costs[j], costs[j] + markups[j] + 1):
                cur[j] = p
                val, _ = eval_min(M, K, costs, markups, cohorts, cur)
                if best_val is None or val >= best_val:
                    best_val = val
                    best_p = p
            cur[j] = best_p
    return cur


def main():
    M, K, costs, markups, cohorts = read_instance()
    start = baseline_prices(M, costs, markups)
    gp = revenue_curve_greedy(M, K, costs, markups, cohorts)

    candidates = [start, gp]
    for base in (start, gp):
        p1 = leximin_polish(M, K, costs, markups, cohorts, base, rounds=4 * M)
        p2 = coord_polish(M, K, costs, markups, cohorts, p1, sweeps=2)
        p3 = leximin_polish(M, K, costs, markups, cohorts, p2, rounds=2 * M)
        candidates += [p1, p2, p3]

    best = None
    best_val = -1.0
    for c in candidates:
        v, _ = eval_min(M, K, costs, markups, cohorts, c)
        if v > best_val:
            best_val = v
            best = c

    print(*best)


if __name__ == "__main__":
    main()
