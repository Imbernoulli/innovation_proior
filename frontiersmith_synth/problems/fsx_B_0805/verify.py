import sys


def read_instance(path):
    toks = open(path).read().split()
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


def eval_prices(M, K, costs, markups, cohorts, prices):
    """Deterministic discrete choice: each customer type buys the utility-maximizing
    product (value - price), tie-break lowest index; if the best surplus does not
    exceed their outside option, they buy nothing (0 profit). Returns (min-cohort
    avg profit, per-cohort avg list)."""
    cohort_avgs = []
    for k in range(K):
        total_profit = 0
        total_pop = 0
        for (n, o, v) in cohorts[k]:
            best_j = -1
            best_u = None
            for j in range(M):
                u = v[j] - prices[j]
                if best_u is None or u > best_u:
                    best_u = u
                    best_j = j
            if best_u is not None and best_u > o:
                profit = prices[best_j] - costs[best_j]
            else:
                profit = 0
            total_profit += n * profit
            total_pop += n
        cohort_avgs.append(total_profit / total_pop)
    return min(cohort_avgs), cohort_avgs


def baseline_prices(M, costs, markups):
    # a conservative, uniformly modest markup over cost -- the checker's own
    # trivial feasible construction, used as the normalizer B.
    return [costs[j] + max(1, markups[j] // 5) for j in range(M)]


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    M, K, costs, markups, cohorts = read_instance(inf)

    raw = open(outf).read().split()
    if len(raw) != M:
        print("Ratio: 0.0 (expected %d price tokens, got %d)" % (M, len(raw)))
        return
    prices = []
    for tok in raw:
        try:
            p = int(tok)
        except ValueError:
            print("Ratio: 0.0 (non-integer / non-finite price token %r)" % tok)
            return
        prices.append(p)

    for j in range(M):
        lo, hi = costs[j], costs[j] + markups[j]
        if prices[j] < lo or prices[j] > hi:
            print("Ratio: 0.0 (price[%d]=%d out of bounds [%d,%d])" % (j, prices[j], lo, hi))
            return

    F, cohort_avgs = eval_prices(M, K, costs, markups, cohorts, prices)

    bp = baseline_prices(M, costs, markups)
    B, _ = eval_prices(M, K, costs, markups, cohorts, bp)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f cohort_avgs=%s Ratio: %.6f" % (F, B, [round(x, 3) for x in cohort_avgs], sc / 1000.0))


if __name__ == "__main__":
    main()
