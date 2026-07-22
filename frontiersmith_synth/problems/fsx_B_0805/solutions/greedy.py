# TIER: greedy
# The "obvious" recipe: price EACH product independently by its own pooled
# revenue curve (classic monopoly pricing over the willingness-to-pay values of
# every customer type, weighted by population), assuming a customer buys product
# j whenever v_j >= price. This ignores that customers actually make ONE
# discrete choice across the WHOLE menu (they might prefer a different product,
# or walk away to their outside option) -- i.e. it treats products as
# independent revenue curves rather than a menu that must retain every cohort.
import sys


def main():
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

    print(*prices)


if __name__ == "__main__":
    main()
