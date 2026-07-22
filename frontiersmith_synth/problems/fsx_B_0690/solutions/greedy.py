# TIER: greedy
# The obvious first idea: this "looks like" a load-balanced bin-packing /
# knapsack-partition problem, so sort the projects by descending welfare
# weight and drop each one, in turn, into whichever bundle currently holds
# the smallest total weight (classic LPT load balancing).  This never reads
# a single supporter/opposer index -- it treats "popularity" as irrelevant
# and "value" as the only signal, i.e. it clusters by weight, not by which
# UNIONS of voter sets actually clear a majority.  It reliably fails to
# rediscover the planted polarizer/sweetener pairings (a heavy polarizer and
# its cheap sweetener are far apart in weight-sorted order, and LPT actively
# avoids piling more onto an already-heavy bundle) and, worse, it will often
# drop a "poison" project into a bundle that otherwise would have passed,
# sinking it entirely.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    P = int(next(it)); V = int(next(it)); K = int(next(it))
    weights = []
    for i in range(P):
        w = int(next(it)); s = int(next(it)); o = int(next(it))
        ns = int(next(it)); no = int(next(it))
        for _ in range(ns):
            next(it)
        for _ in range(no):
            next(it)
        weights.append(w)

    order = sorted(range(P), key=lambda i: -weights[i])
    bundle_total = [0] * (K + 1)  # 1-indexed
    assign = [0] * P
    for i in order:
        # smallest-current-total bundle (ties -> lowest id)
        best_k = min(range(1, K + 1), key=lambda k: (bundle_total[k], k))
        assign[i] = best_k
        bundle_total[best_k] += weights[i]

    print(P)
    print(" ".join(map(str, assign)))


if __name__ == "__main__":
    main()
