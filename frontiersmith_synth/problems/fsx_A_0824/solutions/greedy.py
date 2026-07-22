# TIER: greedy
# The "obvious" recipe: pool the K priors into ONE combined (mixture)
# distribution over items -- average the K weight vectors -- and build a
# standard weighted binary-search tree over items IN THEIR GIVEN ID ORDER,
# always cutting the current CONTIGUOUS id range at whichever point puts the
# closest-to-even mixture mass on each side. This is a textbook, perfectly
# reasonable "optimal binary search tree for the pooled prior" -- and it is
# optimal for minimizing the AVERAGE probe count under the mixture. It never
# looks at any INDIVIDUAL pool's own distribution, so whenever the mixture's
# best cut point happens to slice straight through one pool's own clumped
# cluster, that pool alone pays the price (the other pools' means are what
# made the cut look "balanced" on average).
import sys, json

inst = json.load(sys.stdin)
n = inst["n_items"]
pools = [p["weights"] for p in sorted(inst["pools"], key=lambda p: p["pool"])]
k = len(pools)
mix = [sum(p[i] for p in pools) / k for i in range(n)]


def rec(lo, hi):
    if hi - lo == 1:
        return {"guess": lo}
    total = sum(mix[lo:hi])
    best_mid, best_diff = lo + 1, None
    cum = 0.0
    for mid in range(lo + 1, hi):
        cum = sum(mix[lo:mid])
        diff = abs(cum - total / 2)
        if best_diff is None or diff < best_diff:
            best_diff, best_mid = diff, mid
    query = list(range(lo, best_mid))
    return {"query": query, "yes": rec(lo, best_mid), "no": rec(best_mid, hi)}


tree = rec(0, n)
print(json.dumps({"tree": tree}))
