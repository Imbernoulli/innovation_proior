# TIER: strong
# Insight: greedy's mixture-pooled split treats "half the ITEMS" as the goal,
# so its cut point can slice straight through one pool's own clumped
# cluster -- fine for the mixture average, ruinous for that one pool's own
# mean (it pays for an uninformative probe that neither confirms nor rules
# out its cluster). The fix is a minimax INFORMATION-DESIGN move: never let
# an early probe straddle a scenario's clumping. First, infer each pool's
# home cluster straight from the given weights (the item(s) where that pool
# is the clear argmax -- clumping leaves an unmistakable fingerprint even
# though we were never told the cluster lists directly). Then build the tree
# top-down over WHOLE CLUSTERS ONLY: at each step, search all ways to bucket
# the still-undetermined clusters into a yes/no probe (the cluster count is
# tiny, so this is exhaustive) and keep whichever bucketing sends the
# closest-to-half of the remaining POOLED mass to each side -- but, crucially,
# always as whole clusters, so no single scenario's clumping is ever cut in
# two. That keeps every pool's own residual entropy fully intact (an
# untouched cluster) or fully resolved (an isolated one) at every step,
# instead of being partially, wastefully split by a query aimed at nobody's
# scenario in particular. Only once exactly one cluster remains do we fall
# back to an ordinary weighted binary search within it.
import sys, json

inst = json.load(sys.stdin)
n = inst["n_items"]
pools = [p["weights"] for p in sorted(inst["pools"], key=lambda p: p["pool"])]
k = len(pools)

# --- infer each item's home pool (cluster) from the weights themselves ---
home = [max(range(k), key=lambda p: pools[p][i]) for i in range(n)]
clusters = [[i for i in range(n) if home[i] == c] for c in range(k)]
clusters = [c for c in clusters if c]  # drop any empty (defensive)

mix = [sum(p[i] for p in pools) / k for i in range(n)]
cluster_mass = [sum(mix[i] for i in c) for c in clusters]


def within_cluster_tree(items, home_pool):
    items = sorted(items)
    w = pools[home_pool]

    def rec(lst):
        if len(lst) == 1:
            return {"guess": lst[0]}
        total = sum(w[i] for i in lst)
        best_mid, best_diff = 1, None
        for mid in range(1, len(lst)):
            cum = sum(w[i] for i in lst[:mid])
            diff = abs(cum - total / 2)
            if best_diff is None or diff < best_diff:
                best_diff, best_mid = diff, mid
        left, right = lst[:best_mid], lst[best_mid:]
        return {"query": sorted(left), "yes": rec(left), "no": rec(right)}

    return rec(items)


def phase_a(remaining):
    if len(remaining) == 1:
        c = remaining[0]
        # the cluster's own dominant pool (all its members share that home)
        hp = home[clusters[c][0]]
        return within_cluster_tree(clusters[c], hp)
    total = sum(cluster_mass[c] for c in remaining)
    ncls = len(remaining)
    best_mask, best_diff = None, None
    for mask in range(1, (1 << ncls) - 1):
        s = sum(cluster_mass[remaining[i]] for i in range(ncls) if mask & (1 << i))
        diff = abs(s - total / 2)
        if best_diff is None or diff < best_diff:
            best_diff, best_mask = diff, mask
    yes = [remaining[i] for i in range(ncls) if best_mask & (1 << i)]
    no = [remaining[i] for i in range(ncls) if not (best_mask & (1 << i))]
    yes_items = sorted(i for c in yes for i in clusters[c])
    return {"query": yes_items, "yes": phase_a(yes), "no": phase_a(no)}


if len(clusters) == 1:
    tree = within_cluster_tree(clusters[0], home[clusters[0][0]])
else:
    tree = phase_a(list(range(len(clusters))))

print(json.dumps({"tree": tree}))
