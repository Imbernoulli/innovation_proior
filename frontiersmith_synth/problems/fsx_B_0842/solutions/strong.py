# TIER: strong
# Lookahead prefetch along the drift, honouring the one-step fetch latency: what
# you set as cache_seq[t] can only ever protect step t+1 onward (see statement),
# so the right question at time t is "which items will be worth having resident
# starting NEXT step". Because the FULL weight table is handed to the candidate
# up front, the deterministic drift is already visible in rows > t. Score every
# item by its near-future avoidable-miss value, weighting the very next step
# heaviest and decaying linearly further out:
#     value(i,t) = sum_{d=1}^{L} (L-d+1) * weight[t+d][i] * miss_cost[i]
# A currently-cached item only needs to clear that bar to stay resident (holding
# is free); an item NOT currently cached must clear value(i,t) MINUS its flat
# fetch_cost[i] to be worth bringing in now -- i.e. fetch it exactly early enough
# to land before its hot stretch starts, amortizing the one-time fetch price
# against the whole run of misses it prevents. Recomputing this every step and
# keeping the top-C combined scores means the cache starts filling with the NEXT
# hot ids before the drift ever causes a miss on them, and an item is only
# evicted when a costlier-to-miss upcoming arrival needs its slot -- turning the
# deterministic drift from a liability (as in the reactive lag-1 policy) into
# something to plan a step ahead of.
import sys, json

inst = json.load(sys.stdin)
M = inst["M"]; T = inst["T"]; C = inst["C"]
W = inst["weight"]
miss_cost = inst["miss_cost"]
fetch_cost = inst["fetch_cost"]

L = 2  # short lookahead beyond t: near-term hotness must dominate far-term hotness

cache = []
cached = set()
for t in range(T):
    lo = t + 1
    hi = min(T, lo + L)
    value = [0.0] * M
    for tp in range(lo, hi):
        w = L - (tp - lo)          # linear decay: t+1 counts most, t+L counts least
        row = W[tp]
        for i in range(M):
            value[i] += w * row[i] * miss_cost[i]
    combined = []
    for i in range(M):
        s = value[i] if i in cached else (value[i] - fetch_cost[i])
        combined.append(s)
    order = sorted(range(M), key=lambda i: (-combined[i], i))
    nxt = order[:C]
    cache.append(nxt)
    cached = set(nxt)

print(json.dumps({"cache": cache}))
