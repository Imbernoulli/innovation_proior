# TIER: greedy
# The obvious recipe: naive per-PRODUCT LP inversion. For each of the 3 products
# independently, pick whichever single raw type gives the best yield-per-cost for
# THAT product alone, then buy each chosen source in proportion to the product's
# target share out of a modest fixed spending budget, rounding the continuous
# quantity to the nearest whole number of lots.
#
# This ignores three things at once: (1) most raw types are MIXED recipes, so
# summing several independently-chosen per-product allocations double-counts side
# yields and drags the realized ratio away from target; (2) it never checks whether
# the target ratio is even reachable by any non-negative combination of raw types
# (some are not); (3) it never checks a source's full-yield batch threshold, so it
# can plan for full-rate output from a source that is actually running degraded.
import sys, json

inst = json.load(sys.stdin)
P = inst["P"]
K = inst["K"]
cost = inst["cost"]
yld = inst["yield"]
lot = inst["lot"]
maxOrder = inst["maxOrder"]
target = inst["target"]
spend_cap = sum(maxOrder[j] * cost[j] for j in range(P))

SPEND_FRAC = 0.06
spend_budget = spend_cap * SPEND_FRAC

best_src = []
for k in range(K):
    j_star = max(range(P), key=lambda j: (yld[j][k] / cost[j], -j))
    best_src.append(j_star)

accum_units = [0.0] * P
for k in range(K):
    j = best_src[k]
    budget_k = spend_budget * target[k]
    accum_units[j] += budget_k / cost[j]

order = [0] * P
for j in range(P):
    nb = int(round(accum_units[j] / lot[j]))
    oj = nb * lot[j]
    oj = max(0, min(oj, maxOrder[j]))
    order[j] = oj

print(json.dumps({"order": order}))
