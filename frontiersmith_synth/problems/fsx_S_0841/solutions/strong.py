# TIER: strong
# INSIGHT (the genuine leverage, not "power-of-two on whatever key comes first"):
#  (1) DIAGNOSE THE SKEW FIRST. The migration budget is a cap on the COUNT of moved
#      keys, not on weight, and it is far smaller than N. Since the max shard load is
#      driven almost entirely by whichever handful of keys carry the most weight, the
#      budget should be spent on those keys FIRST -- process keys in weight-descending
#      order, not id order. This alone stops the recipe's failure mode of exhausting the
#      budget on boring keys before ever reaching the viral tail.
#  (2) SPEND THE SAVED BUDGET WITH POWER-OF-TWO CHOICES. For each key (processed
#      heaviest-first), compare its current shard's RUNNING load (state carried across
#      the whole pass, so later heavy keys see the effect of earlier moves and don't all
#      pile onto the same relief shard) against its two precomputed alternative shards,
#      and move it to the lighter alternative only if that's a genuine improvement.
#  (3) LEAVE THE LIGHT MAJORITY ALONE. Once the heavy keys are resolved, any remaining
#      budget mops up the next-heaviest opportunities, but a moved LIGHT key is worth
#      little, so budget spent there is naturally deprioritised versus the heavy tail.
import sys, json

inst = json.load(sys.stdin)
N, S = inst["N"], inst["S"]
shard0 = inst["shard0"]
weight = inst["weight"]
alt = inst["alt"]
budget = inst["budget"]

load = [0.0] * S
for i in range(N):
    load[shard0[i]] += weight[i]

assign = list(shard0)
used = 0
order = sorted(range(N), key=lambda i: -weight[i])
for i in order:
    if used >= budget:
        break
    cur = shard0[i]
    a, b = alt[i]
    target = a if load[a] <= load[b] else b
    if load[cur] > load[target]:
        w = weight[i]
        load[cur] -= w
        load[target] += w
        assign[i] = target
        used += 1

print(json.dumps({"assign": assign}))
