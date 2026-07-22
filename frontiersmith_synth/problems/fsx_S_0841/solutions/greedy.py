# TIER: greedy
# The obvious recipe: a single power-of-two-choices rebalancing PASS over keys in the
# order the instance lists them (creation/id order), spending the migration budget on
# the first keys it meets that look locally beneficial to move. For each key it compares
# its current shard's running load against its two precomputed alternative shards and
# moves it to the less-loaded alternative if that current shard looks more loaded right
# now -- textbook power-of-two-choices load balancing. It never looks at WEIGHT to decide
# which keys are worth spending the scarce budget on, so on a skewed instance it burns
# the whole budget shuffling ordinary low-weight keys it happens to reach first, and
# never gets to the handful of viral high-weight keys sitting at the end of the id list.
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
for i in range(N):
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
