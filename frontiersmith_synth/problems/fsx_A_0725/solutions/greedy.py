# TIER: greedy
# The obvious textbook recipe: rank every node ONCE by its (static) degree in the whole
# contact network, then spend the budget on the highest-degree nodes first, filling each
# round up to rate_cap until total_budget runs out -- a single static plan computed from
# the graph alone, with no regard for which nodes are already infected/recovered or where
# the outbreak's frontier actually is by the time each round arrives. This wins on "protect
# the biggest hubs" instincts but is a trap here: the single highest-degree node sits in a
# decoy cluster the outbreak may never even reach, and the first cluster's own hub -- also
# high degree -- is the outbreak's origin, already infected before round 0 begins.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
T = inst["T"]
rate_cap = inst["rate_cap"]
total_budget = inst["total_budget"]

deg = [0] * N
for a, b in inst["edges"]:
    deg[a] += 1
    deg[b] += 1

order = sorted(range(N), key=lambda i: (-deg[i], i))

schedule = [[] for _ in range(T)]
idx = 0
spent = 0
for t in range(T):
    cnt = 0
    while cnt < rate_cap and spent < total_budget and idx < len(order):
        schedule[t].append(order[idx])
        idx += 1
        cnt += 1
        spent += 1

print(json.dumps({"schedule": schedule}))
