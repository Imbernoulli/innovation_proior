# TIER: trivial
# The TRUE-OPTIMAL fixed cache: pick a single set S, held for every time step,
# that minimizes total cost given the one-step fetch latency (step 0 is always
# served cold, whatever S is; S is resident and paying only hit/miss costs from
# step 1 onward, after a one-time fetch at step 0). Because holding item i for
# steps 1..T-1 nets (miss_cost[i]-hit_cost)*sum(weight[1:][i]) - fetch_cost[i],
# independently of every other item, the optimal S is exactly the C items with
# the highest positive net gain. This never reacts to the hot block moving at
# all -- it is the best possible "ignore the drift" design -- so it still eats
# heavy miss costs whenever the drift has moved the hot block away from S. This
# is exactly the evaluator's own baseline() construction, so it scores ~0.1 by
# definition; no non-adaptive fixed-cache submission can beat it.
import sys, json

inst = json.load(sys.stdin)
M = inst["M"]; T = inst["T"]; C = inst["C"]
W = inst["weight"]
hit_cost = inst["hit_cost"]; miss_cost = inst["miss_cost"]; fetch_cost = inst["fetch_cost"]

tot_w_from1 = [0.0] * M
for t in range(1, T):
    row = W[t]
    for i in range(M):
        tot_w_from1[i] += row[i]

gain = [(miss_cost[i] - hit_cost) * tot_w_from1[i] - fetch_cost[i] for i in range(M)]
order = sorted(range(M), key=lambda i: (-gain[i], i))
fixed = sorted(i for i in order[:C] if gain[i] > 0.0)
cache = [list(fixed) for _ in range(T)]
print(json.dumps({"cache": cache}))
