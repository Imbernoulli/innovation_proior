# TIER: greedy
# Demand-proportional local sizing: give each district a share of the budget
# proportional to its MEAN demand (depot gets nothing). Matches supply to
# average need far better than a uniform split, but sizes purely to the mean
# (no safety buffer for variability, no central pooling reserve).
import sys, json, math
inst = json.load(sys.stdin)
N = inst["N"]; B = inst["B"]; means = inst["means"]
tot = sum(means[1:]) or 1.0
x = [0] * (N + 1)
for i in range(1, N + 1):
    x[i] = int(math.floor(B * means[i] / tot))
used = sum(x)
# hand out any leftover units to the highest-mean districts first
order = sorted(range(1, N + 1), key=lambda i: -means[i])
k = 0
while used < B and order:
    x[order[k % len(order)]] += 1
    used += 1; k += 1
print(json.dumps({"stock": x}))
