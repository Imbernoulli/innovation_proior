# TIER: trivial
# Do nothing clever: split the integer budget evenly across all measurements
# (round-robin remainder, clamp to per-measurement caps). This is exactly the
# evaluator's own weak baseline construction, so it lands at r~0.1 by construction.
import sys, json

inst = json.load(sys.stdin)
M = inst["n_measurements"]
B = inst["budget"]
cap = inst["cap"]

n_each, rem = divmod(B, M)
alloc = [n_each] * M
for i in range(rem):
    alloc[i] += 1
for m in range(M):
    if alloc[m] > cap[m]:
        alloc[m] = cap[m]

print(json.dumps({"alloc": alloc}))
