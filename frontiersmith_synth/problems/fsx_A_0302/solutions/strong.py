# TIER: strong
# Best-Fit-Decreasing: sort requests by descending duration, then place each into the
# open block whose remaining room is smallest but still >= the duration (tightest fit).
# Big requests go down first, and small ones top off the tightest surviving gaps ->
# near-optimal packing, well above first-fit. Assignment mapped back to arrival indices.
import sys, json

inst = json.load(sys.stdin)
cap = inst["capacity"]
items = inst["items"]
n = len(items)

order = sorted(range(n), key=lambda i: -items[i])
rem = []            # remaining capacity per open block
assign = [0] * n
for i in order:
    x = items[i]
    best = -1
    best_rem = cap + 1
    for j in range(len(rem)):
        if rem[j] >= x and rem[j] < best_rem:
            best_rem = rem[j]
            best = j
    if best < 0:
        rem.append(cap - x)
        assign[i] = len(rem) - 1
    else:
        rem[best] -= x
        assign[i] = best

print(json.dumps({"assign": assign}))
