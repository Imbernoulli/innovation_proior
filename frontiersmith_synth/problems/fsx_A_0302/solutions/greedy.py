# TIER: greedy
# First-Fit (online, arrival order): for each request scan all open blocks and drop it
# into the first one with enough remaining room; open a new block only if none fits.
import sys, json

inst = json.load(sys.stdin)
cap = inst["capacity"]
items = inst["items"]

loads = []          # remaining capacity per open block
assign = []
for x in items:
    placed = -1
    for j in range(len(loads)):
        if loads[j] >= x:
            placed = j
            break
    if placed < 0:
        loads.append(cap - x)
        assign.append(len(loads) - 1)
    else:
        loads[placed] -= x
        assign.append(placed)

print(json.dumps({"assign": assign}))
