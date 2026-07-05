# TIER: greedy
# First-Fit (in arrival order): for each fragment scan the already-open haulers in
# order and drop it into the FIRST one that still has room; if none does, open a new
# hauler.  Unlike Next-Fit this back-fills earlier haulers, so it clears the streaming
# baseline comfortably.  But it never reorders arrivals and never reconsiders a
# placement, so on awkward mass mixes it still opens more haulers than a
# decreasing-order packer with clean-up (the strong tier).
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
masses = inst["masses"]

loads = []            # remaining-capacity bookkeeping: loads[b] = current mass in b
assign = []
for m in masses:
    placed = -1
    for b in range(len(loads)):
        if loads[b] + m <= C:
            placed = b
            break
    if placed < 0:
        placed = len(loads)
        loads.append(0)
    loads[placed] += m
    assign.append(placed)

print(json.dumps({"assign": assign}))
