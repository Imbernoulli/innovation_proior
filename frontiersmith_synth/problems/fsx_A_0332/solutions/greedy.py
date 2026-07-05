# TIER: greedy
# First-fit porter loading in arrival order: board each item onto the lowest-index
# EXISTING load that still has weight room AND either already carries the item's
# category or has fewer than K categories; open a new load only if none qualifies.
# Better than next-fit because it reuses room in earlier loads, but it never reorders
# the queue and is blind to which categories should share a load, so it still wastes
# class-slots and weight on late, awkward items.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
K = inst["classes"]
weights = inst["weights"]
category = inst["category"]

rem = []            # remaining weight per open load
cats = []           # set of categories per open load
assign = []
for w, c in zip(weights, category):
    placed = -1
    for i in range(len(rem)):
        if rem[i] >= w and (c in cats[i] or len(cats[i]) < K):
            placed = i
            break
    if placed < 0:
        rem.append(C - w)
        cats.append({c})
        assign.append(len(rem) - 1)
    else:
        rem[placed] -= w
        cats[placed].add(c)
        assign.append(placed)

print(json.dumps({"assign": assign}))
