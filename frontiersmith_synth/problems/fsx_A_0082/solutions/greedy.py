# TIER: greedy
# First-fit dispatch in arrival order: board each party onto the lowest-index
# gondola that still has room; open a new gondola only if none fits.  Better than
# next-fit because it reuses gaps in earlier gondolas, but it never reorders the
# queue, so large late-arriving parties still waste seats.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
parties = inst["parties"]

rem = []            # remaining seats per open gondola
assign = []
for s in parties:
    placed = -1
    for i in range(len(rem)):
        if rem[i] >= s:
            placed = i
            break
    if placed < 0:
        rem.append(C - s)
        assign.append(len(rem) - 1)
    else:
        rem[placed] -= s
        assign.append(placed)

print(json.dumps({"assign": assign}))
