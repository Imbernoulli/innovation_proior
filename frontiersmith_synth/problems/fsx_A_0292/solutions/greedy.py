# TIER: greedy
# 2-D first-fit in arrival order: place each group on the lowest-index tour that
# still has room on BOTH axes; open a new tour only if none fits.  Reuses gaps in
# earlier tours (better than next-fit) but never reorders the queue, so large or
# demanding late-arriving groups still waste one axis.
import sys, json

inst = json.load(sys.stdin)
C = inst["C"]; T = inst["T"]
people = inst["people"]; minutes = inst["minutes"]

rp = []   # remaining crowd capacity per open tour
rf = []   # remaining docent-minutes per open tour
assign = []
for p, f in zip(people, minutes):
    placed = -1
    for i in range(len(rp)):
        if rp[i] >= p and rf[i] >= f:
            placed = i
            break
    if placed < 0:
        rp.append(C - p); rf.append(T - f)
        assign.append(len(rp) - 1)
    else:
        rp[placed] -= p; rf[placed] -= f
        assign.append(placed)

print(json.dumps({"assign": assign}))
