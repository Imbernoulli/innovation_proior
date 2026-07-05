# TIER: greedy
# First-fit slotting in booking order: place each act on the lowest-index stage
# that still has BOTH resource room (footprint fits under C) and a free changeover
# window (fewer than K acts).  Open a new stage only if none qualifies.  Better
# than next-fit because it reuses gaps on earlier stages, but it never reorders the
# lineup, so big late-booked rigs still strand capacity.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
K = inst["max_acts"]
acts = inst["acts"]

rem = []            # remaining resource capacity per open stage
cnt = []            # acts already slotted per open stage
assign = []
for s in acts:
    placed = -1
    for i in range(len(rem)):
        if rem[i] >= s and cnt[i] < K:
            placed = i
            break
    if placed < 0:
        rem.append(C - s)
        cnt.append(1)
        assign.append(len(rem) - 1)
    else:
        rem[placed] -= s
        cnt[placed] += 1
        assign.append(placed)

print(json.dumps({"assign": assign}))
