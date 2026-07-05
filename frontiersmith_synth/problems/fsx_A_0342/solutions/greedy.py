# TIER: greedy
# First-fit dispatch in arrival order: energize each block onto the lowest-index
# transformer that still has both thermal room and a free breaker channel; open a
# new transformer only if none qualifies.  Better than next-fit because it reuses
# gaps in earlier transformers, but it never reorders the queue, so heavy
# late-arriving blocks still strand capacity.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
K = inst["channels"]
demands = inst["demands"]

rem = []            # remaining kVA per energized transformer
cnt = []            # blocks already on each transformer
assign = []
for d in demands:
    placed = -1
    for i in range(len(rem)):
        if rem[i] >= d and cnt[i] < K:
            placed = i
            break
    if placed < 0:
        rem.append(C - d)
        cnt.append(1)
        assign.append(len(rem) - 1)
    else:
        rem[placed] -= d
        cnt[placed] += 1
        assign.append(placed)

print(json.dumps({"assign": assign}))
