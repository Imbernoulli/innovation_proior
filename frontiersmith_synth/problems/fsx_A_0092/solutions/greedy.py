# TIER: greedy
# First-fit in ARRIVAL order, respecting BOTH constraints.  Process wells in the
# order they are listed (no reordering) and drop each into the first already-open
# loop where it still fits (flow) and keeps the temperature spread within band;
# otherwise open a new loop.  Reuses gaps that next-fit wastes (so it beats
# trivial), but not sorting by size leaves it well short of the decreasing-order
# packers.
import sys, json

inst = json.load(sys.stdin)
flow = inst["flow"]
temp = inst["temp"]
C = inst["capacity"]
band = inst["band"]
N = inst["n"]

loops = []          # each: [remaining_cap, tmin, tmax]
assign = [0] * N
for i in range(N):
    f = flow[i]
    t = temp[i]
    chosen = -1
    for j, (rem, tlo, thi) in enumerate(loops):
        if f <= rem:
            nlo = t if t < tlo else tlo
            nhi = t if t > thi else thi
            if nhi - nlo <= band:
                chosen = j
                break
    if chosen < 0:
        loops.append([C - f, t, t])
        assign[i] = len(loops) - 1
    else:
        rem, tlo, thi = loops[chosen]
        loops[chosen] = [rem - f, min(tlo, t), max(thi, t)]
        assign[i] = chosen

print(json.dumps({"assign": assign}))
