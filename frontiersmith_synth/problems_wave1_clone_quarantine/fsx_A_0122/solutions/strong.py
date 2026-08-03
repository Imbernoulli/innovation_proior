# TIER: strong
# Bounded best-fit with capacity-aware eviction.  Keep up to K gantries open.  Route
# each platoon to the open gantry whose remaining room is SMALLEST but still fits
# (best-fit -> tight packing, keeps roomy gantries free for big platoons).  If none
# fits: if a slot is free open a new gantry; otherwise evict the FULLEST open gantry
# (smallest remaining room -- least useful going forward) to free a slot, then open a
# fresh one.  A far better bounded-space policy than next-fit / FIFO first-fit, yet
# the K-open cap and the loose L1 bound keep it below the ideal.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
K = inst["max_open"]
platoons = inst["platoons"]

next_id = 0
open_lanes = []          # list of [gantry_id, remaining]
assign = []

for s in platoons:
    best = None
    best_rem = None
    for lane in open_lanes:
        if lane[1] >= s and (best_rem is None or lane[1] < best_rem):
            best = lane
            best_rem = lane[1]
    if best is None:
        if len(open_lanes) >= K:
            # evict the fullest open gantry (smallest remaining room)
            fullest = min(range(len(open_lanes)), key=lambda j: open_lanes[j][1])
            open_lanes.pop(fullest)
        best = [next_id, C]
        next_id += 1
        open_lanes.append(best)
    best[1] -= s
    assign.append(best[0])

print(json.dumps({"assign": assign}))
