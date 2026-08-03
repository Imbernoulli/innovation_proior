# TIER: greedy
# Bounded first-fit.  Keep up to K gantries open.  Route each arriving platoon to
# the FIRST open gantry (lowest id) with room; if none fits and a slot is free,
# open a new gantry; if none fits and all K are open, CLOSE the oldest open gantry
# (FIFO) to free a slot, then open a new one.  Reuses gaps that next-fit throws
# away, so it opens fewer gantries -- but the naive FIFO eviction wastes room.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
K = inst["max_open"]
platoons = inst["platoons"]

next_id = 0
open_lanes = []          # list of [gantry_id, remaining] in open ORDER (oldest first)
assign = []

for s in platoons:
    placed = None
    for lane in open_lanes:
        if lane[1] >= s:
            placed = lane
            break
    if placed is None:
        if len(open_lanes) >= K:
            open_lanes.pop(0)          # close the oldest open gantry
        placed = [next_id, C]
        next_id += 1
        open_lanes.append(placed)
    placed[1] -= s
    assign.append(placed[0])

print(json.dumps({"assign": assign}))
