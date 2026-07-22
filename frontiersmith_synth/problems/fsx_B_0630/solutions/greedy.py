# TIER: greedy
# The "obvious" nearest-car dispatcher: process calls in arrival order, and for
# each one, hand it to whichever of the 2*S cars has the smallest PREDICTED
# distance to the origin floor (predicted position updated to the call's
# destination once "assigned" -- a normal single-pass dispatch heuristic).
# Every car's park/idle floor is left at the building's physical default
# (bottom for a lower car, top for an upper car) -- this heuristic reacts to
# calls as they are considered; it never commits to a demand-tuned band and
# never re-parks a car in anticipation of where the NEXT call will be. When
# demand clusters away from the extremes (or bursts near the shared safety
# gap), both cars of a shaft keep getting pulled toward the same busy zone,
# and the non-passing gap serializes them.
import sys, json

inst = json.load(sys.stdin)
F, S = inst["F"], inst["S"]
calls = sorted(inst["calls"], key=lambda c: (c["t"], c["id"]))
ncars = 2 * S

pos = []
for s in range(S):
    pos.append(0)
    pos.append(F - 1)

assign = [None] * len(inst["calls"])
for c in calls:
    best = min(range(ncars), key=lambda k: (abs(pos[k] - c["o"]), k))
    sh, ro = best // 2, best % 2
    assign[c["id"]] = [sh, ro]
    pos[best] = c["d"]

park = []
for s in range(S):
    park.append(0)
    park.append(F - 1)

print(json.dumps({"assign": assign, "park": park}))
