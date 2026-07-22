# TIER: greedy
# Online best-fit: load each arriving parcel onto whichever already-open truck
# leaves the LEAST slack after loading it; open a new truck only if nothing
# fits. This is the textbook "smarter than next-fit" online bin-packing rule
# an average strong coder reaches for first -- it never reorders, never
# reserves headroom, and never spends the repack budget. On a steady mix it
# beats next-fit-commit by reusing gaps; but when medium parcels arrive first,
# best-fit packs trucks to a level that leaves no room at all for a later
# shift toward oversized parcels, so it can lose most of its advantage (or
# worse) exactly on the surge instances.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
sizes = inst["sizes"]

rem = []          # remaining capacity per open truck
assign = []
for s in sizes:
    best = -1
    best_rem = None
    for i in range(len(rem)):
        if rem[i] >= s:
            leftover = rem[i] - s
            if best == -1 or leftover < best_rem:
                best = i
                best_rem = leftover
    if best == -1:
        rem.append(C - s)
        assign.append(len(rem) - 1)
    else:
        rem[best] -= s
        assign.append(best)

print(json.dumps({"placements": assign, "moves": []}))
