# TIER: greedy
# Color-aware FIRST-FIT in arrival order: house each contact in the lowest-index
# open pod that (a) still has room for its load AND (b) either already contains
# its lineage or has fewer than K distinct lineages.  Open a new pod only when
# none fits.  Reuses gaps in earlier pods -- better than next-fit -- but never
# reorders arrivals, so large / late contacts still strand capacity.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
K = inst["max_strains"]
loads = inst["loads"]
strains = inst["strains"]

rem = []            # remaining capacity per open pod
cols = []           # set of lineages per open pod
assign = []
for w, s in zip(loads, strains):
    placed = -1
    for i in range(len(rem)):
        if rem[i] >= w and (s in cols[i] or len(cols[i]) < K):
            placed = i
            break
    if placed < 0:
        rem.append(C - w)
        cols.append({s})
        assign.append(len(rem) - 1)
    else:
        rem[placed] -= w
        cols[placed].add(s)
        assign.append(placed)

print(json.dumps({"assign": assign}))
