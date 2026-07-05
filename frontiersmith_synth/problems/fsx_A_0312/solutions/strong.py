# TIER: strong
# Best-Fit-Decreasing + hauler-emptying local search.
#   1) Sort fragments by DECREASING mass and place each with Best-Fit: into the
#      hauler that is left with the LEAST slack that still fits it (tie-break lowest
#      index), else open a new hauler.  Packing big fragments first and hugging
#      capacity is the classic ~11/9-OPT rule and beats plain First-Fit.
#   2) Local search: repeatedly take the LEAST-loaded hauler and try to relocate all
#      of its fragments into OTHER haulers by Best-Fit; if every one fits elsewhere,
#      dispatch one fewer hauler.  Deterministic scan + a fixed pass cap keep it
#      reproducible.
# The output maps every fragment back to its ORIGINAL arrival index.  This clears
# both online rules, but the loose L1 lower bound keeps the normalized score < 1.0.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
masses = inst["masses"]
n = len(masses)

# ---- 1) Best-Fit-Decreasing -------------------------------------------------
order = sorted(range(n), key=lambda i: (-masses[i], i))
bins = []            # list of lists of original indices
loads = []           # parallel loads
for i in order:
    m = masses[i]
    best = -1
    best_slack = None
    for b in range(len(bins)):
        slack = C - loads[b]
        if slack >= m and (best_slack is None or slack < best_slack):
            best_slack = slack
            best = b
    if best < 0:
        best = len(bins)
        bins.append([])
        loads.append(0)
    bins[best].append(i)
    loads[best] += m

# ---- 2) least-loaded-hauler emptying ---------------------------------------
def try_empty():
    if len(bins) <= 1:
        return False
    # index of the least-loaded hauler
    src = min(range(len(bins)), key=lambda b: (loads[b], b))
    # tentative relocation of every fragment in src via Best-Fit into the others
    tmp_loads = list(loads)
    moves = []
    for i in bins[src]:
        m = masses[i]
        best = -1
        best_slack = None
        for b in range(len(bins)):
            if b == src:
                continue
            slack = C - tmp_loads[b]
            if slack >= m and (best_slack is None or slack < best_slack):
                best_slack = slack
                best = b
        if best < 0:
            return False
        tmp_loads[best] += m
        moves.append((i, best))
    # commit
    for (i, b) in moves:
        bins[b].append(i)
    loads[:] = tmp_loads
    del bins[src]
    del loads[src]
    return True

for _ in range(2 * n):
    if not try_empty():
        break

# ---- emit assignment in original arrival order ------------------------------
assign = [0] * n
for b in range(len(bins)):
    for i in bins[b]:
        assign[i] = b

print(json.dumps({"assign": assign}))
