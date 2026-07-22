# TIER: greedy
# Textbook recipe: treat changeovers as a FIXED matrix (the direct dilution cost
# between every pair of colours, computed once) and build a single-visit wheel with
# nearest-neighbour construction over that matrix -- classic "cheapest changeover
# next" heuristic, no lookahead, no reordering search.  Lot sizes are set by a
# self-consistent EOQ-style rule: iterate lot_i ~= demand_i * cycle_time until the
# implied cycle time stops moving.  This never questions whether a colour should be
# produced more than once, and never asks whether some OTHER colour, not adjacent to
# the expensive pair, could shrink the flush by being routed through in between --
# it only ever looks at the matrix of DIRECT costs.
import sys, json, math

inst = json.load(sys.stdin)
colors = inst["colors"]
k = inst["k"]
lam = inst["lambda"]
flush_cost = inst["flush_cost"]
max_lot = inst["max_lot"]
max_campaigns = inst["max_campaigns"]

tints = [c["tint"] for c in colors]
taus = [c["tau"] for c in colors]
demand = [c["demand"] for c in colors]
minlot = [c["min_lot"] for c in colors]


def waste(i, j):
    if i == j:
        return 0
    diff = abs(tints[i] - tints[j])
    if diff <= taus[j]:
        return 0
    ratio = diff / float(taus[j])
    steps = math.log(ratio) / math.log(1.0 / lam)
    return flush_cost * max(0, int(math.ceil(steps - 1e-9)))


W = [[waste(i, j) for j in range(k)] for i in range(k)]

# nearest-neighbour tour over the DIRECT matrix only
unvisited = set(range(k))
cur = 0
unvisited.discard(cur)
tour = [cur]
while unvisited:
    nxt = min(unvisited, key=lambda j: W[cur][j])
    tour.append(nxt)
    unvisited.discard(nxt)
    cur = nxt


def timeline(order, lots):
    t = 0
    prev = order[-1]
    for idx, c in enumerate(order):
        t += waste(prev, c)
        t += lots[idx]
        prev = c
    return t


Test = 300.0
lots = [minlot[c] for c in tour]
for _ in range(8):
    lots = []
    for c in tour:
        L = max(minlot[c], int(round(demand[c] * Test)))
        L = min(L, max_lot)
        lots.append(L)
    Test = timeline(tour, lots)

wheel = [{"color": tour[idx], "lot": lots[idx]} for idx in range(len(tour))]
wheel = wheel[:max_campaigns]
print(json.dumps({"wheel": wheel}))
