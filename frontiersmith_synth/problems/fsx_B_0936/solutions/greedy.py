# TIER: greedy
# The obvious first-pass recipe: rank stations by their OWN net demand over
# the whole horizon (trips originated minus trips received -- a single
# scalar per station, computed once) and water-fill the shared budget to the
# neediest stations first, capped by dock capacity. This never resimulates
# and never looks at WHEN a station's trips happen, so it cannot see that a
# station with near-zero net demand can still have a large transient
# deficit if its outflow and inflow are separated in time (the coupling the
# statement warns about).
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
cap = inst["capacity"]
budget = inst["budget"]
trips = inst["trips"]

outc = [0] * n
inc = [0] * n
for (t, o, d, dt) in trips:
    outc[o] += 1
    inc[d] += 1
net = [max(0, outc[s] - inc[s]) for s in range(n)]

order = sorted(range(n), key=lambda s: (-net[s], s))
alloc = [0] * n
remaining = budget
want = list(net)
active = [s for s in order if want[s] > 0]
for _round in range(6):
    if remaining <= 0 or not active:
        break
    tot_want = sum(want[s] for s in active)
    if tot_want <= 0:
        break
    new_active = []
    given = 0
    for s in active:
        share = remaining * (want[s] / tot_want)
        give = min(share, want[s], cap[s] - alloc[s])
        give = int(give + 1e-9)  # floor
        if give > 0:
            alloc[s] += give
            given += give
            want[s] -= give
        if want[s] > 0 and alloc[s] < cap[s]:
            new_active.append(s)
    remaining -= given
    active = new_active
    if given == 0:
        break

print(json.dumps({"init": alloc}))
