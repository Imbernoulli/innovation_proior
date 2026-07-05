# TIER: strong
# 2-D vector packing.  Because a tour must fill BOTH the crowd cap C and the
# docent-minute budget T, we sort groups large-first under several normalized keys
# and place each with a BEST-FIT rule that minimizes leftover slack on both axes
# (a dot-product / tightest-remaining heuristic).  We run every (order, fit) combo
# and keep whichever uses the fewest tours.  Sorting big-first lets small/quick
# groups top off partially-filled tours, and best-fit prefers the tour whose
# residual capacity best matches the group's two-component demand -- so waste
# drops well below the online rules.  The loose volume bound keeps the normalized
# score below 1.0 on most instances.
import sys, json

inst = json.load(sys.stdin)
C = inst["C"]; T = inst["T"]
people = inst["people"]; minutes = inst["minutes"]
n = inst["n"]

# normalized demands in [0,1] per axis
np_ = [p / C for p in people]
nf_ = [f / T for f in minutes]

orders = {
    "sum":  sorted(range(n), key=lambda i: np_[i] + nf_[i], reverse=True),
    "max":  sorted(range(n), key=lambda i: max(np_[i], nf_[i]), reverse=True),
    "ppl":  sorted(range(n), key=lambda i: (people[i], minutes[i]), reverse=True),
    "min":  sorted(range(n), key=lambda i: minutes[i], reverse=True),
}


def pack(order, fit):
    # remaining capacity per tour on each axis
    rp = []; rf = []
    gof = [0] * n
    for i in order:
        p = people[i]; f = minutes[i]
        best = -1; best_key = None
        for b in range(len(rp)):
            if rp[b] >= p and rf[b] >= f:
                if fit == "tight":
                    # minimize combined leftover slack (normalized)
                    key = (rp[b] - p) / C + (rf[b] - f) / T
                elif fit == "dot":
                    # maximize alignment of demand with the tour's residual -> pick
                    # the tour whose leftover it consumes most fully (negate to min)
                    key = -((p / C) * (rp[b] / C) + (f / T) * (rf[b] / T))
                else:  # first-fit
                    key = b
                if best < 0 or key < best_key:
                    best_key = key; best = b
        if best < 0:
            rp.append(C - p); rf.append(T - f)
            gof[i] = len(rp) - 1
        else:
            rp[best] -= p; rf[best] -= f
            gof[i] = best
    return gof, len(rp)


best_assign = None
best_bins = None
for oname, order in orders.items():
    for fit in ("tight", "dot", "first"):
        gof, bins = pack(order, fit)
        if best_bins is None or bins < best_bins:
            best_bins = bins; best_assign = gof

print(json.dumps({"assign": best_assign}))
