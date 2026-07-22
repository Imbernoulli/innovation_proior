# TIER: strong
# INSIGHT: group allocations by anticipated FREE-TIME, not by size or arrival.
# Because the vault can never compact, the enemy is a long-lived block stranded in
# the middle of a region that short-lived cohorts keep churning.  So:
#   * segregate the long-lived "spine" (blocks that share the maximal death time)
#     and pack it LOW and contiguously -- it never has to move again;
#   * process the rest in death-time order so an entire equal-death COHORT lands in
#     one contiguous band that reopens as a single clean hole for the next stage.
# We realize this by choosing a PROCESSING ORDER keyed on death time (longest-lived
# first, ties by birth then size) and placing each block lowest-fit.  Lifetime
# order keeps holes contiguous; several lifetime-based orders are tried and the
# best (lowest high-water mark) is returned.  The loose concurrent-demand bound
# keeps the normalized score well under 1.0, so headroom remains above this
# reference.
import sys, json

inst = json.load(sys.stdin)
blocks = inst["blocks"]
M = len(blocks)


def place(order):
    placed = []          # (birth, death, off, size)
    off = [0] * M
    peak = 0
    for i in order:
        s = blocks[i]["size"]
        b = blocks[i]["birth"]
        d = blocks[i]["death"]
        occ = []
        for (pb, pd, po, ps) in placed:
            if pb < d and b < pd:
                occ.append((po, po + ps))
        occ.sort()
        o = 0
        for (lo, hi) in occ:
            if o + s <= lo:
                break
            if o < hi:
                o = hi
        off[i] = o
        placed.append((b, d, o, s))
        if o + s > peak:
            peak = o + s
    return off, peak


sz = [blocks[i]["size"] for i in range(M)]
bir = [blocks[i]["birth"] for i in range(M)]
dea = [blocks[i]["death"] for i in range(M)]
maxdeath = max(dea) if M else 0

orders = []
# (a) longest-lived first: death desc, then birth asc, then size desc
orders.append(sorted(range(M), key=lambda i: (-dea[i], bir[i], -sz[i])))
# (b) longevity (death-birth) desc: reserve the load-bearing long spans low
orders.append(sorted(range(M), key=lambda i: (-(dea[i] - bir[i]), bir[i], -sz[i])))
# (c) explicit spine/cohort split: spine (death == maxdeath) low by size desc,
#     then cohorts by ascending death time, within a cohort by size desc
def key_c(i):
    spine = 0 if dea[i] == maxdeath else 1
    return (spine, dea[i], -sz[i], bir[i])
orders.append(sorted(range(M), key=key_c))
# (d) death asc, birth asc (cohorts contiguous, short churn on top)
orders.append(sorted(range(M), key=lambda i: (dea[i], bir[i], -sz[i])))
# (e) plain arrival order -- so lifetime awareness never LOSES on unstructured
#     instances where first-fit is already near optimal
orders.append(sorted(range(M), key=lambda i: (bir[i], i)))

best_off = None
best_peak = None
for od in orders:
    off, peak = place(od)
    if best_peak is None or peak < best_peak:
        best_peak = peak
        best_off = off

print(json.dumps({"offset": best_off}))
