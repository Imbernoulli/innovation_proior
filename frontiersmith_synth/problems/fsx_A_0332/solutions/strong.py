# TIER: strong
# Class-aware packing.  The extra distinct-category limit means the winning idea is
# to keep items of the SAME category together (so a load spends few of its K class
# slots) while still filling weight.  We run two class-aware policies and keep the
# plan that dispatches fewer loads:
#
#   * grouped-FFD: order items by (category, then heaviest-first) so same-category
#     items are considered consecutively, then first-fit-decreasing into any load
#     whose weight and K-category limits both still allow the item -- preferring a
#     load that ALREADY carries this category (no class-slot cost).
#   * best-fit-decreasing, class-aware: sort ALL items heaviest-first and drop each
#     into the TIGHTEST feasible load, again preferring loads that already carry the
#     category so K-slots are conserved.
#
# Both beat the online rules by reusing weight gaps AND spending class slots
# deliberately; the loose two-term lower bound keeps the normalized score below 1.0
# on most instances.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
K = inst["classes"]
weights = inst["weights"]
category = inst["category"]
N = len(weights)


def pack(order):
    rem = []            # remaining weight per load
    cats = []           # category set per load
    bof = [0] * N       # load index per item
    for i in order:
        w = weights[i]
        c = category[i]
        # candidates that already carry this category (free on the class limit)
        best = -1
        best_rem = C + 1
        best_has = False
        for b in range(len(rem)):
            if rem[b] < w:
                continue
            has = (c in cats[b])
            if not has and len(cats[b]) >= K:
                continue
            # prefer loads that already carry c; among equal, tightest fit (best-fit)
            better = False
            if has and not best_has:
                better = True
            elif has == best_has and rem[b] < best_rem:
                better = True
            if better:
                best = b
                best_rem = rem[b]
                best_has = has
        if best < 0:
            rem.append(C - w)
            cats.append({c})
            bof[i] = len(rem) - 1
        else:
            rem[best] -= w
            cats[best].add(c)
            bof[i] = best
    return bof, len(rem)


# policy 1: grouped by category, heaviest-first within a category
grouped_order = sorted(range(N), key=lambda i: (category[i], -weights[i]))
g_assign, g_loads = pack(grouped_order)

# policy 2: global best-fit-decreasing by weight
bfd_order = sorted(range(N), key=lambda i: -weights[i])
b_assign, b_loads = pack(bfd_order)

assign = g_assign if g_loads <= b_loads else b_assign

print(json.dumps({"assign": assign}))
