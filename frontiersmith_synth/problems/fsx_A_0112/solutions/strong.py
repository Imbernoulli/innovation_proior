# TIER: strong
# Class-aware decreasing-order packing.  Plain FFD/BFD is not enough here: the
# K-lineage limit means naive weight-sorting can strand small contacts of a
# lineage that no pod will accept.  We run several deterministic constructive
# policies and keep whichever opens the FEWEST pods:
#   A) BFD-load        : largest loads first, best-fit (tightest feasible pod).
#   B) BFD-load+color  : same, but among feasible pods prefer one that already
#                        contains the contact's lineage (free color slot).
#   C) strain-grouped  : order contacts by (lineage, decreasing load) so each
#                        lineage packs contiguously and pods fill their K color
#                        slots with whole lineages before spilling over.
# A pod is feasible for a contact iff load fits AND (lineage present OR <K colors).
# The loose L1 bound (which ignores colors) keeps the normalized score below 1.0
# on the color-dominated instances, so real headroom remains.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
K = inst["max_strains"]
loads = inst["loads"]
strains = inst["strains"]
N = len(loads)


def _feasible(rem, cols, i, w, s):
    return rem[i] >= w and (s in cols[i] or len(cols[i]) < K)


def pack(order, color_pref):
    rem = []
    cols = []
    gof = [0] * N
    for idx in order:
        w = loads[idx]
        s = strains[idx]
        best = -1
        best_key = None
        for i in range(len(rem)):
            if not _feasible(rem, cols, i, w, s):
                continue
            # primary: prefer pods already holding this lineage (if color_pref);
            # secondary: tightest remaining capacity (best-fit).
            has_color = 0 if (color_pref and s in cols[i]) else 1
            key = (has_color, rem[i])
            if best < 0 or key < best_key:
                best = i
                best_key = key
        if best < 0:
            rem.append(C - w)
            cols.append({s})
            gof[idx] = len(rem) - 1
        else:
            rem[best] -= w
            cols[best].add(s)
            gof[idx] = best
    return gof, len(rem)


order_load = sorted(range(N), key=lambda i: loads[i], reverse=True)
order_strain = sorted(range(N), key=lambda i: (strains[i], -loads[i]))

candidates = [
    pack(order_load, False),
    pack(order_load, True),
    pack(order_strain, True),
]

best_assign, best_pods = min(candidates, key=lambda t: t[1])
print(json.dumps({"assign": best_assign}))
