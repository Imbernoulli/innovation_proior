# TIER: strong
# Insight: the non-passing constraint turns dispatch into an ONLINE
# INTERVAL-PARTITION problem, not a per-call matching problem. A nearest-car
# heuristic (the greedy trap) picks whichever car LOOKS closest right now
# and never checks whether the target floor is even physically reachable
# given the safety gap to its shaft partner -- so it routinely assigns a
# lower-role car a pickup that sits above (partner's floor - G), which is
# permanently unreachable while the partner holds its ground. This solution
# instead looks at the FULL known demand mix once and commits to a floor
# BAND per car that is provably safe to serve:
#
#   1. Cross-shaft split: sort all calls by ORIGIN and slice into S
#      contiguous, load-balanced groups -> each shaft owns a demand slice.
#   2. Role split within a shaft: sort that shaft's calls by DESTINATION
#      (not origin -- origins alone are uninformative when a trace clusters
#      all pickups near one floor, but destinations reveal the true spread
#      of where each car will actually need to travel) and take the lower
#      half as role 0, upper half as role 1.
#   3. Anticipatory parking: park each car at the MEDIAN of every floor it
#      will ever touch (its own calls' origins AND destinations), but never
#      closer to its partner's own working range than the safety gap --
#      i.e. size each park position off the PARTNER's farthest required
#      reach, not just this car's own footprint, so an idle car can never
#      sit exactly where its partner needs to go.
#   4. Repair pass: any call whose ORIGIN (the floor its assigned car MUST
#      physically reach to pick it up) ends up on the wrong side of the
#      other role's resting floor is reassigned to the role that can safely
#      reach it, and the park positions are recomputed -- this is what
#      prevents the exact reachability trap greedy falls into.
#
# Bands pushed apart by at least the safety gap mean the two cars of a shaft
# are (almost) never both drawn toward the same boundary, and batching falls
# out for free: a car whose band matches a burst of nearby calls sweeps them
# in one pass instead of re-approaching the same zone repeatedly.
import sys, json


def compute_park(F, G, calls, lo_idxs, hi_idxs, default_lo, default_hi):
    lo_touched = []
    for i in lo_idxs:
        lo_touched.append(calls[i]["o"]); lo_touched.append(calls[i]["d"])
    hi_touched = []
    for i in hi_idxs:
        hi_touched.append(calls[i]["o"]); hi_touched.append(calls[i]["d"])
    if lo_touched:
        lo_touched.sort()
        lo_center = lo_touched[len(lo_touched) // 2]
        lo_reach = max(lo_touched)
    else:
        lo_center, lo_reach = default_lo, default_lo
    if hi_touched:
        hi_touched.sort()
        hi_center = hi_touched[len(hi_touched) // 2]
        hi_reach = min(hi_touched)
    else:
        hi_center, hi_reach = default_hi, default_hi
    # hi car's park must stay clear of everywhere lo car ever needs to reach,
    # and vice versa -- otherwise a resting car can permanently block a busy
    # partner from a floor it must physically visit.
    hi_park = max(hi_center, lo_reach + G)
    lo_park = min(lo_center, hi_reach - G)
    lo_park = max(0, min(F - 1, lo_park))
    hi_park = max(0, min(F - 1, hi_park))
    if hi_park - lo_park < G:
        mid = (lo_park + hi_park) // 2
        lo_park = max(0, mid - G // 2)
        hi_park = min(F - 1, lo_park + G)
        if hi_park - lo_park < G:
            lo_park = max(0, hi_park - G)
    return lo_park, hi_park


inst = json.load(sys.stdin)
F, S, G = inst["F"], inst["S"], inst["G"]
calls = inst["calls"]
M = len(calls)
org = [calls[i]["o"] for i in range(M)]
dst = [calls[i]["d"] for i in range(M)]

# Step 1: contiguous, size-balanced shaft slices by ORIGIN.
order = sorted(range(M), key=lambda i: (org[i], calls[i]["id"]))
shaft_of = [0] * M
base, extra = M // S, M % S
ptr = 0
for s in range(S):
    take = base + (1 if s < extra else 0)
    for j in range(ptr, ptr + take):
        shaft_of[order[j]] = s
    ptr += take

assign = [None] * M
park = [0] * (2 * S)
default_lo = max(0, F // 2 - G)
default_hi = min(F - 1, F // 2 + G)

for s in range(S):
    idxs = [i for i in range(M) if shaft_of[i] == s]
    if not idxs:
        park[2 * s], park[2 * s + 1] = default_lo, default_hi
        continue
    # Step 2: role split within the shaft by DESTINATION (captures the true
    # spread of travel even when origins are all clustered together).
    idxs_by_dst = sorted(idxs, key=lambda i: (dst[i], calls[i]["id"]))
    n = len(idxs)
    half = max(1, n // 2) if n > 1 else 1
    lo_set = set(idxs_by_dst[:half] if n > 1 else idxs_by_dst)
    hi_set = set(idxs_by_dst[half:] if n > 1 else [])

    lo_park, hi_park = compute_park(F, G, calls, lo_set, hi_set, default_lo, default_hi)
    # Step 4: repair pass -- reassign any call whose ORIGIN sits too close
    # to / past the OTHER role's resting floor (this is what actually
    # causes an unreachable pickup / permanent deadlock).
    for _ in range(4):
        moved = False
        for i in list(lo_set):
            if org[i] > hi_park - G:
                lo_set.discard(i); hi_set.add(i); moved = True
        for i in list(hi_set):
            if org[i] < lo_park + G:
                hi_set.discard(i); lo_set.add(i); moved = True
        if not lo_set and hi_set:
            mv = min(hi_set, key=lambda i: org[i])
            hi_set.discard(mv); lo_set.add(mv); moved = True
        if not hi_set and lo_set:
            mv = max(lo_set, key=lambda i: org[i])
            lo_set.discard(mv); hi_set.add(mv); moved = True
        lo_park, hi_park = compute_park(F, G, calls, lo_set, hi_set, default_lo, default_hi)
        if not moved:
            break

    for i in lo_set:
        assign[i] = [s, 0]
    for i in hi_set:
        assign[i] = [s, 1]
    park[2 * s], park[2 * s + 1] = lo_park, hi_park

for i in range(M):
    if assign[i] is None:
        assign[i] = [0, 0]

print(json.dumps({"assign": assign, "park": park}))
