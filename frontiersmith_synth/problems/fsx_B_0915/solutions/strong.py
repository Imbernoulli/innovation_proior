# TIER: strong
# Inverts the standard decomposition: schedule CURB RESERVATIONS first,
# then fit vehicles to them (instead of nearest-vehicle-first).
#   1. Group pending requests by preferred pickup curb.
#   2. Within each group, form pickup EVENTS of up to veh_capacity riders
#      (pooling -- one vehicle trip serves several riders at once, cutting
#      the number of distinct curb visits the fleet needs to make).
#   3. Resolve which PHYSICAL curb each event actually uses by spreading
#      load across the request's tolerated pickup_options (a small walk
#      penalty) so two events aren't both planned onto the same curb this
#      tick -- this is the curb-capacity reservation, decided BEFORE any
#      vehicle is chosen.
#   4. Only THEN match the nearest still-available idle vehicle to each
#      resolved event, most-urgent (oldest release) event first.
#   5. Order each pooled trip's dropoffs by proximity to the pickup curb
#      to keep the realized detour small.
import sys, json
from collections import defaultdict

def main():
    view = json.load(sys.stdin)
    pending = view["pending"]
    idle = [v for v in view["vehicles"] if v["status"] == "idle"]
    VC = view["veh_capacity"]
    if not pending or not idle:
        print(json.dumps({"assign": {}}))
        return

    groups = defaultdict(list)
    for r in pending:
        groups[r["pickup_pref"]].append(r)

    curb_load = defaultdict(int)
    events = []  # {"curb":c, "requests":[...]}
    for pref in sorted(groups):
        reqs = groups[pref]
        # Only pool riders whose dropoff is on the SAME SIDE of the pickup
        # (both ahead, or both behind) -- pooling riders headed opposite
        # ways is a bad trade (huge realized detour), so those become
        # SEPARATE single-rider events instead, which is exactly what
        # forces curb spreading to avoid contention between them.
        left = sorted([r for r in reqs if r["dropoff"] < pref],
                       key=lambda r: (-r["dropoff"], r["release_tick"], r["id"]))
        right = sorted([r for r in reqs if r["dropoff"] > pref],
                        key=lambda r: (r["dropoff"], r["release_tick"], r["id"]))
        same = [r for r in reqs if r["dropoff"] == pref]
        batches = []
        for lst in (left, right):
            i = 0
            while i < len(lst):
                batches.append(lst[i:i + VC])
                i += VC
        for r in same:
            batches.append([r])
        for batch in batches:
            common = set(batch[0]["pickup_options"])
            for r in batch[1:]:
                common &= set(r["pickup_options"])
            if not common:
                common = {pref}
            chosen = min(common, key=lambda c: (curb_load[c], abs(c - pref), c))
            curb_load[chosen] += 1
            events.append({"curb": chosen, "requests": batch})

    events.sort(key=lambda e: (min(r["release_tick"] for r in e["requests"]),
                                min(r["id"] for r in e["requests"])))

    avail = list(idle)
    assign = {}
    for ev in events:
        if not avail:
            break
        best = min(avail, key=lambda v: (abs(v["pos"] - ev["curb"]), v["id"]))
        avail = [v for v in avail if v["id"] != best["id"]]
        stops = [{"action": "pickup", "request": r["id"], "curb": ev["curb"]} for r in ev["requests"]]
        drop_order = sorted(ev["requests"], key=lambda r: (abs(r["dropoff"] - ev["curb"]), r["id"]))
        for r in drop_order:
            stops.append({"action": "dropoff", "request": r["id"]})
        assign[str(best["id"])] = stops

    print(json.dumps({"assign": assign}))

main()
