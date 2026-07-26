# TIER: strong
# System identification, not rule selection. scan_period/scan_span/
# scan_phase and inversion_period/n_eras are handed to us verbatim every
# round, so we never have to *detect* a tour bus or a crowd rotation --
# we compute directly from the round counter whether one is happening.
#
#  (a) scan-resistance: during a disclosed bus round, never spend a
#      memorized slot on a one-shot tourist face. Zero cost -- a face that
#      is mathematically guaranteed never to return would give at most one
#      hit if admitted, at the price of evicting a regular who is worth
#      many.
#  (b) rotation-aware eviction: the club alternates between n_eras fixed
#      regular crowds on a disclosed period. We tag every face with the
#      era it was first met in (ghost-history bookkeeping threaded through
#      "state", since each round is a fresh isolated process with no other
#      memory) and give each era a fair-share reserve of capacity//n_eras
#      slots. When the list is full we evict from whichever era currently
#      holds the most slots ABOVE its fair share (breaking ties by
#      recency) -- a soft water-filling reserve, not a hard wall, so an
#      era with real demand can still borrow room, but an era that is
#      hogging slots gets drained first. A crowd due back next rotation
#      keeps its fair share resident instead of aging out completely
#      while it's away.
import sys, json

inst = json.load(sys.stdin)
capacity = inst["capacity"]
r = inst["round"]
sp, ss, sph = inst["scan_period"], inst["scan_span"], inst["scan_phase"]
ip, ne = inst["inversion_period"], inst["n_eras"]
floor = set(inst["floor"])
arrivals = inst["arrivals"]
state = inst.get("state") or {}
clock = int(state.get("clock", 0))
recency = {k: v for k, v in (state.get("recency") or {}).items() if k in floor}
era_of = {k: v for k, v in (state.get("era_of") or {}).items() if k in floor}

is_scan = sp > 0 and ((r - sph) % sp) < ss
cur_era = ((r // ip) % ne) if (ne > 1 and ip > 0) else 0
n_eras_eff = ne if ne > 0 else 1
reserve = [capacity // n_eras_eff + (1 if e < capacity % n_eras_eff else 0) for e in range(n_eras_eff)]

decisions = []
if is_scan:
    for key in arrivals:
        clock += 1
        if key in floor:
            recency[key] = clock
            decisions.append({"action": "skip", "evict": None})
        else:
            decisions.append({"action": "skip", "evict": None})
else:
    for key in arrivals:
        clock += 1
        if key in floor:
            recency[key] = clock
            decisions.append({"action": "skip", "evict": None})
            continue
        if len(floor) < capacity:
            floor.add(key)
            era_of[key] = cur_era
            recency[key] = clock
            decisions.append({"action": "admit", "evict": None})
        else:
            counts = [0] * n_eras_eff
            for k in floor:
                counts[era_of.get(k, cur_era)] += 1
            over_era = max(range(n_eras_eff), key=lambda e: counts[e] - reserve[e])
            pool = [k for k in floor if era_of.get(k, cur_era) == over_era]
            if not pool:
                pool = list(floor)
            victim = min(pool, key=lambda k: recency.get(k, -1))
            floor.discard(victim)
            recency.pop(victim, None)
            era_of.pop(victim, None)
            floor.add(key)
            era_of[key] = cur_era
            recency[key] = clock
            decisions.append({"action": "admit", "evict": victim})

out_state = {"clock": clock, "recency": recency, "era_of": era_of}
print(json.dumps({"decisions": decisions, "state": out_state}))
