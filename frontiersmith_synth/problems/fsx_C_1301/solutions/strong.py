# TIER: strong
# Fleet-level charging-SLOT allocation, not per-van charging-TIME allocation.
# Start every van at the obvious plan (charge to full at its late stop p2, like
# greedy -- individually correct in isolation). Then repeatedly ask: of the vans
# still on that plan, is there ONE whose move to its EARLY stop p1 instead --
# charging before it strictly needs to, at a *different* charger resource and a
# *different, earlier* point in time -- improves the fleet's TRUE on-time
# fraction the most, when the whole fleet's plugs are re-simulated together? Take
# that move, commit it, and repeat. A single van in isolation never has a reason
# to prefer p1 (p2 alone is always individually sufficient), so no per-van
# planner ever considers this; it only becomes visible once every van's plan is
# evaluated side by side against the shared, scarce chargers. Because every
# candidate move is checked against the SAME discrete-event simulation the judge
# uses, this hill-climb only ever accepts moves that make the real fleet-wide
# schedule strictly better, so it can never do worse than the naive plan it
# starts from.
import sys, json, math, heapq

inst = json.load(sys.stdin)
vans = inst["vans"]
chargers = {c["id"]: c for c in inst["chargers"]}


def cum(legs):
    t = 0; e = 0
    times = [0]; energies = [0]
    for leg in legs:
        t += leg["time"]; e += leg["energy"]
        times.append(t); energies.append(e)
    return times, energies


info = {}
for van in vans:
    times, energies = cum(van["legs"])
    p1, p2 = van["p1"], van["p2"]
    info[van["id"]] = {
        "need_p1": energies[p1], "need_p2": energies[p2],
        "L": len(van["legs"]),
    }


def simulate_fraction(assign):
    """assign: vid -> 'p1' or 'p2'. Runs the same discrete-event fleet simulation
    the judge uses and returns the on-time fraction (public-info only)."""
    plug_free = {c["id"]: [0.0] * c["slots"] for c in inst["chargers"]}
    heap = [(0.0, van["id"], 0, float(van["capacity"])) for van in vans]
    heapq.heapify(heap)
    by_id = {van["id"]: van for van in vans}
    stranded = set()
    on_time = 0
    total = sum(len(van["legs"]) for van in vans)
    while heap:
        t, vid, i, soc = heapq.heappop(heap)
        if vid in stranded:
            continue
        van = by_id[vid]
        L = len(van["legs"])
        if i >= L:
            continue
        leg = van["legs"][i]
        arr_time = t + leg["time"]
        arr_soc = soc - leg["energy"]
        stop_idx = i + 1
        if arr_soc < -1e-9:
            stranded.add(vid)
            continue
        amt = 0.0
        charger = None
        if stop_idx == van["p1"] and assign[vid] == "p1":
            amt = info[vid]["need_p1"]
            charger = chargers[van["p1_charger"]]
        elif stop_idx == van["p2"] and assign[vid] == "p2":
            amt = info[vid]["need_p2"]
            charger = chargers[van["p2_charger"]]
        headroom = max(0.0, van["capacity"] - arr_soc)
        amt = min(amt, headroom)
        if charger is not None and amt > 1e-12:
            pf = plug_free[charger["id"]]
            k = min(range(len(pf)), key=lambda x: pf[x])
            start = max(arr_time, pf[k])
            dur = math.ceil(amt / charger["rate"])
            finish = start + dur
            pf[k] = finish
            soc_after = arr_soc + amt
            depart = finish
        else:
            soc_after = arr_soc
            depart = arr_time
        if arr_time <= van["deadlines"][i] + 1e-9:
            on_time += 1
        if i + 1 < L:
            heapq.heappush(heap, (depart, vid, i + 1, soc_after))
    return on_time / total if total else 0.0


assign = {van["id"]: "p2" for van in vans}
best_frac = simulate_fraction(assign)
still_p2 = [van["id"] for van in vans]
max_rounds = min(len(vans), 15)

for _round in range(max_rounds):
    best_move = None
    best_gain = 1e-9
    for vid in still_p2:
        trial = dict(assign); trial[vid] = "p1"
        f = simulate_fraction(trial)
        if f - best_frac > best_gain:
            best_gain = f - best_frac
            best_move = (vid, f)
    if best_move is None:
        break
    vid, f = best_move
    assign[vid] = "p1"
    best_frac = f
    still_p2.remove(vid)

out_vans = []
for van in vans:
    vid = van["id"]; ii = info[vid]
    if assign[vid] == "p1":
        out_vans.append({"id": vid, "charge_at_p1": ii["need_p1"], "charge_at_p2": 0})
    else:
        out_vans.append({"id": vid, "charge_at_p1": 0, "charge_at_p2": ii["need_p2"]})

print(json.dumps({"vans": out_vans}))
