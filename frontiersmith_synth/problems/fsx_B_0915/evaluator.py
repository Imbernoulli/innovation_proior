import sys, json, random, isorun
from collections import defaultdict

# ==========================================================================
# fsx_B_0915 -- curbside-pooling-dispatch (Format B, isolated candidate)
# Theme: "shared shuttles fighting over scarce curb space".
#
# A small fleet of pooled shuttles serves an ONLINE stream of ride requests
# on a 1-D strip of S curb slots (0..S-1). Each request has a PREFERRED
# pickup curb but tolerates boarding one slot to either side (a small walk
# penalty applies). Every curb slot can service at most `curb_cap` vehicles
# in the SAME tick; a vehicle that reaches a curb already at capacity must
# CIRCLE (fixed penalty, retried next tick). Vehicles carry up to
# `veh_capacity` riders at once; picking up a second rider before dropping
# the first (pooling) is allowed but the vehicle's real path is replayed,
# so any zig-zag it causes is charged as DETOUR to every onboard rider.
#
# The episode runs for T ticks, graded in T causal DECISION EPOCHS: at the
# start of every tick the candidate is invoked FRESH (isolated OS-sandboxed
# subprocess) with every request released so far that is still unassigned,
# and the status/position of every vehicle, and must return route
# assignments for any subset of the CURRENTLY IDLE vehicles only. Requests
# it doesn't assign simply stay pending for a later tick. This guarantees
# the candidate never sees a request before it is released.
# ==========================================================================

S = 10
V = 3
VC = 2          # veh_capacity (riders onboard at once)
T = 14
CIRCLE_PENALTY = 6.0
WALK_PENALTY = 2.0
ABANDON_PENALTY = 25.0


def opts_of(pref):
    return sorted(set([max(0, pref - 1), pref, min(S - 1, pref + 1)]))


def mk_req(rid, rt, pref, drop):
    return {"id": rid, "release_tick": rt, "pickup_pref": pref,
            "pickup_options": opts_of(pref), "dropoff": drop}


def gen_random_fill(rng, start_id, n, avoid_ticks_used, tmax):
    out = []
    rid = start_id
    for _ in range(n):
        rt = rng.randint(0, tmax)
        pref = rng.randint(0, S - 1)
        drop = rng.randint(0, S - 1)
        while drop == pref:
            drop = rng.randint(0, S - 1)
        out.append(mk_req(rid, rt, pref, drop))
        rid += 1
    return out, rid


def build(idx, vehicle_start, curb_cap, hand_reqs, rng, n_fill, tmax=T - 4):
    reqs = list(hand_reqs)
    start_id = (max((r["id"] for r in reqs), default=-1)) + 1
    fill, _ = gen_random_fill(rng, start_id, n_fill, None, tmax)
    reqs.extend(fill)
    reqs.sort(key=lambda r: (r["release_tick"], r["id"]))
    pub = {"S": S, "V": V, "veh_capacity": VC, "T": T, "curb_cap": curb_cap,
           "vehicle_start": list(vehicle_start)}
    return {"public": pub, "hidden": {"requests": reqs}}


def make_instances():
    out = []

    # ---- Instance 0: classic curb-convergence trap -- ALL THREE vehicles
    #      equidistant from one hot curb (two share a start slot), tight
    #      capacity. Greedy's nearest-vehicle rule sends all three straight
    #      at the same curb, arriving on the SAME tick (double circling). A
    #      curb-reservation-first policy pools the two same-direction
    #      riders and spreads the third onto an adjacent curb instead. ----
    rng = random.Random(9000)
    hand = [mk_req(0, 0, 5, 8), mk_req(1, 0, 5, 9), mk_req(2, 0, 5, 1)]
    out.append(build(0, [2, 2, 8], [1] * S, hand, rng, 2))

    # ---- Instance 1: a second, differently-shaped trap -- one guaranteed
    #      2-vehicle equidistant collision at curb 3 (v0,v1, both dist 2),
    #      PLUS a fleet-scarcity test: by the time the curb-7 pair releases
    #      (tick 1) both v0 and v1 are already committed to curb 3, so only
    #      v2 is free -- nearest-vehicle dispatch grabs it for one curb-7
    #      request immediately and leaves the other stranded, whereas
    #      curb-reservation-first recognizes both curb-7 riders share a
    #      pickup and pools them onto v2's single trip instead. ----
    rng = random.Random(9001)
    hand = [mk_req(0, 0, 3, 9), mk_req(1, 0, 3, 0),
            mk_req(2, 1, 7, 0), mk_req(3, 1, 7, 1)]
    out.append(build(1, [1, 5, 9], [1] * S, hand, rng, 3))

    # ---- Instance 2: pooling-necessity (loose curb cap=2 isolates the
    #      pooling benefit from contention): two riders share a pickup curb
    #      AND a compatible dropoff direction while the fleet is otherwise
    #      busy on decoys -- only pooling them onto ONE trip frees a vehicle
    #      fast enough to also serve everything else well. ----
    rng = random.Random(9002)
    hand = [mk_req(0, 0, 4, 6), mk_req(1, 0, 4, 9),
            mk_req(2, 0, 0, 2), mk_req(3, 1, 9, 7)]
    out.append(build(2, [4, 0, 9], [2] * S, hand, rng, 4))

    # ---- Instance 3: combined trap -- collision AND pooling opportunity
    #      overlapping, tight capacity. ----
    rng = random.Random(9003)
    hand = [mk_req(0, 0, 6, 8), mk_req(1, 0, 6, 9),
            mk_req(2, 0, 6, 3), mk_req(3, 2, 1, 0)]
    out.append(build(3, [3, 9, 9], [1] * S, hand, rng, 4))

    # ---- Instances 4-9: seeded pseudo-random generalization/control mix
    #      (varying curb capacity, vehicle starts, request volume; some
    #      loose (cap=2, near i.i.d.), some tight and cluster-prone). ----
    ctrl_specs = [
        (4, [0, 5, 9], [1] * S, 10, 9004),
        (5, [2, 4, 6], [1] * S, 11, 9005),
        (6, [0, 3, 9], [2] * S, 10, 9006),   # loose control
        (7, [1, 5, 8], [1] * S, 12, 9007),
        (8, [0, 9, 4], [2] * S, 9, 9008),    # loose control
        (9, [3, 3, 7], [1] * S, 12, 9009),
    ]
    for idx, vstart, cap, n_fill, seed in ctrl_specs:
        rng = random.Random(seed)
        out.append(build(idx, vstart, cap, [], rng, n_fill, tmax=T - 3))

    return out


# ------------------------------- physics --------------------------------

def run_episode(inst, decide):
    """Drive the causal per-tick simulation. `decide(view) -> assign dict,
    or the string '__FAIL__' to abort the whole instance (candidate crash
    / malformed output for that tick)."""
    pub = inst["public"]
    Sn, capL, VCn, Vn, Tn = pub["S"], pub["curb_cap"], pub["veh_capacity"], pub["V"], pub["T"]
    requests = inst["hidden"]["requests"]
    req_by_id = {r["id"]: r for r in requests}
    reqs_by_tick = defaultdict(list)
    for r in requests:
        reqs_by_tick[r["release_tick"]].append(r)

    pending_ids = set()
    delivered = set()
    wait = {}
    pickup_tick = {}
    pickup_curb = {}
    vehicles = [{"id": i, "pos": pub["vehicle_start"][i], "route": []} for i in range(Vn)]
    total = 0.0

    for t in range(Tn):
        for r in reqs_by_tick.get(t, []):
            pending_ids.add(r["id"])

        view = {
            "S": Sn, "curb_cap": capL[0] if isinstance(capL, list) else capL,
            "veh_capacity": VCn, "V": Vn, "T": Tn, "tick": t,
            "vehicles": [{"id": v["id"], "pos": v["pos"],
                          "status": "idle" if not v["route"] else "busy"} for v in vehicles],
            "pending": [{"id": rid, "release_tick": req_by_id[rid]["release_tick"],
                         "pickup_pref": req_by_id[rid]["pickup_pref"],
                         "pickup_options": list(req_by_id[rid]["pickup_options"]),
                         "dropoff": req_by_id[rid]["dropoff"]}
                        for rid in sorted(pending_ids)],
        }
        assign = decide(view)
        if assign == "__FAIL__":
            return False, 0.0
        if assign is None:
            assign = {}
        if not isinstance(assign, dict):
            return False, 0.0

        idle_ids = {v["id"] for v in vehicles if not v["route"]}
        used_this_tick = set()
        new_routes = {}
        for vid_raw, stops in assign.items():
            try:
                vid = int(vid_raw)
            except (TypeError, ValueError):
                return False, 0.0
            if vid not in idle_ids or vid in new_routes:
                return False, 0.0
            if not isinstance(stops, list) or not stops or len(stops) > 64:
                return False, 0.0
            onboard = set()
            route = []
            for st in stops:
                if not isinstance(st, dict):
                    return False, 0.0
                act = st.get("action")
                rid = st.get("request")
                if act not in ("pickup", "dropoff"):
                    return False, 0.0
                if isinstance(rid, bool) or not isinstance(rid, int) or rid not in req_by_id:
                    return False, 0.0
                if act == "pickup":
                    if rid not in pending_ids or rid in used_this_tick:
                        return False, 0.0
                    c = st.get("curb")
                    opts = req_by_id[rid]["pickup_options"]
                    if isinstance(c, bool) or not isinstance(c, int) or c not in opts:
                        return False, 0.0
                    if len(onboard) >= VCn:
                        return False, 0.0
                    onboard.add(rid)
                    used_this_tick.add(rid)
                    route.append(("pickup", rid, c))
                else:
                    if rid not in onboard:
                        return False, 0.0
                    onboard.discard(rid)
                    route.append(("dropoff", rid, req_by_id[rid]["dropoff"]))
            if onboard:
                return False, 0.0
            new_routes[vid] = route

        for vid, route in new_routes.items():
            vehicles[vid]["route"] = route
            for kind, rid, _c in route:
                if kind == "pickup":
                    pending_ids.discard(rid)

        cap = capL[0] if isinstance(capL, list) else capL
        ready = []
        for v in vehicles:
            if not v["route"]:
                continue
            kind, rid, curb = v["route"][0]
            if v["pos"] != curb:
                v["pos"] += 1 if curb > v["pos"] else -1
            # Same-tick service: a vehicle that is AT (or just arrived at)
            # its stop's curb this tick attempts to service it this tick,
            # matching the statement ("when it reaches that curb it
            # attempts to service the stop").
            if v["pos"] == curb:
                ready.append(v["id"])

        bycurb = defaultdict(list)
        for vid in ready:
            kind, rid, curb = vehicles[vid]["route"][0]
            bycurb[curb].append(vid)
        for curb, vids in bycurb.items():
            vids_sorted = sorted(vids)
            winners = vids_sorted[:cap]
            losers = vids_sorted[cap:]
            total += CIRCLE_PENALTY * len(losers)
            for vid in winners:
                kind, rid, curb2 = vehicles[vid]["route"].pop(0)
                if kind == "pickup":
                    pickup_tick[rid] = t
                    pickup_curb[rid] = curb2
                    wait[rid] = t - req_by_id[rid]["release_tick"]
                    pref = req_by_id[rid]["pickup_pref"]
                    total += WALK_PENALTY * abs(curb2 - pref)
                else:
                    # Charge the FULL realized ride time (not just any
                    # excess over the direct distance): this guarantees
                    # ride >= direct distance always (the physics can only
                    # add ticks via detour/queueing, never subtract), which
                    # is exactly what makes `oracle_O` (sum of direct
                    # distances) a genuine, never-beatable lower bound.
                    ride = t - pickup_tick[rid]
                    total += ride + wait[rid]
                    delivered.add(rid)

    for r in requests:
        if r["id"] not in delivered:
            total += ABANDON_PENALTY

    if total != total or total < 0:
        return False, 0.0
    return True, total


def naive_policy(view):
    """Single-vehicle-serial baseline used ONLY to compute the evaluator's
    own reference W -- never sees a candidate, purely internal."""
    pending = view["pending"]
    if not pending:
        return {}
    v0 = next((v for v in view["vehicles"] if v["id"] == 0 and v["status"] == "idle"), None)
    if not v0:
        return {}
    r = min(pending, key=lambda r: (r["release_tick"], r["id"]))
    return {"0": [{"action": "pickup", "request": r["id"], "curb": r["pickup_pref"]},
                  {"action": "dropoff", "request": r["id"]}]}


def baseline_W(inst):
    ok, total = run_episode(inst, naive_policy)
    return total if ok else float(len(inst["hidden"]["requests"])) * ABANDON_PENALTY


def oracle_O(inst):
    return sum(abs(r["dropoff"] - r["pickup_pref"]) for r in inst["hidden"]["requests"])


def score(inst, cand_path):
    def decide(view):
        ans, st = isorun.run_candidate(cand_path, view, timeout=8)
        if st != "OK":
            return "__FAIL__"
        if ans is None:
            return {}
        if not isinstance(ans, dict):
            return "__FAIL__"
        a = ans.get("assign", {})
        if a is None:
            a = {}
        if not isinstance(a, dict):
            return "__FAIL__"
        return a
    return run_episode(inst, decide)


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        try:
            ok, obj = score(inst, cand)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        W = baseline_W(inst)
        O = oracle_O(inst)
        if W <= O + 1e-9:
            vec.append(0.0)
            continue
        r = (W - obj) / (W - O)
        r = max(0.0, min(1.0, r))
        vec.append(r if (r == r) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
