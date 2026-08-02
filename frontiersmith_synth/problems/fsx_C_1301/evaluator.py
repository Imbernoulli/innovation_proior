#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1301 -- "Charge the Fleet, Not the Van: Slot Allocation
for Electric Delivery Vans" (family: fleet-charging-policy; format B, quality-metric).

THEME.  A depot operates a fleet of electric delivery vans.  Each van has a fixed
multi-stop route with a per-leg energy cost that depends on terrain (flat / hilly /
mountain legs cost different amounts per unit of travel time -- "route-energy-
uncertainty": consumption is NOT uniform along the route, so a van's charging need
must be computed from the actual per-leg energy draw, not guessed from a flat rate).
Each route passes exactly two charge-capable stops: an EARLY one (p1) and a LATE,
usually-closer-to-the-popular-hub one (p2).  Charging at either stop, brought to
full, is individually always enough energy-wise to finish the route ("state-of-
charge-planning": the candidate must compute the correct top-up amount from the
input's per-leg energies -- guessing low stresses/strands the van later, guessing
high just wastes nothing since amounts are clamped to headroom).  Every charger has
a small number of parallel plugs ("charger-queue-contention"): many vans' LATE stop
sits on the SAME popular charger, so if every van independently waits until its own
late stop to top up (the obvious, individually-rational policy), their charging
windows cluster and they queue -- charging still succeeds physically, but the wait
eats the schedule's slack and later deliveries land after their deadlines.  The
INSIGHT (innovation hook): a fleet-level allocator can have SOME vans charge at
their EARLY stop instead -- before they strictly need to -- spreading demand across
two separate charger resources and two different points in time, which no
per-van-only planner considers because in isolation the early stop looks
unnecessary.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object -- the full public instance (see statement.md):
            {"name": str, "chargers": [...], "vans": [...]}
  stdout: ONE JSON object:
            {"vans": [{"id": <int>, "charge_at_p1": <number>=0>,
                       "charge_at_p2": <number>=0>}, ...]}   # one entry per van id
          Amounts are energy units requested at that van's early/late charge stop;
          they are clamped to actual battery headroom at arrival (no penalty for
          asking for more than needed). Missing/duplicate/extra van ids, wrong
          types, negative or non-finite amounts -> instance scores 0.0.

SCORING (deterministic; no wall-time).  For each instance we run the SAME
discrete-event fleet simulator (this file, in the parent process) on:
  frac_base = NEVER charging at all, computed by THIS evaluator -- a van reliably
              strands partway through its route (capacity is sized so no single
              leg or half-route alone exceeds it, but the whole route does), so
              this is a genuinely weak, feasibility-blind floor. This is exactly
              what solutions/trivial.py also submits.
  frac_cand = the candidate's submitted plan, run through the identical simulator
              (so real queue contention from ALL vans sharing chargers applies).
  frac_ub   = 1.0 (every stop of every van delivered on time) -- a loose, usually
              unreachable ceiling once contention is severe, so there is headroom.
Normalized (never-charge floor -> 0.1, ideal -> 1.0; the denominator is floored so
a near-perfect floor on a low-contention instance can't blow a small deviation up
into an extreme ratio):
    r = clamp( 0.1 + 0.9 * (frac_cand - frac_base) / max(0.12, 1.0 - frac_base), 0, 1)
A stop counts as delivered "on time" iff the van reaches it (never stranded: SoC
would have to go negative on some earlier leg) at or before its deadline; charging
dwell time delays only the STOPS AFTER the charge, never the stop being charged at
(delivery happens on arrival, before any charging). Being stranded (insufficient
charge requested anywhere) forfeits that stop and every later stop on the route.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the public instance. All references
(baseline, simulator, deadlines) live and run in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math, random
import isorun


# ----------------------------- instance construction ------------------------
DWELL = 0            # no fixed per-stop service time (keeps the arithmetic simple)
CAP_MARGIN = 1.08     # capacity margin over the larger single-charge-mode need
SLACK_FRAC = 0.22
SLACK_MIN = 8
TERRAIN_MULTS = [100, 100, 140, 140, 190]  # weighted: mostly flat/hilly, some mountain


def _cum(legs):
    """Cumulative (time, energy) at stop 0..L assuming NO charging anywhere."""
    t = 0; e = 0
    times = [0]; energies = [0]
    for leg in legs:
        t += leg["time"] + DWELL
        e += leg["energy"]
        times.append(t); energies.append(e)
    return times, energies


def _build_van(rng, vid, L, chargers_by_role):
    legs = []
    for _ in range(L):
        tt = rng.randint(8, 14)
        mult = rng.choice(TERRAIN_MULTS)
        en = max(1, math.ceil(tt * mult / 100.0))
        legs.append({"time": tt, "energy": en})

    p1 = max(1, round(L * 0.35))
    p2 = max(p1 + 1, round(L * 0.7))
    if p2 >= L:
        p2 = L - 1
    if p1 >= p2:
        p1 = p2 - 1

    p1_charger = chargers_by_role["early"]
    p2_charger = chargers_by_role["popular"] if rng.random() < chargers_by_role["popular_pct"] \
        else chargers_by_role["secondary"]

    times, energies = _cum(legs)
    L_energy = energies[-1]
    seg1 = energies[p1]
    seg1_rest = L_energy - energies[p1]
    seg2 = energies[p2]
    seg2_rest = L_energy - energies[p2]
    cap = math.ceil(CAP_MARGIN * max(seg1, seg1_rest, seg2, seg2_rest, 1))

    # JIT reference plan (charge to full at p2 only, as an isolated van would) ->
    # used purely to derive deadlines with a realistic amount of slack.
    rate_p2 = chargers_by_role["rates"][p2_charger]
    t = 0
    deadlines = []
    for i in range(1, L + 1):
        leg = legs[i - 1]
        t_arr = t + leg["time"] + DWELL
        deadlines.append(t_arr + max(SLACK_MIN, round(SLACK_FRAC * t_arr)))
        if i == p2:
            amt = seg2
            dur = math.ceil(amt / rate_p2) if amt > 0 else 0
            t = t_arr + dur
        else:
            t = t_arr

    return {
        "id": vid, "legs": legs, "p1": p1, "p2": p2,
        "p1_charger": p1_charger, "p2_charger": p2_charger,
        "capacity": cap, "deadlines": deadlines,
        "need_p1": seg1, "need_p2": seg2,
    }


def _build_instance(name, seed, V, L, popular_slots, secondary_slots, early_slots, popular_pct):
    rng = random.Random(seed)
    chargers = [
        {"id": 0, "slots": popular_slots, "rate": 6},
        {"id": 1, "slots": secondary_slots, "rate": 6},
        {"id": 2, "slots": early_slots, "rate": 5},
    ]
    roles = {"popular": 0, "secondary": 1, "early": 2, "popular_pct": popular_pct,
             "rates": {0: 6, 1: 6, 2: 5}}
    vans = [_build_van(rng, vid, L, roles) for vid in range(V)]
    return {"name": name, "chargers": chargers, "vans": vans}


def _build_instances():
    specs = [
        # (name, seed, V, L, popular_slots, secondary_slots, early_slots, popular_pct)
        ("fleet3_easy",     1001, 3,  6, 2, 2, 2, 0.70),
        ("fleet5_easy",     1002, 5,  6, 2, 2, 2, 0.70),
        ("fleet8_mild",     1003, 8,  6, 2, 2, 2, 0.75),
        ("fleet12_build",   1004, 12, 6, 2, 2, 2, 0.80),
        ("fleet18_trap",    1005, 18, 7, 2, 1, 2, 0.80),
        ("fleet24_trap",    1006, 24, 7, 2, 1, 2, 0.85),
        ("fleet30_trap",    1007, 30, 7, 2, 1, 3, 0.85),
        # harder / held-out
        ("fleet30_trap_h1", 2101, 30, 8, 1, 2, 3, 0.90),
        ("fleet26_trap_h2", 2102, 26, 6, 2, 2, 2, 0.90),
        ("fleet34_trap_h3", 2103, 34, 7, 2, 2, 3, 0.80),
    ]
    return [_build_instance(*s) for s in specs]


# ----------------------------- simulator ------------------------------------
def _simulate(inst, plan):
    """plan: dict vid -> (amt_p1, amt_p2), amounts already validated finite >=0.
    Returns fraction of (van, stop) deliveries made on time."""
    import heapq
    vans = inst["vans"]
    chargers = {c["id"]: c for c in inst["chargers"]}
    plug_free = {c["id"]: [0.0] * c["slots"] for c in inst["chargers"]}

    heap = [(0.0, van["id"], 0, float(van["capacity"])) for van in vans]
    heapq.heapify(heap)
    by_id = {van["id"]: van for van in vans}
    stranded = set()
    on_time_count = 0
    total_stops = sum(len(van["legs"]) for van in vans)

    while heap:
        t, vid, i, soc = heapq.heappop(heap)
        if vid in stranded:
            continue
        van = by_id[vid]
        L = len(van["legs"])
        if i >= L:
            continue
        leg = van["legs"][i]
        arr_time = t + leg["time"] + DWELL
        arr_soc = soc - leg["energy"]
        stop_idx = i + 1  # 1-based
        if arr_soc < -1e-9:
            stranded.add(vid)
            continue

        amt = 0.0
        charger = None
        if stop_idx == van["p1"]:
            amt = plan[vid][0]
            charger = chargers[van["p1_charger"]]
        elif stop_idx == van["p2"]:
            amt = plan[vid][1]
            charger = chargers[van["p2_charger"]]

        headroom = max(0.0, van["capacity"] - arr_soc)
        amt = min(amt, headroom)

        if charger is not None and amt > 1e-12:
            pf = plug_free[charger["id"]]
            idx_min = min(range(len(pf)), key=lambda k: pf[k])
            start = max(arr_time, pf[idx_min])
            dur = math.ceil(amt / charger["rate"])
            finish = start + dur
            pf[idx_min] = finish
            soc_after = arr_soc + amt
            depart_time = finish
        else:
            soc_after = arr_soc
            depart_time = arr_time

        if arr_time <= van["deadlines"][i] + 1e-9:
            on_time_count += 1

        if i + 1 < L:
            heapq.heappush(heap, (depart_time, vid, i + 1, soc_after))

    return on_time_count / total_stops if total_stops else 0.0


def baseline(inst):
    """Weak reference construction the evaluator computes itself: NEVER charge
    at all. Every van's route needs at least one recharge to finish (capacity is
    sized so no single leg nor either half-route alone exceeds it, but the full
    route does), so a never-charging van reliably strands partway through and
    forfeits every stop from that point on. This anchors the score's floor."""
    plan = {van["id"]: (0.0, 0.0) for van in inst["vans"]}
    return _simulate(inst, plan)


# ----------------------------- validation ------------------------------------
def _validate(inst, answer):
    if not isinstance(answer, dict):
        return None
    lst = answer.get("vans")
    if not isinstance(lst, list):
        return None
    need_ids = set(van["id"] for van in inst["vans"])
    plan = {}
    seen = set()
    for item in lst:
        if not isinstance(item, dict):
            return None
        vid = item.get("id")
        if isinstance(vid, bool) or not isinstance(vid, int):
            return None
        if vid not in need_ids or vid in seen:
            return None
        seen.add(vid)
        a1 = item.get("charge_at_p1")
        a2 = item.get("charge_at_p2")
        vals = []
        for a in (a1, a2):
            if isinstance(a, bool) or not isinstance(a, (int, float)):
                return None
            fa = float(a)
            if fa != fa or fa in (float("inf"), float("-inf")):
                return None
            if fa < 0:
                return None
            vals.append(fa)
        plan[vid] = (vals[0], vals[1])
    if seen != need_ids:
        return None
    return plan


def score(inst, answer):
    plan = _validate(inst, answer)
    if plan is None:
        return False, 0.0
    frac = _simulate(inst, plan)
    return True, frac


# ----------------------------- main ------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        frac_base = baseline(inst)
        denom = max(0.12, 1.0 - frac_base)  # floor: don't let a near-perfect
        # baseline (easy, low-contention instances) blow a tiny deviation up
        # into an extreme ratio -- keeps normalization well-conditioned.
        public = {"name": inst["name"], "chargers": inst["chargers"],
                  "vans": [{"id": v["id"], "legs": v["legs"], "p1": v["p1"],
                            "p2": v["p2"], "p1_charger": v["p1_charger"],
                            "p2_charger": v["p2_charger"], "capacity": v["capacity"],
                            "deadlines": v["deadlines"]} for v in inst["vans"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, frac_cand = score(inst, ans)
        except Exception:
            ok = False; frac_cand = 0.0
        if not ok:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (frac_cand - frac_base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
