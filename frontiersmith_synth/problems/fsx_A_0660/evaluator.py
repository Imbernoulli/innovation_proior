#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0660 -- "Monsoon Cascade: Three-Dam Flood Policy"
(family: cascade-gate-flood-policy; format B, quality-metric).

THEME.  Monsoon season on a three-dam river.  Dam 1 (upstream) feeds Dam 2
(middle) feeds Dam 3 (downstream); below Dam 3 sits the town.  Released water
does not arrive instantly -- it travels: water Dam 1 releases at tick t shows
up as inflow to Dam 2 at tick t+delay1; water Dam 2 releases shows up at Dam 3
at tick t+delay2 (RESERVOIR ROUTING).  Each dam also has its own local
catchment inflow.  The operator is handed a full monsoon-event forecast (all
three catchments' inflow hydrographs for the whole horizon) and must commit,
in advance, to a release schedule for all three gates (MULTI-DAM
COORDINATION).  A dam that is not drawn down before its water arrives is
forced into an uncontrolled emergency spill the instant it hits capacity --
and if that spill happens to land when the town's own local inflow is also
peaking, the result is a superflood.  The only way to avoid this is
ANTICIPATORY DRAWDOWN: start releasing from a dam *before* its inflow peaks
(a locally "wasteful" release, since the dam looks far from full), so there is
absorption headroom in place when the delayed, routed pulse actually arrives.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object -- the PUBLIC instance:
    {"name": str, "t_steps": T (int),
     "dam1": {"capacity":.., "storage0":.., "release_max":.., "inflow":[T floats]},
     "dam2": {"capacity":.., "storage0":.., "release_max":.., "inflow":[T floats]},
     "dam3": {"capacity":.., "storage0":.., "release_max":.., "inflow":[T floats]},
     "delay12": int, "delay23": int,
     "town_inflow": [T floats]}
  stdout: ONE JSON object:
    {"release1":[T floats], "release2":[T floats], "release3":[T floats]}
  where release_i[t] >= 0 is the commanded gate opening for dam i at tick t
  (physical units, e.g. million-liters/tick).  Values above that dam's
  release_max are silently clipped down by physics (a gate cannot open past
  its mechanical limit); NEGATIVE values, non-finite values, wrong types, or
  wrong lengths make the WHOLE instance score 0.0.

SIMULATION (deterministic, run by THIS evaluator -- never trust the candidate).
  For t = 0..T-1, for each dam i (1,2,3) in order:
    avail_i(t)   = storage_i(t) + own_inflow_i(t) + routed_in_i(t)
                   (routed_in_1 = 0; routed_in_2(t) = actual_release1(t-delay12)
                    if t>=delay12 else 0; routed_in_3(t) = actual_release2(t-delay23)
                    if t>=delay23 else 0)
    requested    = clip(candidate release_i[t], 0, release_max_i)
    actual_release_i(t) = min(requested, avail_i(t))            # can't release water you don't have
    leftover     = avail_i(t) - actual_release_i(t)
    if leftover > capacity_i: emergency_spill = leftover - capacity_i
         actual_release_i(t) += emergency_spill                 # UNCONTROLLED forced spill
         storage_i(t+1) = capacity_i
    else: storage_i(t+1) = leftover
  town_flow(t) = actual_release_3(t) + town_inflow(t)
  OBJECTIVE (per instance): minimize peak_town = max_t town_flow(t).

SCORING (deterministic affine anchor, minimization; small candidate scores near
0.1, an (unreachable, loose) ideal scores near 1.0):
    q_base = peak_town under an internal REACTIVE-THRESHOLD baseline policy
             (each dam reacts only to ITS OWN current storage fraction --
              no forecast, no coordination -- the "obvious" first approach)
    q_lb   = a provable, instance-only LOWER BOUND on any policy's peak_town:
              max( max_t town_inflow(t),
                   (storage0_3 + sum(dam3.inflow) - capacity_3) / T )
             (town_flow(t) >= town_inflow(t) always since release>=0; and by
              storage conservation at dam 3, cumulative release3 over the
              horizon is at least storage0_3 + sum(own inflow) - capacity_3,
              even before counting anything routed in from dam 2, so the
              average -- hence the peak -- can never beat this)
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )

ISOLATION.  The candidate runs in a FRESH SUBPROCESS via isorun.run_candidate;
it sees only the public instance.  q_base and q_lb are computed by THIS parent
process from the full (public) instance data, never from candidate internals,
so a frame-walking / introspecting candidate learns nothing useful.

CLI: python3 evaluator.py <solution.py>
Prints "Ratio: <mean r>" and "Vector: [r_1, ..., r_n]" on their own last lines.
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u = (state >> 11) / float(1 << 53)          # in [0,1)
        return lo + u * (hi - lo)

    return nxt


def _pulse(t, peak, width, amp, base):
    """Triangular pulse: base flow plus a ramp-up/ramp-down bump around `peak`."""
    d = abs(t - peak)
    if d >= width:
        return base
    return base + amp * (1.0 - d / width)


# ----------------------------- instance family ------------------------------
def _build_hydro(seed, T, peak, width, amp, base):
    nx = _rng(seed)
    jt = nx(-1.0, 1.0)           # small deterministic jitter to peak timing
    jw = 0.9 + 0.2 * nx(0.0, 1.0)
    out = []
    for t in range(T):
        v = _pulse(t, peak + jt, width * jw, amp, base)
        v += 0.15 * amp * math.sin(0.7 * t + seed % 7) * 0.15   # mild deterministic ripple
        out.append(max(0.0, round(v, 4)))
    return out


def _make_instance(spec):
    (seed, T, cap1, cap2, cap3, s01, s02, s03, r1, r2, r3, d12, d23,
     pt1, w1, a1, b1, pt2, w2, a2, b2, pt3, w3, a3, b3, ptt, wt, at, bt) = spec
    inflow1 = _build_hydro(seed + 1, T, pt1, w1, a1, b1)
    inflow2 = _build_hydro(seed + 2, T, pt2, w2, a2, b2)
    inflow3 = _build_hydro(seed + 3, T, pt3, w3, a3, b3)
    town = _build_hydro(seed + 4, T, ptt, wt, at, bt)
    return {
        "name": f"cascade{seed}", "t_steps": T,
        "dam1": {"capacity": cap1, "storage0": s01, "release_max": r1, "inflow": inflow1},
        "dam2": {"capacity": cap2, "storage0": s02, "release_max": r2, "inflow": inflow2},
        "dam3": {"capacity": cap3, "storage0": s03, "release_max": r3, "inflow": inflow3},
        "delay12": d12, "delay23": d23, "town_inflow": town,
    }


def _build_instances():
    # (seed, T, cap1,cap2,cap3, s01,s02,s03, r1,r2,r3, d12,d23,
    #  pt1,w1,a1,b1,  pt2,w2,a2,b2,  pt3,w3,a3,b3,  ptt,wt,at,bt)
    specs = [
        # --- TRAP cases: upstream pulse routed through both delays lands
        #     exactly where dam3's own inflow / the town's local inflow ALSO
        #     peaks -> a purely reactive per-dam controller synchronizes all
        #     three emergency spills into one superflood.
        (901, 40, 260, 220, 170, 130, 110, 85, 26, 22, 18, 5, 4,
         12, 6, 24, 4,  0, 6, 4, 3,   21, 5, 16, 3,   21, 5, 12, 3),   # trap: pt3==ptt==pt1+d12+d23
        (902, 42, 300, 230, 180, 150, 115, 90, 28, 22, 17, 6, 5,
         10, 7, 30, 5,  0, 6, 5, 3,   21, 5, 18, 3,   21, 5, 14, 3),   # trap
        (903, 38, 240, 200, 160, 120, 100, 78, 24, 20, 16, 4, 5,
         13, 5, 22, 4,  0, 5, 4, 3,   22, 5, 15, 3,   22, 6, 13, 3),   # trap
        (904, 44, 320, 250, 190, 160, 125, 95, 30, 24, 19, 5, 6,
         11, 6, 28, 5,  0, 6, 4, 3,   22, 6, 17, 3,   22, 6, 15, 3),   # trap, larger/held-out
        # --- calmer / staggered cases: pulses spread out, greedy is only
        #     moderately suboptimal, more forgiving for a competent heuristic
        (905, 40, 260, 220, 170, 110, 95, 70, 26, 22, 18, 5, 4,
         8, 6, 18, 4,   18, 6, 10, 3,  30, 6, 8, 3,    5, 5, 6, 3),
        (906, 36, 230, 190, 150, 95, 80, 62, 22, 19, 15, 4, 3,
         6, 5, 14, 4,   14, 5, 8, 3,   26, 5, 6, 3,    33, 5, 9, 3),
        (907, 40, 270, 210, 165, 120, 95, 72, 25, 20, 16, 5, 5,
         9, 6, 20, 4,   0, 5, 3, 3,    28, 6, 6, 3,    16, 6, 8, 3),
        # --- single dominant catchment (no upstream pulse at all): tests
        #     that a policy isn't ONLY tuned to the delay-routing trap
        (908, 34, 220, 190, 150, 100, 90, 70, 20, 18, 15, 4, 4,
         30, 5, 2, 3,   30, 5, 2, 3,   14, 6, 22, 4,   15, 6, 18, 3),
        # --- harder held-out: bigger horizon, tighter gates relative to volume
        (909, 48, 340, 270, 200, 190, 150, 110, 24, 20, 16, 6, 6,
         14, 7, 30, 5,  0, 7, 5, 3,    26, 6, 20, 3,   26, 6, 16, 3),   # trap, tight gates
        (910, 46, 300, 240, 185, 150, 120, 90, 27, 22, 17, 5, 5,
         16, 7, 16, 4,  22, 6, 12, 3,  0, 5, 5, 3,     36, 6, 10, 3),
    ]
    return [_make_instance(s) for s in specs]


# ----------------------------- simulation core ------------------------------
def _simulate(inst, rel1, rel2, rel3):
    """Run the true cascade physics. rel_i are length-T lists (already
    type/shape validated by the caller). Returns (town_flow list, peak)."""
    T = inst["t_steps"]
    d1, d2, d3 = inst["dam1"], inst["dam2"], inst["dam3"]
    delay12, delay23 = inst["delay12"], inst["delay23"]
    town_in = inst["town_inflow"]

    s1, s2, s3 = d1["storage0"], d2["storage0"], d3["storage0"]
    cap1, cap2, cap3 = d1["capacity"], d2["capacity"], d3["capacity"]
    rmax1, rmax2, rmax3 = d1["release_max"], d2["release_max"], d3["release_max"]

    actual1 = [0.0] * T
    actual2 = [0.0] * T
    actual3 = [0.0] * T
    town_flow = [0.0] * T

    for t in range(T):
        routed2 = actual1[t - delay12] if t - delay12 >= 0 else 0.0
        avail1 = s1 + d1["inflow"][t]
        req1 = min(max(rel1[t], 0.0), rmax1)
        act1 = min(req1, avail1)
        leftover1 = avail1 - act1
        if leftover1 > cap1:
            act1 += leftover1 - cap1
            s1 = cap1
        else:
            s1 = leftover1
        actual1[t] = act1

        avail2 = s2 + d2["inflow"][t] + routed2
        req2 = min(max(rel2[t], 0.0), rmax2)
        act2 = min(req2, avail2)
        leftover2 = avail2 - act2
        if leftover2 > cap2:
            act2 += leftover2 - cap2
            s2 = cap2
        else:
            s2 = leftover2
        actual2[t] = act2

        routed3 = actual2[t - delay23] if t - delay23 >= 0 else 0.0
        avail3 = s3 + d3["inflow"][t] + routed3
        req3 = min(max(rel3[t], 0.0), rmax3)
        act3 = min(req3, avail3)
        leftover3 = avail3 - act3
        if leftover3 > cap3:
            act3 += leftover3 - cap3
            s3 = cap3
        else:
            s3 = leftover3
        actual3[t] = act3

        town_flow[t] = act3 + town_in[t]

    return town_flow, max(town_flow)


def _reactive_baseline(inst):
    """Internal weak reference: each dam reacts ONLY to its own current
    storage fraction, no forecast, no coordination between dams."""
    T = inst["t_steps"]

    def per_dam_release(cap, rmax, storage_frac):
        if storage_frac > 0.80:
            return rmax
        elif storage_frac > 0.55:
            return 0.35 * rmax
        return 0.0

    d1, d2, d3 = inst["dam1"], inst["dam2"], inst["dam3"]
    delay12, delay23 = inst["delay12"], inst["delay23"]
    s1, s2, s3 = d1["storage0"], d2["storage0"], d3["storage0"]
    cap1, cap2, cap3 = d1["capacity"], d2["capacity"], d3["capacity"]
    rmax1, rmax2, rmax3 = d1["release_max"], d2["release_max"], d3["release_max"]

    actual1 = [0.0] * T
    actual2 = [0.0] * T
    actual3 = [0.0] * T
    town_flow = [0.0] * T

    for t in range(T):
        routed2 = actual1[t - delay12] if t - delay12 >= 0 else 0.0
        avail1 = s1 + d1["inflow"][t]
        req1 = per_dam_release(cap1, rmax1, s1 / cap1)
        act1 = min(req1, avail1)
        leftover1 = avail1 - act1
        if leftover1 > cap1:
            act1 += leftover1 - cap1; s1 = cap1
        else:
            s1 = leftover1
        actual1[t] = act1

        avail2 = s2 + d2["inflow"][t] + routed2
        req2 = per_dam_release(cap2, rmax2, s2 / cap2)
        act2 = min(req2, avail2)
        leftover2 = avail2 - act2
        if leftover2 > cap2:
            act2 += leftover2 - cap2; s2 = cap2
        else:
            s2 = leftover2
        actual2[t] = act2

        routed3 = actual2[t - delay23] if t - delay23 >= 0 else 0.0
        avail3 = s3 + d3["inflow"][t] + routed3
        req3 = per_dam_release(cap3, rmax3, s3 / cap3)
        act3 = min(req3, avail3)
        leftover3 = avail3 - act3
        if leftover3 > cap3:
            act3 += leftover3 - cap3; s3 = cap3
        else:
            s3 = leftover3
        actual3[t] = act3

        town_flow[t] = act3 + town_in_val(inst, t)

    return max(town_flow)


def town_in_val(inst, t):
    return inst["town_inflow"][t]


def _lower_bound(inst):
    d3 = inst["dam3"]
    T = inst["t_steps"]
    total_in3 = sum(d3["inflow"])
    avg_floor = max(0.0, (d3["storage0"] + total_in3 - d3["capacity"]) / T)
    peak_town_in = max(inst["town_inflow"])
    return max(avg_floor, peak_town_in)


# ----------------------------- answer validation -----------------------------
def _extract_releases(inst, answer):
    T = inst["t_steps"]
    if not isinstance(answer, dict):
        return None
    out = []
    for key in ("release1", "release2", "release3"):
        arr = answer.get(key)
        if not isinstance(arr, list) or len(arr) != T:
            return None
        clean = []
        for v in arr:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            fv = float(v)
            if not (fv == fv) or fv in (float("inf"), float("-inf")):
                return None
            if fv < 0.0:
                return None
            clean.append(fv)
        out.append(clean)
    return out


def score(inst, answer):
    rels = _extract_releases(inst, answer)
    if rels is None:
        return False, None
    rel1, rel2, rel3 = rels
    _, peak = _simulate(inst, rel1, rel2, rel3)
    if not (peak == peak) or peak in (float("inf"), float("-inf")):
        return False, None
    return True, peak


def baseline(inst):
    return _reactive_baseline(inst)


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        q_base = baseline(inst)
        q_lb = _lower_bound(inst)
        denom = q_base - q_lb
        if denom < 1e-9:
            denom = 1e-9

        public = {
            "name": inst["name"], "t_steps": inst["t_steps"],
            "dam1": dict(inst["dam1"]), "dam2": dict(inst["dam2"]),
            "dam3": dict(inst["dam3"]),
            "delay12": inst["delay12"], "delay23": inst["delay23"],
            "town_inflow": list(inst["town_inflow"]),
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, q_cand = score(inst, ans)
        except Exception:
            ok, q_cand = False, None
        if not ok:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (q_base - q_cand) / denom
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
