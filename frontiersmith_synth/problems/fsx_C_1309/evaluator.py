#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1309 -- "Admitting Patients When Discharge Is The Bottleneck:
Specialty-Bed Reserve Policy" (family: hospital-bed-flow-policy; format B, quality-metric).

THEME.  A small hospital has K specialty wards (each with a fixed bed CAPACITY).  Every
patient belongs to one home specialty and, on arrival, either gets a bed or joins a
BOARDING queue.  Over a fixed horizon of T days the operator submits a RESERVE schedule
per ward per day: the minimum number of that ward's beds that must stay free before it
will accept an OUTLIER patient (someone whose home ward is full, borrowed from another
specialty).  The evaluator applies this policy against a fixed, deterministic arrival
stream and scores patients admitted minus boarding cost minus diversion cost.

MECHANISM 1 (specialty-bed-substitution).  When a patient's home ward is full, they may
be placed as an OUTLIER in any OTHER ward that currently has more free beds than that
ward's reserve for the day.  This lets slack capacity absorb load from a busy ward --
but it consumes a bed that ward may need again soon for its OWN patients.

MECHANISM 2 (discharge-timing-dependency).  A patient's expected length of stay depends
on WHERE they are admitted: home-ward patients discharge at rate 1/mean_los[ward]; an
outlier patient's stay is inflated by a fixed multiplier PLUS an administrative
transfer-coordination delay (`outlier_multiplier`, `transfer_delay`), so outlier beds
free up much more slowly than home-ward beds.  A bed lent out today stays lent for a
disproportionately long time.

MECHANISM 3 (boarding-cost).  Patients who cannot be placed (home ward full, no ward
with spare-beyond-reserve capacity) board: every day spent boarding accrues
`boarding_cost_coef` per patient-day, and a constant hazard `abandon_rate = 1/max_wait`
diverts (permanently loses, at `divert_penalty` each) a fraction of the backlog daily.

INNOVATION HOOK.  Admitting an outlier the instant a bed is free (reserve = 0 always)
maximizes same-day utilization.  On light, uncorrelated arrival streams this is
harmless (each ward almost always has enough of its OWN capacity).  But when wards take
turns surging (one ward's demand ramps up shortly after another ward's surge has
borrowed its beds), a reserve=0 policy is caught with its ward full of slow-discharging
outliers exactly when its own real patients arrive -- forcing THEM into worse outlier
placements or boarding, and the inflated outlier length-of-stay (mechanism 2) means that
mistake keeps costing for days.  The insight: hold back a SMALL reserve of specialty
capacity, sized to the ward's own forecasted near-term demand net of its organic
discharge turnover -- protect only what you will need back soon, and lend freely
otherwise.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "T": int, "wards": [str,...K],
             "capacity": [K floats], "mean_los": [K floats],
             "outlier_multiplier": float, "transfer_delay": float,
             "boarding_cost_coef": float, "max_wait": float, "divert_penalty": float,
             "init_occ_home": [K floats], "arrival_forecast": [[T floats], ...K lists]}
  stdout: ONE JSON object:
            {"reserve": [[r_{w,0}, ..., r_{w,T-1}], ...K lists]}
          reserve[w][t] is the number of ward w's beds to keep free from outlier
          admission on day t.  Values are silently CLIPPED to [0, capacity[w]] (a
          reserve of 0 lends freely; >= capacity blocks all outlier admission).  The
          answer must be a dict with a "reserve" key holding exactly K lists of exactly
          T finite non-negative numbers (ints/floats; no bool/str/nan/inf).  Any
          shape/type violation, a crash, a timeout, or non-JSON output scores 0.0 for
          that instance.

DYNAMICS (per day t; evaluator-side, uses the TRUE hidden arrival realization).
  pending[w]      = board[w] + true_arrivals[w][t]
  home admit      = min(pending[w], free_home_capacity[w])              (fills own ward first)
  outlier admit    = remaining pending of ward w placed, in ward-list order, into other
                     wards j with free[j] - clip(reserve[j][t],0,cap[j]) > 0 -- filling as
                     much of ward j's spare-beyond-reserve capacity as available, then
                     moving to the next ward in order if patients still remain unplaced
  boarding_cost   += boarding_cost_coef * sum(still-unplaced patients)
  diverted        = abandon_rate * (still-unplaced patients); divert_cost += divert_penalty * diverted
  board[w]        <- unplaced - diverted   (carried to day t+1)
  discharge       : occ_home[w] -= occ_home[w]/mean_los[w]
                    occ_outlier[j] -= occ_outlier[j]/(mean_los[j]*outlier_multiplier+transfer_delay)
  objective       += (home admits + outlier admits) - boarding_cost - divert_cost   [this day]

SCORING (deterministic).  Per instance the evaluator computes, itself:
    q_base  = objective under the rigid "reserve = capacity always" policy (NEVER lends
              a bed across wards -- the naive non-strategic reference).
    q_ideal = sum of all true arrivals over the horizon (every patient admitted
              instantly, zero boarding, zero diversion -- a generous, essentially
              unreachable ceiling since simultaneous cross-ward bursts make some
              boarding/outlier-lag unavoidable even for a perfect policy).
    q_cand  = candidate's realized objective.
    r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(eps, q_ideal - q_base), 0, 1 )

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance (never the true hidden
arrival realization, q_base, or q_ideal).  All references and the true dynamics are
computed by THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math, random
import isorun


# ----------------------------- instance construction ------------------------
# Bursts / baseline load are specified as UTILIZATION targets (fraction of
# capacity a ward would settle at, at steady state, under the memoryless
# discharge model: steady_occ = arrival_rate * mean_los). This keeps every
# instance's load calibrated relative to its own capacity/mean_los instead of
# hand-picked raw rates, so "baseline util ~0.65" / "burst util ~1.5" mean the
# same thing across wards of different sizes.
def _util_curve(t, util_base, bursts):
    u = util_base
    for (day, width, peak_util) in bursts:
        if day <= t < day + width:
            u = max(u, peak_util)
    return u


def _build_instance(spec):
    rnd = random.Random(spec["seed"])
    T = spec["T"]
    wards = spec["wards"]
    K = len(wards)
    capacity = spec["capacity"]
    mean_los = spec["mean_los"]

    arrival_forecast = []
    true_arrivals = []
    init_occ_home = []
    for w in range(K):
        util_base = spec["util_base"][w]
        bursts = spec["bursts"][w]
        rate_row = [_util_curve(t, util_base, bursts) * capacity[w] / mean_los[w]
                    for t in range(T)]
        t_row = [max(0.0, rate_row[t] * (0.9 + 0.2 * rnd.random())) for t in range(T)]
        arrival_forecast.append(rate_row)
        true_arrivals.append(t_row)
        init_occ_home.append(util_base * capacity[w])

    return {
        "name": spec["name"], "T": T, "wards": list(wards),
        "capacity": list(capacity), "mean_los": list(mean_los),
        "outlier_multiplier": spec["outlier_multiplier"],
        "transfer_delay": spec["transfer_delay"],
        "boarding_cost_coef": spec["boarding_cost_coef"],
        "max_wait": spec["max_wait"],
        "divert_penalty": spec["divert_penalty"],
        "init_occ_home": init_occ_home,
        "arrival_forecast": [[round(x, 4) for x in row] for row in arrival_forecast],
        "_true_arrivals": [[round(x, 4) for x in row] for row in true_arrivals],
        "_abandon_rate": 1.0 / spec["max_wait"],
    }


def _build_instances():
    W = ["CARD", "ORTHO", "NEURO"]
    specs = [
        # 1-3: LOW-LOAD warm-ups. Capacity is comfortable relative to steady demand
        #      (baseline utilization ~0.6-0.65) and any bursts are mild -> wards
        #      rarely need each other's help, so a reserve costs little to skip.
        dict(name="calm_balanced", seed=101, T=45, wards=W,
             capacity=[12.0, 11.0, 10.0], mean_los=[5.0, 5.0, 5.0],
             outlier_multiplier=1.8, transfer_delay=2.5,
             boarding_cost_coef=0.20, max_wait=5.5, divert_penalty=2.0,
             util_base=[0.68, 0.66, 0.70],
             bursts=[[(18, 6, 1.4)], [], [(30, 5, 1.35)]]),
        dict(name="calm_mild_ripple", seed=102, T=45, wards=W,
             capacity=[13.0, 12.0, 9.0], mean_los=[5.5, 4.5, 5.0],
             outlier_multiplier=1.8, transfer_delay=2.5,
             boarding_cost_coef=0.20, max_wait=5.5, divert_penalty=2.0,
             util_base=[0.68, 0.70, 0.62],
             bursts=[[(20, 6, 1.45)], [], []]),
        dict(name="calm_two_small_bumps", seed=103, T=45, wards=W,
             capacity=[11.0, 12.0, 12.0], mean_los=[4.5, 5.0, 5.5],
             outlier_multiplier=1.8, transfer_delay=2.5,
             boarding_cost_coef=0.20, max_wait=6.0, divert_penalty=1.8,
             util_base=[0.66, 0.68, 0.66],
             bursts=[[], [(15, 5, 1.4)], [(30, 5, 1.35)]]),

        # 4-10: HIGH-LOAD trap streams. Wards take turns surging to ~1.5x their own
        #       capacity/turnover rate; a slack ward's spare beds are tempting to
        #       lend, but outlier length-of-stay (mean_los*outlier_multiplier +
        #       transfer_delay) is long enough that beds lent during one ward's
        #       burst are often still occupied when the LENDER's own burst begins.
        dict(name="trap_handoff_co", seed=204, T=45, wards=W,
             capacity=[10.0, 10.0, 10.0], mean_los=[5.0, 5.0, 5.0],
             outlier_multiplier=1.8, transfer_delay=2.5,
             boarding_cost_coef=0.20, max_wait=5.0, divert_penalty=2.0,
             util_base=[0.62, 0.62, 0.62],
             bursts=[[(4, 6, 1.6), (25, 6, 1.6)], [(11, 6, 1.6), (32, 6, 1.6)],
                     [(18, 6, 1.6), (39, 6, 1.6)]]),
        dict(name="trap_handoff_oc", seed=205, T=45, wards=W,
             capacity=[9.0, 11.0, 9.0], mean_los=[4.5, 5.5, 4.5],
             outlier_multiplier=1.8, transfer_delay=2.5,
             boarding_cost_coef=0.22, max_wait=5.0, divert_penalty=2.2,
             util_base=[0.60, 0.65, 0.60],
             bursts=[[(3, 6, 1.55), (24, 6, 1.55)], [(10, 6, 1.55), (31, 6, 1.55)],
                     [(17, 6, 1.55), (38, 6, 1.55)]]),
        dict(name="trap_two_ward_pair", seed=206, T=45, wards=W,
             capacity=[10.0, 10.0, 13.0], mean_los=[5.0, 5.0, 5.0],
             outlier_multiplier=1.8, transfer_delay=2.5,
             boarding_cost_coef=0.20, max_wait=5.0, divert_penalty=2.0,
             util_base=[0.65, 0.65, 0.55],
             bursts=[[(5, 7, 1.6), (25, 7, 1.6)], [(13, 7, 1.6), (33, 7, 1.6)], []]),
        dict(name="trap_tight_capacity", seed=207, T=45, wards=W,
             capacity=[8.0, 9.0, 8.0], mean_los=[5.0, 4.5, 5.0],
             outlier_multiplier=1.9, transfer_delay=2.7,
             boarding_cost_coef=0.25, max_wait=4.5, divert_penalty=2.3,
             util_base=[0.60, 0.60, 0.60],
             bursts=[[(3, 6, 1.55), (23, 6, 1.55)], [(9, 6, 1.55), (29, 6, 1.55)],
                     [(15, 6, 1.55), (35, 6, 1.55)]]),
        dict(name="trap_fast_cycle", seed=208, T=45, wards=W,
             capacity=[10.0, 10.0, 10.0], mean_los=[5.5, 5.5, 5.5],
             outlier_multiplier=1.8, transfer_delay=2.8,
             boarding_cost_coef=0.22, max_wait=5.0, divert_penalty=2.2,
             util_base=[0.62, 0.62, 0.62],
             bursts=[[(7, 6, 1.5), (26, 6, 1.5)], [(14, 6, 1.5), (33, 6, 1.5)],
                     [(0, 6, 1.5), (19, 6, 1.5)]]),
        # 9-10: harder held-out (longer horizon, tighter cap, three-way overlap window).
        dict(name="held_out_triple_overlap", seed=309, T=50, wards=W,
             capacity=[9.0, 9.0, 9.0], mean_los=[5.5, 5.0, 5.5],
             outlier_multiplier=1.9, transfer_delay=3.0,
             boarding_cost_coef=0.25, max_wait=4.5, divert_penalty=2.5,
             util_base=[0.62, 0.62, 0.62],
             bursts=[[(10, 8, 1.55), (34, 6, 1.5)], [(20, 9, 1.55)],
                     [(6, 6, 1.45), (28, 8, 1.55)]]),
        dict(name="held_out_asymmetric_caps", seed=310, T=50, wards=W,
             capacity=[7.0, 12.0, 9.0], mean_los=[5.0, 5.5, 4.5],
             outlier_multiplier=1.9, transfer_delay=3.0,
             boarding_cost_coef=0.25, max_wait=4.5, divert_penalty=2.5,
             util_base=[0.60, 0.65, 0.62],
             bursts=[[(8, 8, 1.55)], [(0, 6, 1.3), (26, 8, 1.55)], [(18, 8, 1.55)]]),
    ]
    return [_build_instance(s) for s in specs]


# ----------------------------- dynamics / scoring ---------------------------
def _simulate(inst, reserve):
    T = inst["T"]; wards = inst["wards"]; K = len(wards)
    cap = inst["capacity"]; mean_los = inst["mean_los"]
    outlier_mult = inst["outlier_multiplier"]; transfer_delay = inst["transfer_delay"]
    boarding_cost_coef = inst["boarding_cost_coef"]; abandon_rate = inst["_abandon_rate"]
    divert_penalty = inst["divert_penalty"]
    true_arrivals = inst["_true_arrivals"]

    occ_home = list(inst["init_occ_home"])
    occ_outlier = [0.0] * K
    board = [0.0] * K
    objective = 0.0

    for t in range(T):
        pending = [board[w] + true_arrivals[w][t] for w in range(K)]
        admit_home = [0.0] * K
        for w in range(K):
            free_w = cap[w] - occ_home[w] - occ_outlier[w]
            if free_w < 0.0: free_w = 0.0
            a = min(pending[w], free_w)
            admit_home[w] = a
            occ_home[w] += a
        remaining = [pending[w] - admit_home[w] for w in range(K)]

        outlier_total = 0.0
        for w in range(K):
            rem = remaining[w]
            if rem <= 1e-12:
                remaining[w] = 0.0
                continue
            for j in range(K):
                if j == w:
                    continue
                rv = reserve[j][t]
                if rv < 0.0: rv = 0.0
                if rv > cap[j]: rv = cap[j]
                free_j = cap[j] - occ_home[j] - occ_outlier[j]
                if free_j < 0.0: free_j = 0.0
                avail = free_j - rv
                if avail <= 0.0:
                    continue
                take = min(rem, avail)
                occ_outlier[j] += take
                rem -= take
                outlier_total += take
                if rem <= 1e-12:
                    break
            remaining[w] = max(0.0, rem)

        total_admitted_today = sum(admit_home) + outlier_total
        board_cost_today = boarding_cost_coef * sum(remaining)
        diverted_today = [remaining[w] * abandon_rate for w in range(K)]
        divert_cost_today = divert_penalty * sum(diverted_today)
        objective += total_admitted_today - board_cost_today - divert_cost_today

        for w in range(K):
            disc_h = occ_home[w] / mean_los[w]
            los_out = mean_los[w] * outlier_mult + transfer_delay
            disc_o = occ_outlier[w] / los_out
            occ_home[w] = max(0.0, occ_home[w] - disc_h)
            occ_outlier[w] = max(0.0, occ_outlier[w] - disc_o)

        for w in range(K):
            b = remaining[w] - diverted_today[w]
            board[w] = b if b > 0.0 else 0.0

    return objective


def _validate_reserve(inst, answer):
    if not isinstance(answer, dict):
        return None
    reserve = answer.get("reserve")
    K = len(inst["wards"]); T = inst["T"]
    if not isinstance(reserve, list) or len(reserve) != K:
        return None
    out = []
    for row in reserve:
        if not isinstance(row, list) or len(row) != T:
            return None
        r_out = []
        for x in row:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                return None
            xf = float(x)
            if xf != xf or xf in (float("inf"), float("-inf")) or xf < 0.0:
                return None
            r_out.append(xf)
        out.append(r_out)
    return out


def score(inst, answer):
    reserve = _validate_reserve(inst, answer)
    if reserve is None:
        return False, 0.0
    obj = _simulate(inst, reserve)
    return True, obj


def baseline(inst):
    K = len(inst["wards"]); T = inst["T"]
    rigid = [[inst["capacity"][w]] * T for w in range(K)]
    return _simulate(inst, rigid)


def ideal(inst):
    return sum(sum(row) for row in inst["_true_arrivals"])


# ----------------------------- scoring driver -------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = {k: v for k, v in inst.items() if not k.startswith("_")}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        q_ideal = ideal(inst)
        denom = q_ideal - b
        if denom < 1e-6:
            denom = 1e-6
        r = 0.1 + 0.9 * (obj - b) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
