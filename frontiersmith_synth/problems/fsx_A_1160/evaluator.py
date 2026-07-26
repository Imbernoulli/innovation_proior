#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_1160 -- "Winter Freight: The Season's Spacetime Budget"
(family: ice-road-cargo-window; format B, quality-metric).

THEME.  Winter freight convoys cross a freezing lake over a 42-day season along
4 fixed corridors.  Each corridor's ice thickness evolves on its OWN clock
(a deterministic freeze/thaw response to the day's published temperature,
independent of anyone's usage).  Crossing a corridor with a heavy convoy also
FATIGUES it -- fatigue erodes the corridor's *effective* thickness and decays
on a SEPARATE, slower clock that ticks every day regardless of use.  Capacity
is therefore not a fixed property of a corridor (a "path"); it is a moving
target set by two independent, coupled clocks -- the season's freeze/thaw
curve (time-varying capacity) and self-inflicted wear (a consumable that
regenerates on its own schedule).  A convoy that is too heavy for a
corridor's CURRENT effective thickness cracks the ice: that day's cargo is
lost and the corridor takes a large structural fatigue hit (load-stress-limit
mechanism).  Which corridor you use on which day (route-schedule-coupling)
jointly determines how much the season can ever deliver.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance) -- see statement.md for the
          exact schema (name, n_days, freeze_point, mechanics{}, routes[],
          temps[]).  The whole season (all constants + the full temperature
          series) is known in advance -- a full-information planning problem;
          there is no hidden state to withhold.
  stdout: ONE JSON object: {"routes": [r_0..r_{n_days-1}], "masses": [m_0..]}
          r_d in {-1,0,1,2,3} (-1 = rest day, no crossing that day; masses on
          a rest day are ignored).  m_d: finite, non-negative cargo mass.
          At most one corridor may be used per day.

SCORING (deterministic; no wall-time).  The evaluator replays all n_days in
order: it grows each corridor's ice thickness from the published temperature
curve, computes the corridor's fatigue-eroded effective thickness, and
checks the requested mass against stress_limit*eff^2/length_factor.  A safe
crossing delivers its mass and adds proportional fatigue; an overloaded
crossing delivers 0 for that day and adds a much larger structural fatigue
penalty (the ice cracked) -- the SESSION IS NOT REJECTED, only that one
day is zeroed, matching "zero for any day a stress limit is exceeded."
Objective = total cargo delivered over the season.  We normalize with a
fixed, unreachable anchor computed by the evaluator ITSELF from the public
instance alone (never from the answer): for every day, the best safe mass
ANY single corridor could carry if it had NEVER accumulated fatigue --
summed over the season. This is unreachable (a real plan uses at most one
corridor/day and always pays real fatigue), leaving headroom above 1.0.
    r = clamp(0.1 + 0.9 * total_cargo / anchor, 0, 1)
Delivering nothing scores exactly 0.1.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the PUBLIC instance (which happens
to be the FULL instance here -- a full-information game, nothing to hide;
isolation is about the judge process/filesystem, not withholding data).

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun

MASK = (1 << 64) - 1
EPS = 1e-6

FREEZE_POINT = 0.0
STRESS_LIMIT = 0.42
FATIGUE_GAIN_K = 0.22
CRACK_PENALTY_K = 0.30
FATIGUE_DECAY = 0.70
N_ROUTES = 4
N_DAYS = 42


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = [seed & MASK]

    def nxt():
        state[0] = (state[0] * 6364136223846793005 + 1442695040888963407) & MASK
        return (state[0] >> 11) / float(1 << 53)

    return nxt


def _ffloat(nxt, lo, hi):
    return lo + nxt() * (hi - lo)


# ----------------------------- instance family ------------------------------
def _make_routes(nxt):
    routes = []
    for i in range(N_ROUTES):
        lf = round(1.0 + 0.55 * i + _ffloat(nxt, -0.08, 0.08), 3)
        gr = round(0.50 + 0.14 * i + _ffloat(nxt, -0.03, 0.03), 3)
        tr = round(0.95 - 0.20 * i + _ffloat(nxt, -0.04, 0.04), 3)
        h0 = round(4.5 + 1.0 * i + _ffloat(nxt, -0.3, 0.3), 2)
        routes.append({"length_factor": lf, "growth_rate": gr, "thaw_rate": tr, "h0": h0})
    return routes


def _make_temps(nxt, kind):
    temps = []
    D = N_DAYS
    if kind == "trap":
        # warm/mild opening (tempts heavy early use of the shortest corridor),
        # then a deep, sustained late-season freeze that thickens the FAR
        # corridors dramatically -- exactly when a corridor hammered all
        # season is pinned near its own fatigue ceiling.
        for d in range(D):
            frac = d / (D - 1)
            if frac < 0.22:
                base = _ffloat(nxt, -3, 2)
            elif frac < 0.5:
                base = _ffloat(nxt, -8, -2)
            else:
                base = _ffloat(nxt, -16, -8)
            temps.append(round(base, 2))
    elif kind == "mild":
        # marginal winter throughout -- no dramatic deep-freeze window, tests
        # that the trap still costs something even without an extreme swing.
        for d in range(D):
            temps.append(round(_ffloat(nxt, -5, 1), 2))
    elif kind == "reverse":
        # cold FRONT-loaded, mild finish -- the opposite of the "classic"
        # trap shape, so a solver that hardcodes "deep freeze comes late"
        # instead of reading the data would misjudge this one.
        for d in range(D):
            frac = d / (D - 1)
            if frac < 0.3:
                base = _ffloat(nxt, -16, -9)
            elif frac < 0.6:
                base = _ffloat(nxt, -6, 0)
            else:
                base = _ffloat(nxt, -2, 4)
            temps.append(round(base, 2))
    else:  # "control": smooth single-peak winter
        for d in range(D):
            frac = d / (D - 1)
            base = -10 * math.sin(math.pi * frac * 0.9) + _ffloat(nxt, -3, 3) - 1
            temps.append(round(base, 2))
    return temps


def _make_instance(seed, kind, name):
    nxt = _rng(seed)
    routes = _make_routes(nxt)
    temps = _make_temps(nxt, kind)
    return {
        "name": name, "n_days": N_DAYS, "freeze_point": FREEZE_POINT,
        "mechanics": {"stress_limit": STRESS_LIMIT, "fatigue_gain_k": FATIGUE_GAIN_K,
                      "crack_penalty_k": CRACK_PENALTY_K, "fatigue_decay": FATIGUE_DECAY},
        "routes": routes, "temps": temps,
    }


def _build_instances():
    # 10 seeded seasons: 5 sharp "trap" seasons (warm open + deep late freeze),
    # 2 "mild" (marginal all season), 2 "control" (smooth single-peak winter),
    # 1 "reverse" (cold front-loaded) held out for generalization.
    specs = [
        (5001, "trap"), (5002, "trap"), (5003, "trap"), (5004, "trap"), (5005, "trap"),
        (5006, "mild"), (5007, "mild"), (5008, "control"), (5009, "control"), (5010, "reverse"),
    ]
    return [_make_instance(seed, kind, f"lake{seed}") for seed, kind in specs]


# ----------------------------- physics --------------------------------------
def _thickness_curve(inst):
    """h[r][d] for d in 0..n_days, independent of any usage."""
    D = inst["n_days"]
    routes = inst["routes"]
    h = [[0.0] * (D + 1) for _ in routes]
    for r, rt in enumerate(routes):
        h[r][0] = rt["h0"]
    for d in range(D):
        T = inst["temps"][d]
        for r, rt in enumerate(routes):
            g = rt["growth_rate"] * max(0.0, inst["freeze_point"] - T) \
                - rt["thaw_rate"] * max(0.0, T - inst["freeze_point"])
            h[r][d + 1] = max(0.0, h[r][d] + g)
    return h


def _max_safe(h_eff, length_factor, stress_limit):
    return stress_limit * h_eff * h_eff / length_factor


def _obj_ref(inst, h):
    """Unreachable anchor: best safe mass ANY single corridor could carry
    that day if it had never accumulated fatigue, summed over the season.
    Computed purely from the public instance -- never from an answer."""
    m = inst["mechanics"]
    D = inst["n_days"]
    routes = inst["routes"]
    total = 0.0
    for d in range(D):
        total += max(_max_safe(h[r][d], routes[r]["length_factor"], m["stress_limit"])
                      for r in range(len(routes)))
    return total


# ----------------------------- validation + simulation -----------------------
def _validate_answer(inst, answer):
    if not isinstance(answer, dict):
        return None
    routes_ans = answer.get("routes")
    masses_ans = answer.get("masses")
    D = inst["n_days"]
    nR = len(inst["routes"])
    if not isinstance(routes_ans, list) or len(routes_ans) != D:
        return None
    if not isinstance(masses_ans, list) or len(masses_ans) != D:
        return None
    out_r, out_m = [], []
    for x in routes_ans:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        if float(x) != x or x != int(x):
            return None
        xi = int(x)
        if xi < -1 or xi >= nR:
            return None
        out_r.append(xi)
    for x in masses_ans:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        fx = float(x)
        if fx != fx or fx in (float("inf"), float("-inf")) or fx < 0.0:
            return None
        out_m.append(fx)
    return out_r, out_m


def _simulate(inst, h, routes_ans, masses_ans):
    m = inst["mechanics"]
    routes = inst["routes"]
    D = inst["n_days"]
    nR = len(routes)
    fatigue = [0.0] * nR
    total = 0.0
    for d in range(D):
        r_idx = routes_ans[d]
        mass = masses_ans[d]
        add = 0.0
        if r_idx is not None and 0 <= r_idx < nR and mass > 0.0:
            rt = routes[r_idx]
            h_d = h[r_idx][d]
            eff = max(h_d - fatigue[r_idx], 0.0)
            ms = _max_safe(eff, rt["length_factor"], m["stress_limit"])
            if mass <= ms + EPS:
                total += mass
                denom = m["stress_limit"] * max(eff, 1e-6) ** 2
                stress_frac = (mass * rt["length_factor"]) / denom
                add = m["fatigue_gain_k"] * stress_frac * h_d
            else:
                add = m["crack_penalty_k"] * h_d
        for r in range(nR):
            if r == r_idx:
                fatigue[r] = m["fatigue_decay"] * fatigue[r] + add
            else:
                fatigue[r] = m["fatigue_decay"] * fatigue[r]
    return total


def score(inst, answer):
    parsed = _validate_answer(inst, answer)
    if parsed is None:
        return False, 0.0
    routes_ans, masses_ans = parsed
    h = _thickness_curve(inst)
    total = _simulate(inst, h, routes_ans, masses_ans)
    return True, total


def baseline(inst):
    """Trivial-construction reference the evaluator computes itself: rest
    every day (0 cargo)."""
    return 0.0


# ----------------------------- scoring driver -------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        h = _thickness_curve(inst)
        anchor = _obj_ref(inst, h)
        if anchor < 1e-9:
            anchor = 1e-9
        public = {"name": inst["name"], "n_days": inst["n_days"],
                  "freeze_point": inst["freeze_point"], "mechanics": inst["mechanics"],
                  "routes": inst["routes"], "temps": inst["temps"]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, 0.0
        if not ok:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (obj / anchor)
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
