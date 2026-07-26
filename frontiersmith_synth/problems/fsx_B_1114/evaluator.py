import sys, json, random, isorun

# ==========================================================================
# fsx_B_1114 -- scan-resistant-policy-forge (Format B, isolated candidate)
# Theme: "doorman of a club raided by tour buses". Mechanisms composed:
#   (1) eviction-policy-program   -- the per-round admit/evict decision fn
#   (2) ghost-history-accounting  -- each round is a FRESH isolated subprocess
#       call, so any recency/identity bookkeeping the policy wants MUST be
#       explicitly threaded through an opaque "state" blob it emits and gets
#       back verbatim next round -- there is no implicit process memory.
#   (3) scan-resistance-design    -- periodic tour-bus floods of one-shot
#       faces must not be allowed to evict the doorman's memorized regulars.
#
# INNOVATION HOOK: the trace family's own parameters -- scan_period/span/
# phase and inversion_period/n_eras -- are DISCLOSED VERBATIM in every
# round's public payload. The winning policy does not need to *detect*
# scans or crowd-rotations reactively (the average-strong-coder move is
# plain global LRU, which reacts to raw recency only); it can compute
# directly from the round counter "is this a bus round" and "whose turn is
# it right now", and schedule its own admission/eviction regime switches
# ahead of time -- policy design as system identification, not rule
# selection.
#
# PROTOCOL. One club "night" = one instance = a fixed sequence of ROUNDS.
# The candidate program is invoked ONCE PER ROUND, isolated, no shared
# memory between calls. Each call's stdin is the PUBLIC view for that round
# only: the schedule constants, the doorman's CURRENT memorized list
# ("floor", ground truth, given fresh every round), whatever opaque "state"
# blob the candidate itself emitted last round, and this round's arrivals.
# The candidate must return one decision per arrival plus its next "state".
#
# SCORING. Per round, a "hit" is an arrival who is already on the
# memorized list (instant recognition, no decision needed). The session
# score is the TOTAL hit count over the whole night, MAXIMIZED. A trivial
# "memorize the first faces you see and never forget" policy is always
# valid and reproduces the calibrated 0.1 baseline exactly.
# ==========================================================================

STATE_MAX_CHARS = 60000


# ----------------------------- instance generator ---------------------------
def _is_scan_round(r, sp, ss, sph):
    if sp <= 0:
        return False
    return ((r - sph) % sp) < ss


def _era_of_round(r, ip, ne):
    if ne <= 1 or ip <= 0:
        return 0
    return (r // ip) % ne


def _build_instance(spec):
    seed = spec["seed"]
    rng = random.Random(seed)
    # opaque per-instance token so raw seed/name never appear verbatim in a
    # token the candidate can read.
    tagrng = random.Random(f"tag:{seed}")
    alnum = "abcdefghijklmnopqrstuvwxyz0123456789"
    tag = "".join(tagrng.choice(alnum) for _ in range(8))

    C = spec["C"]; R = spec["R"]
    sp, ss, sph = spec["scan_period"], spec["scan_span"], spec["scan_phase"]
    ip, ne = spec["inversion_period"], spec["n_eras"]
    eps, cb, sb, skew = spec["era_pool_size"], spec["core_batch"], spec["scan_batch"], spec["skew"]

    era_pools = [[f"v{tag}_{e}_{k}" for k in range(eps)] for e in range(ne)]
    era_weights = [(1.0 / ((k + 1) ** skew)) for k in range(eps)]

    rounds = []
    for r in range(R):
        if _is_scan_round(r, sp, ss, sph):
            n = sb + rng.randint(-sb // 10, sb // 10)
            keys = [f"b{tag}_{r}_{i}" for i in range(max(1, n))]
        else:
            e = _era_of_round(r, ip, ne)
            pool = era_pools[e]
            n = cb + rng.randint(-cb // 8, cb // 8)
            keys = rng.choices(pool, weights=era_weights, k=max(1, n))
        rounds.append(keys)
    return {
        "capacity": C, "scan_period": sp, "scan_span": ss, "scan_phase": sph,
        "inversion_period": ip, "n_eras": ne, "rounds": rounds, "name": spec["name"],
    }


def make_instances():
    specs = [
        dict(name="scan_trap", seed=4101, C=90, R=45, scan_period=4, scan_span=1, scan_phase=2,
             inversion_period=0, n_eras=1, era_pool_size=100, core_batch=20, scan_batch=200, skew=0.4),
        dict(name="rotation_only", seed=4102, C=70, R=48, scan_period=0, scan_span=0, scan_phase=0,
             inversion_period=4, n_eras=2, era_pool_size=120, core_batch=30, scan_batch=0, skew=0.9),
        dict(name="rush_a", seed=4103, C=110, R=54, scan_period=6, scan_span=1, scan_phase=2,
             inversion_period=3, n_eras=3, era_pool_size=90, core_batch=30, scan_batch=120, skew=0.8),
        dict(name="rush_b", seed=4104, C=100, R=54, scan_period=5, scan_span=1, scan_phase=2,
             inversion_period=3, n_eras=3, era_pool_size=90, core_batch=30, scan_batch=160, skew=0.8),
        dict(name="rush_c", seed=4105, C=80, R=54, scan_period=6, scan_span=1, scan_phase=2,
             inversion_period=3, n_eras=3, era_pool_size=70, core_batch=30, scan_batch=160, skew=0.8),
        dict(name="rush_d", seed=4106, C=110, R=54, scan_period=7, scan_span=1, scan_phase=2,
             inversion_period=3, n_eras=3, era_pool_size=70, core_batch=30, scan_batch=120, skew=0.8),
        dict(name="rush_e", seed=4107, C=100, R=54, scan_period=5, scan_span=1, scan_phase=2,
             inversion_period=3, n_eras=3, era_pool_size=70, core_batch=30, scan_batch=160, skew=0.8),
        dict(name="rush_f", seed=4108, C=110, R=54, scan_period=6, scan_span=1, scan_phase=2,
             inversion_period=3, n_eras=3, era_pool_size=70, core_batch=30, scan_batch=120, skew=0.6),
        dict(name="rush_g", seed=4109, C=110, R=54, scan_period=5, scan_span=1, scan_phase=2,
             inversion_period=3, n_eras=3, era_pool_size=90, core_batch=30, scan_batch=160, skew=0.8),
        dict(name="held_out_large", seed=4110, C=140, R=90, scan_period=7, scan_span=1, scan_phase=2,
             inversion_period=5, n_eras=3, era_pool_size=120, core_batch=35, scan_batch=180, skew=0.75),
    ]
    return [{"public": None, "hidden": _build_instance(s)} for s in specs]


def baseline(inst):
    """Evaluator-computed trivial-construction objective: memorize the first
    faces you ever see, never forget, never re-learn. Pure python, no
    candidate call. Also exactly what solutions/trivial.py implements via
    the protocol, so a faithful trivial candidate reproduces this exactly."""
    h = inst["hidden"]
    C = h["capacity"]
    floor = set()
    hits = 0
    for keys in h["rounds"]:
        for k in keys:
            if k in floor:
                hits += 1
            elif len(floor) < C:
                floor.add(k)
    return hits


# ----------------------------- per-round validation --------------------------
def _validate_round(ans, arrivals, floor_set, capacity):
    """Returns (ok, hits_this_round, new_floor_set, new_state)."""
    if not isinstance(ans, dict):
        return False, 0, floor_set, None
    decisions = ans.get("decisions")
    state = ans.get("state", None)
    if not isinstance(decisions, list) or len(decisions) != len(arrivals):
        return False, 0, floor_set, None
    try:
        if state is not None and len(json.dumps(state)) > STATE_MAX_CHARS:
            return False, 0, floor_set, None
    except (TypeError, ValueError):
        return False, 0, floor_set, None

    floor = set(floor_set)
    hits = 0
    for key, d in zip(arrivals, decisions):
        if not isinstance(key, str):
            return False, 0, floor_set, None
        if key in floor:
            hits += 1
            continue
        if not isinstance(d, dict):
            return False, 0, floor_set, None
        action = d.get("action")
        evict = d.get("evict", None)
        if action == "skip":
            if evict is not None:
                return False, 0, floor_set, None
            continue
        elif action == "admit":
            if len(floor) < capacity:
                if evict is not None:
                    return False, 0, floor_set, None
                floor.add(key)
            else:
                if not isinstance(evict, str) or evict not in floor:
                    return False, 0, floor_set, None
                floor.discard(evict)
                floor.add(key)
        else:
            return False, 0, floor_set, None
    return True, hits, floor, state


def score(inst, cand):
    """Runs the full multi-round night for one instance against the
    candidate program (a fresh isolated subprocess call per round).
    Returns (ok, total_hits)."""
    h = inst["hidden"]
    C = h["capacity"]
    floor = set()
    state = None
    total_hits = 0
    R = len(h["rounds"])
    for r, arrivals in enumerate(h["rounds"]):
        public = {
            "round": r, "total_rounds": R, "capacity": C,
            "scan_period": h["scan_period"], "scan_span": h["scan_span"],
            "scan_phase": h["scan_phase"],
            "inversion_period": h["inversion_period"], "n_eras": h["n_eras"],
            "floor": sorted(floor), "state": state, "arrivals": list(arrivals),
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            return False, 0.0
        try:
            ok, hits, floor, state = _validate_round(ans, arrivals, floor, C)
        except Exception:
            ok = False
        if not ok:
            return False, 0.0
        total_hits += hits
    if total_hits <= 0:
        return False, 0.0
    return True, float(total_hits)


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
        b = baseline(inst)
        r = min(1.0, 0.1 * obj / max(b, 1e-9))
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
