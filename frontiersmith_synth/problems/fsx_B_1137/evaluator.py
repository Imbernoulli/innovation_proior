#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_1137 -- "One-Way Conveyor: Warm-Up Windows vs. Tool Fatigue".

Family: sequence-setup-noreturn-conveyor. Jobs are ordered ONCE onto a one-way conveyor
that threads through M fixed stations in a fixed order; because the line has no buffers
that can reorder work and a job can never travel backward to revisit a station, EVERY
station sees the jobs in exactly the SAME relative order -- the single global permutation
the candidate chooses. Each station m keeps a rolling "tooling state": a trailing memory
window of the last H_m job TYPES it has processed (sequence-dependent setup with a
lookahead/lookback horizon -- a station is progressively "warm" for a type the more
recently that type has passed through, not just on the immediately preceding job) and a
tool-fatigue counter that penalizes an UNBROKEN run of the same type once it exceeds a
per-station threshold. Because the same permutation feeds every station simultaneously,
committing to a long same-type run to keep one station's memory warm can be exactly what
silently detonates a DIFFERENT station's fatigue limit far downstream -- and once a job
has passed a station there is no way to go back and fix it. THE NOVELTY: an ordering's
cost is dominated by the downstream tooling states it irreversibly commits every station
into over the whole run, not by any single job's own attributes (e.g. its processing
time) -- so a classic shortest-processing-time (SPT) style sort, which reasons about jobs
one at a time, can commit the line to badly-timed warm/fatigue states across the M
stations it never modeled at all.

The candidate is UNTRUSTED model output: it runs in an ISOLATED subprocess via `isorun`,
sees ONLY the public instance on stdin, and returns ONLY its answer on stdout, so it can
never reach the evaluator's frames / scorer / baseline / hidden state.

Scoring (deterministic; no wall-time):
  baseline b = cost of the "as-given" identity order (order = job ids 0..n-1, the order
               jobs already arrive at the head of the line). Always feasible (it's a
               permutation by construction).
  For a FEASIBLE answer with objective obj:  r = min(1, 0.1 * b / obj)
  -> the identity order maps to exactly 0.1; a design k times cheaper than the identity
     order maps to min(1, 0.1*k). Infeasible / malformed answer -> 0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


# ----------------------------- instance family -----------------------------
def _gen_one(seed, n, m, k, horizons, fatigue_flags, thresholds, proc_mode,
             cold_lo, cold_hi, warm_lo, warm_hi, fatigue_lo, fatigue_hi):
    r = _rng(seed)
    base_time = [r(3, 30) for _ in range(k)]
    jobs = []
    for i in range(n):
        t = r(0, k - 1)
        if proc_mode == "corr":
            pt = max(1, base_time[t] + r(-2, 2))
        else:  # "trap": processing time is a genuinely independent attribute
            pt = r(1, 40)
        jobs.append({"id": i, "type": t, "proc_time": pt})
    stations = []
    for mm in range(m):
        cold = [r(cold_lo, cold_hi) for _ in range(k)]
        warm_raw = [r(warm_lo, warm_hi) for _ in range(k)]
        warm = [min(warm_raw[i], cold[i] - 1) for i in range(k)]
        stations.append({
            "id": mm, "horizon": horizons[mm], "weight": round(1.0 + 0.5 * r(0, 2), 2),
            "cold_cost": cold, "warm_cost": warm,
            "fatigue_on": fatigue_flags[mm], "fatigue_threshold": thresholds[mm],
            "fatigue_cost": [r(fatigue_lo, fatigue_hi) for _ in range(k)] if fatigue_flags[mm] else [0] * k,
        })
    return {"n": n, "m": m, "k": k, "jobs": jobs, "stations": stations}


_SPECS = [
    # seed, m, horizons, fatigue_flags, thresholds, proc_mode, n, k
    (101, 3, [3, 5, 4], [False, False, False], [0, 0, 0], "corr", 60, 4),
    (102, 4, [2, 4, 3, 6], [True, False, True, False], [3, 0, 4, 0], "trap", 60, 4),
    (103, 3, [4, 3, 5], [False, False, False], [0, 0, 0], "corr", 60, 4),
    (201, 4, [2, 3, 6, 3], [True, True, False, True], [2, 3, 0, 4], "trap", 60, 4),
    (202, 5, [2, 3, 4, 3, 5], [True, False, True, False, True], [2, 0, 3, 0, 3], "trap", 60, 4),
    (203, 4, [3, 2, 5, 4], [True, True, False, False], [3, 2, 0, 0], "trap", 60, 4),
    (204, 3, [2, 4, 3], [True, False, True], [2, 0, 3], "trap", 60, 4),
    (301, 4, [3, 4, 3, 5], [False, False, False, False], [0, 0, 0, 0], "corr", 60, 4),
    (302, 5, [2, 5, 3, 4, 6], [True, False, True, False, False], [2, 0, 3, 0, 0], "trap", 75, 6),
    (303, 4, [3, 3, 4, 4], [True, True, False, False], [3, 4, 0, 0], "trap", 45, 4),
]
_COST_RANGE = dict(cold_lo=25, cold_hi=50, warm_lo=1, warm_hi=2, fatigue_lo=3, fatigue_hi=8)


def make_instances():
    """Deterministic, seeded. Returns [{'public':..., 'hidden':{}}]."""
    out = []
    for seed, m, horizons, fatigue_flags, thresholds, proc_mode, n, k in _SPECS:
        pub = _gen_one(seed, n, m, k, horizons, fatigue_flags, thresholds, proc_mode, **_COST_RANGE)
        out.append({"public": pub, "hidden": {}})
    return out


# ----------------------------- cost model -----------------------------------
def compute_cost(pub, order):
    """Cost of applying permutation `order` (a list of job ids) identically at every
    station -- the line is one-way with no re-sequencing buffers, so every station sees
    the SAME relative job order."""
    jobs = pub["jobs"]
    type_of = [0] * len(jobs)
    for j in jobs:
        type_of[j["id"]] = j["type"]
    seq_types = [type_of[i] for i in order]
    N = len(order)
    total = 0.0
    for st in pub["stations"]:
        H = st["horizon"]; w = st["weight"]
        cold = st["cold_cost"]; warm = st["warm_cost"]
        fatigue_on = st["fatigue_on"]; R = st["fatigue_threshold"]; fcost = st["fatigue_cost"]
        window = []
        counts = {}
        run_len = 0
        prev_type = None
        station_cost = 0.0
        for i in range(N):
            t = seq_types[i]
            c = counts.get(t, 0)
            frac_cold = 1.0 - (min(c, H) / H)
            station_cost += warm[t] + (cold[t] - warm[t]) * frac_cold
            run_len = run_len + 1 if t == prev_type else 1
            if fatigue_on and run_len > R:
                station_cost += fcost[t] * (run_len - R)
            prev_type = t
            window.append(t)
            counts[t] = counts.get(t, 0) + 1
            if len(window) > H:
                old = window.pop(0)
                counts[old] -= 1
        total += w * station_cost
    return total


def baseline(inst):
    """Cost of the 'as-given' identity order -- the order jobs already arrive in."""
    pub = inst["public"]
    return compute_cost(pub, list(range(pub["n"])))


def score(inst, answer):
    """Strictly validate the answer against the instance; return (ok, obj)."""
    pub = inst["public"]
    n = pub["n"]
    if not isinstance(answer, dict):
        return False, None
    order = answer.get("order", None)
    if not isinstance(order, list) or len(order) != n:
        return False, None
    try:
        order_i = []
        for x in order:
            if isinstance(x, bool):
                return False, None
            if isinstance(x, float):
                if not math.isfinite(x) or abs(x - round(x)) > 1e-9:
                    return False, None
                x = int(round(x))
            if not isinstance(x, int):
                return False, None
            order_i.append(x)
    except (TypeError, ValueError):
        return False, None
    if sorted(order_i) != list(range(n)):
        return False, None
    obj = compute_cost(pub, order_i)
    if not math.isfinite(obj) or obj <= 0.0:
        return False, None
    return True, float(obj)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, None
        if not ok or obj is None or obj <= 0:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
