#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0747 -- "Wear Before the Rush: Fatigue-Budgeted Fleet
Routing" (family: fatigue-balanced-routing).

Theme: route incoming jobs across an aging machine fleet. Each of 10 hidden
instances gives the candidate FULL offline visibility of a fleet (units, each
with a per-tick job CAPACITY, a FATIGUE_RATE, a HAZARD_CLIFF, a REPAIR_TIME, and
an idle RECOVER_RATE) and a full job arrival schedule (a demand curve that
includes one seeded, denser SURGE window). The candidate submits ONE routing
decision per job (which unit serves it), and the evaluator deterministically
replays the whole horizon tick by tick:

  * load-fatigue-failure-hazard: every job a unit processes adds
    fatigue_rate * 1 to that unit's cumulative fatigue. The instant a unit's
    fatigue reaches its hazard_cliff, it fails.
  * maintenance-downtime-commitment: a failed unit is completely unavailable
    for the next repair_time ticks (a hard commitment, not a probability), and
    its fatigue resets to 0 only once repair ends.
  * idle recovery: a tick in which a unit is NOT in repair and is assigned NO
    job lets its fatigue decay by recover_rate (floored at 0) -- rest banks
    headroom back.
  * per-tick capacity: a unit can only start up to `capacity` jobs in a single
    tick; jobs routed to an unavailable (repairing) or already-saturated unit
    that tick are dropped (0 value), not queued.

Objective: maximize completed job weight (every job has weight 1, so this is
throughput). Ratio per instance = completed / total (no extra rescaling needed
-- every instance is deliberately over-subscribed so even the reference
strategies never reach 1.0; see calibration below). Final Ratio = mean over the
10 instances. A malformed / out-of-range / wrong-length answer scores 0 on
that instance.

Candidate contract (isolated, stdin/stdout JSON, via isorun):
  input:  {"n_units":.., "horizon":.., "units":[{...}], "jobs":[{...}]}  (the
          FULL instance -- this is an offline problem, nothing is hidden)
  output: {"assignment": [u_0, u_1, ..., u_{m-1}]}   one unit id per job, in
          the SAME order as the input "jobs" array.

CLI: python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean over 10 instances, in [0,1]>
  Vector: [r_1, ..., r_10]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
class RNG:
    def __init__(self, seed):
        self.s = seed & ((1 << 64) - 1)

    def _step(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return self.s

    def uf(self):
        return (self._step() >> 11) / float(1 << 53)

    def randint(self, a, b):
        return a + int(self.uf() * (b - a + 1))


# ----------------------------- instance construction ------------------------
def _unit(capacity, fatigue_rate, hazard_cliff, repair_time, recover_rate):
    return dict(capacity=capacity, fatigue_rate=fatigue_rate, hazard_cliff=hazard_cliff,
                repair_time=repair_time, recover_rate=recover_rate)


def _make_one(seed, n_units, T, blo, bhi, peak_start, peak_len, plo, phi, unit_specs):
    """Build one instance. Every tick draws a job count from a base range
    [blo,bhi] outside the surge window and a heavier [plo,phi] range inside
    [peak_start, peak_start+peak_len) -- the seeded demand peak. Every job has
    weight 1 (throughput objective)."""
    rng = RNG(seed)
    jobs = []
    jid = 0
    for t in range(T):
        if peak_start <= t < peak_start + peak_len:
            cnt = rng.randint(plo, phi)
        else:
            cnt = rng.randint(blo, bhi)
        for _ in range(cnt):
            jobs.append({"id": jid, "t": t, "weight": 1.0})
            jid += 1
    units = [dict(spec, id=i) for i, spec in enumerate(unit_specs)]
    inst = {"n_units": n_units, "horizon": T, "units": units, "jobs": jobs}
    return {"public": inst, "hidden": inst}


def _instance_specs():
    def u3():
        return [_unit(4, 1.0, 20.0, 12, 1.0), _unit(2, 1.0, 16.0, 6, 1.0), _unit(1, 1.0, 40.0, 2, 1.0)]

    def u4():
        return [_unit(4, 1.0, 14.0, 13, 1.0), _unit(3, 1.0, 15.0, 8, 1.0),
                _unit(2, 1.0, 16.0, 5, 1.0), _unit(1, 1.0, 40.0, 2, 1.0)]

    def u5():
        return [_unit(5, 1.0, 15.0, 15, 1.0), _unit(3, 1.0, 15.0, 8, 1.0), _unit(2, 1.0, 16.0, 5, 1.0),
                _unit(1, 1.0, 42.0, 2, 1.0), _unit(1, 1.0, 48.0, 2, 1.0)]

    S = []
    # 0,1,2 -- classic 3-unit fast/fragile trap
    S.append(dict(seed=101, n=3, T=44, blo=0, bhi=2, ps=28, pl=10, plo=5, phi=8, units=u3()))
    S.append(dict(seed=102, n=3, T=48, blo=0, bhi=2, ps=30, pl=10, plo=6, phi=9, units=u3()))
    S.append(dict(seed=103, n=3, T=40, blo=0, bhi=2, ps=24, pl=9, plo=5, phi=8, units=u3()))
    # 3,4 -- 4-unit fleet, tighter fast-unit budget
    S.append(dict(seed=104, n=4, T=52, blo=0, bhi=2, ps=32, pl=11, plo=7, phi=10, units=u4()))
    S.append(dict(seed=105, n=4, T=56, blo=0, bhi=3, ps=34, pl=12, plo=7, phi=11, units=u4()))
    # 5 -- 5-unit fleet
    S.append(dict(seed=106, n=5, T=64, blo=0, bhi=3, ps=40, pl=14, plo=8, phi=13, units=u5()))
    # 6 -- softer 3-unit fleet (smaller capacity spread)
    S.append(dict(seed=201, n=3, T=44, blo=0, bhi=1, ps=26, pl=8, plo=4, phi=6,
                   units=[_unit(2, 1.0, 10.0, 6, 1.0), _unit(2, 1.0, 16.0, 5, 1.0), _unit(1, 1.0, 30.0, 2, 1.0)]))
    # 7 -- softer 4-unit fleet
    S.append(dict(seed=202, n=4, T=48, blo=0, bhi=2, ps=28, pl=9, plo=6, phi=10,
                   units=[_unit(3, 1.0, 12.0, 8, 1.0), _unit(2, 1.0, 15.0, 6, 1.0),
                          _unit(2, 1.0, 17.0, 5, 1.0), _unit(1, 1.0, 35.0, 2, 1.0)]))
    # 8,9 -- held-out generalization: early peak, larger fleets
    S.append(dict(seed=301, n=4, T=60, blo=0, bhi=2, ps=18, pl=8, plo=6, phi=9, units=u4()))
    S.append(dict(seed=302, n=5, T=70, blo=0, bhi=3, ps=20, pl=8, plo=7, phi=12, units=u5()))
    return S


def make_instances():
    out = []
    for sp in _instance_specs():
        out.append(_make_one(sp["seed"], sp["n"], sp["T"], sp["blo"], sp["bhi"],
                              sp["ps"], sp["pl"], sp["plo"], sp["phi"], sp["units"]))
    return out


# ----------------------------- simulation -----------------------------------
def simulate(inst, assignment):
    """Deterministic tick-by-tick replay. Returns (completed_weight, total_weight)
    or None if `assignment` is structurally invalid."""
    units = inst["units"]; T = inst["horizon"]; jobs = inst["jobs"]; n = inst["n_units"]
    if not isinstance(assignment, list) or len(assignment) != len(jobs):
        return None
    for a in assignment:
        if isinstance(a, bool) or not isinstance(a, int) or a < 0 or a >= n:
            return None

    by_tick = {}
    for j, a in zip(jobs, assignment):
        by_tick.setdefault(j["t"], []).append((j, a))

    fatigue = [0.0] * n
    repair_until = [0] * n
    completed_weight = 0.0

    for t in range(T):
        this_tick = by_tick.get(t, [])
        assigned_count = [0] * n
        for j, a in this_tick:
            assigned_count[a] += 1
        # idle recovery: a unit that is NOT in repair and got 0 jobs this tick rests
        for u in range(n):
            if t >= repair_until[u] and assigned_count[u] == 0:
                fatigue[u] = max(0.0, fatigue[u] - units[u]["recover_rate"])
        processed_count = [0] * n
        for j, a in this_tick:
            u = a
            if t < repair_until[u]:
                continue                                    # unit mid-repair: job dropped
            if processed_count[u] >= units[u]["capacity"]:
                continue                                    # over this tick's capacity: dropped
            processed_count[u] += 1
            fatigue[u] += units[u]["fatigue_rate"]
            completed_weight += j["weight"]
            if fatigue[u] >= units[u]["hazard_cliff"]:
                repair_until[u] = t + 1 + units[u]["repair_time"]
                fatigue[u] = 0.0
    total_weight = sum(j["weight"] for j in jobs)
    return completed_weight, total_weight


def _naive_all_to_zero(inst):
    return [0] * len(inst["jobs"])


def score(inst, answer):
    """Validate + simulate the candidate's routing. Return (ok, ratio in [0,1])."""
    if not isinstance(answer, dict) or "assignment" not in answer:
        return False, None
    assignment = answer["assignment"]
    res = simulate(inst["hidden"], assignment)
    if res is None:
        return False, None
    cw, tw = res
    if not (math.isfinite(cw) and math.isfinite(tw)) or tw <= 0:
        return False, None
    ratio = cw / tw
    if not math.isfinite(ratio):
        return False, None
    return True, max(0.0, min(1.0, ratio))


def baseline(inst):
    """Informational only (not used to rescale score()): the throughput ratio of
    the naive 'route every job to unit 0' construction."""
    res = simulate(inst["hidden"], _naive_all_to_zero(inst["hidden"]))
    if res is None:
        return 0.0
    cw, tw = res
    return cw / tw if tw > 0 else 0.0


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
        if not ok or obj is None:
            vec.append(0.0); continue
        vec.append(obj if (obj == obj and 0.0 <= obj <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
