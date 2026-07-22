#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0769 -- "Ridgemont ER: Pre-Escalation Bay Triage"
(family: pre-escalation-triage; format B, quality-metric).

THEME.  An emergency room has N treatment bays and receives a seeded stream
of P patients over a T-step shift.  Every waiting patient's acuity climbs
along a DETERMINISTIC escalation curve (acuity += rate every step they wait,
public per-patient rate); acuity is FROZEN the instant a bay starts treating
them.  Two frictions compose into one objective:

  (1) deadline-acuity-escalation: the reward multiplier AND the treatment
      duration both depend on the acuity level at the moment treatment
      STARTS. Starting at or below the mid threshold C is fast (base_need
      steps) and pays full multiplier (1.0). Once acuity has already tipped
      past C, treatment both pays a reduced multiplier (0.35, a "reactive
      rescue") AND takes strictly longer (base_need + slope*(acuity-C)
      steps) -- reacting late costs a bay MORE capacity, not less. If a
      waiting patient's acuity ever exceeds the death cap D, they die
      (reward = -weight), full stop.
  (2) preemption-switching-cost: a bay may be reassigned to a different
      patient mid-treatment. Doing so is a PREEMPTION: the abandoned
      patient's progress resets to zero (they resume waiting, resume
      escalating) and the bay is forced into S steps of unusable setup
      downtime, AND the evaluator subtracts switch_penalty per preemption
      event directly from the score. Freeing a bay to idle (not switching
      to a new patient) is free but still drops the abandoned patient's
      progress.

The two mechanisms interact: a policy that always chases whichever patient
looks worst RIGHT NOW keeps preempting bays as new, flashier arrivals show
up (paying S-step downtime + switch_penalty repeatedly, and reinflating the
abandoned patients' eventual treatment length), while patients who are only
moderately urgent right now but about to cross C on their OWN deterministic
curve get silently ignored until they, too, spike into a reactive-rescue or
a death. The insight the strong solution exploits: use the public rate/
acuity fields to predict EACH waiting patient's own time-to-tip
((C - acuity) / rate) and fill idle bays by that predicted deadline, almost
never preempting an already-running treatment -- preventing crises rather
than reacting to them.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
    {"name": str, "T": int, "N": int, "P": int,
     "arrival": [P ints], "L0": [P floats], "rate": [P floats],
     "weight": [P ints], "C": float, "D": float,
     "base_need": int, "slope": float, "S": int, "switch_penalty": float}
  stdout: ONE JSON object:
    {"assign": [[b_0, ..., b_{N-1}] for t in 0..T-1]}
    assign[t][b] is the patient id (1..P) requested for bay b at step t, or
    0 for "no request / idle".

  Feasibility (checked by the evaluator on the FULL instance):
    - `assign` is a list of exactly T rows, each a list of exactly N ints
      (bools rejected), each in [0, P].
  Wrong shape/length, an out-of-range/non-integer entry, non-finite value, a
  crash, a timeout, or non-JSON output makes that instance score 0.0.
  Everything else (targeting a patient who hasn't arrived, is already done,
  dead, or is being treated at another bay this step) is NOT a hard
  invalidation -- such a request is simply ignored (treated as idle) for
  that bay that step, so a minor scheduling slip never zeroes the instance.

SIMULATION (deterministic; the evaluator replays `assign` itself; no
wall-time).  Each step: (a) patients whose arrival==t become 'waiting' at
L0; (b) every already-waiting patient's acuity += rate (patients who die,
acuity > D, are marked dead with reward -weight); (c) bays are processed in
index order: continuing the same patient advances progress by 1 (patient
resolves, with reward = weight * multiplier(start_acuity), once progress
reaches need(start_acuity)); requesting a NEW patient into an idle bay
starts a fresh attempt (start_acuity = current acuity); requesting a
DIFFERENT patient into a bay that is mid-treatment triggers a preemption
(abandoned patient -> waiting, progress lost, bay -> S steps of setup, one
switch_penalty charged, the new request itself is wasted this step). A
patient still un-resolved at the shift's end scores 0 (neutral, not death).

SCORING.  obj = sum(patient rewards) - switch_penalty * (#preemptions). The
evaluator computes a loose, deliberately-unreachable upper bound q_ideal =
sum(weight) (every patient resolved at multiplier 1.0, zero preemptions --
never actually reachable given the seeded arrival bursts and bay/time
scarcity) and anchors:
    r = clamp( 0.1 + 0.9 * obj / q_ideal, 0, 1 )
Zero-reward (e.g. doing nothing) scores exactly 0.1; obj < 0 (net deaths)
scores below 0.1, clipped at 0.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  All
simulation/validation/bounds are computed by THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    def nxt_float(lo, hi):
        raw = nxt_int(0, 1_000_000)
        return lo + (hi - lo) * raw / 1_000_000.0

    return nxt_int, nxt_float


# ----------------------------- patient-family builders ----------------------
def _urgent_burst(ni, nf, patients, start_t, count, gap, l0_lo, l0_hi, rate_lo, rate_hi):
    """Repeated 'flashy' newcomers: high-looking initial acuity, arriving
    every `gap` steps. These are what tempts a chase-the-sickest policy into
    preempting bays over and over."""
    t = start_t
    for _ in range(count):
        patients.append({
            "arrival": t, "L0": round(nf(l0_lo, l0_hi), 2),
            "rate": round(nf(rate_lo, rate_hi), 2),
            "weight": ni(4, 5),
        })
        t += gap


def _sleepers(ni, nf, patients, start_lo, start_hi, count, spread,
              l0_lo, l0_hi, rate_lo, rate_hi):
    """Patients that look only moderately urgent when they arrive but climb
    FAST -- ignored by a policy that only looks at current rank, they tip
    into crisis (or die) while attention is elsewhere."""
    for i in range(count):
        a = ni(start_lo, start_hi) + i * spread
        patients.append({
            "arrival": a, "L0": round(nf(l0_lo, l0_hi), 2),
            "rate": round(nf(rate_lo, rate_hi), 2),
            "weight": ni(3, 5),
        })


def _filler(ni, nf, patients, start_lo, start_hi, count, spread):
    """Low-acuity, slow-climbing background patients -- easy multiplier-1.0
    pickups that keep bays legitimately busy between crises."""
    for i in range(count):
        a = ni(start_lo, start_hi) + i * spread
        patients.append({
            "arrival": a, "L0": round(nf(2.0, 3.2), 2),
            "rate": round(nf(0.25, 0.5), 2),
            "weight": ni(1, 3),
        })


def _finalize_patients(patients, T):
    for p in patients:
        p["arrival"] = max(0, min(T - 1, p["arrival"]))
    patients.sort(key=lambda p: p["arrival"])
    return patients


def _congestion_burst(ni, nf, patients, start_t, count, gap, l0_lo, l0_hi, rate_lo, rate_hi,
                       w_lo=4, w_hi=5):
    """A cluster of `count` patients that arrive close together (gap steps
    apart) already fairly acute. With only N bays, more than N of them
    converging inside a short window is UNAVOIDABLE local congestion: no
    matter how smart the scheduler is, someone in the cluster tips past C
    (or worse) before a bay can reach them -- this is what keeps even the
    best strategy off a perfect score."""
    t = start_t
    for _ in range(count):
        patients.append({
            "arrival": t, "L0": round(nf(l0_lo, l0_hi), 2),
            "rate": round(nf(rate_lo, rate_hi), 2),
            "weight": ni(w_lo, w_hi),
        })
        t += gap


def _build_instances():
    specs = []

    # idx0: calm, small -- gentle ramp, generous slack, no clustering and
    # no repeated-newcomer temptation; N=3 bays gives real parallel
    # headroom a single bay does not have
    specs.append(("calm_small", 811, 34, 3, lambda ni, nf: _finalize_patients(
        (lambda ps: (_filler(ni, nf, ps, 0, 10, 4, 6),
                     _sleepers(ni, nf, ps, 8, 16, 2, 6, 5.0, 6.0, 0.55, 0.8),
                     ps)[-1])([]), 34)))

    # idx1: calm, medium -- a bit more load but still comfortably slack
    # with 3 bays available, no clustering
    specs.append(("calm_medium", 812, 42, 3, lambda ni, nf: _finalize_patients(
        (lambda ps: (_filler(ni, nf, ps, 0, 12, 5, 6),
                     _sleepers(ni, nf, ps, 6, 18, 4, 6, 4.8, 5.8, 0.55, 0.8),
                     ps)[-1])([]), 42)))

    # idx2: TRAP -- fillers occupy bays first; a burst of flashy newcomers
    # then repeatedly tempts preemption while fast-climbing sleepers,
    # buried inside/just after the burst, get ignored because they don't
    # yet LOOK the worst; survivable with 2 bays IF you never thrash
    specs.append(("trap_duel_sleepers", 813, 40, 2, lambda ni, nf: _finalize_patients(
        (lambda ps: (_filler(ni, nf, ps, 0, 2, 2, 2),
                     _sleepers(ni, nf, ps, 3, 6, 3, 2, 4.2, 5.2, 1.1, 1.4),
                     _urgent_burst(ni, nf, ps, 6, 4, 3, 7.2, 8.3, 0.7, 0.95),
                     ps)[-1])([]), 40)))

    # idx3: TRAP -- tighter version: fillers + shorter gap between urgent
    # newcomers, sleepers climb a bit faster, still 2 bays
    specs.append(("trap_tight", 814, 38, 2, lambda ni, nf: _finalize_patients(
        (lambda ps: (_filler(ni, nf, ps, 0, 2, 2, 2),
                     _sleepers(ni, nf, ps, 2, 5, 4, 2, 4.0, 5.0, 1.2, 1.5),
                     _urgent_burst(ni, nf, ps, 5, 5, 3, 7.3, 8.4, 0.7, 0.95),
                     ps)[-1])([]), 38)))

    # idx4: calm, single bay -- light load, shows 1 bay is fine when
    # arrivals stay sparse and nobody clusters
    specs.append(("calm_tight_bays", 815, 40, 1, lambda ni, nf: _finalize_patients(
        (lambda ps: (_filler(ni, nf, ps, 0, 12, 3, 10),
                     _sleepers(ni, nf, ps, 22, 30, 2, 10, 4.8, 5.6, 0.5, 0.75),
                     ps)[-1])([]), 40)))

    # idx5: TRAP -- fillers, then two moderate congestion clusters with
    # sleepers interleaved between them, 2 bays
    specs.append(("trap_multi_cluster", 816, 55, 2, lambda ni, nf: _finalize_patients(
        (lambda ps: (_filler(ni, nf, ps, 0, 3, 2, 2),
                     _congestion_burst(ni, nf, ps, 4, 3, 3, 6.8, 7.8, 0.7, 0.95),
                     _sleepers(ni, nf, ps, 3, 6, 3, 3, 4.2, 5.2, 1.0, 1.3),
                     _congestion_burst(ni, nf, ps, 30, 3, 3, 6.8, 7.8, 0.7, 0.95),
                     _sleepers(ni, nf, ps, 27, 31, 3, 3, 4.2, 5.2, 1.0, 1.3),
                     ps)[-1])([]), 55)))

    # idx6: calm, long horizon -- diluted arrivals, plenty of slack, N=3
    specs.append(("calm_long", 817, 60, 3, lambda ni, nf: _finalize_patients(
        (lambda ps: (_filler(ni, nf, ps, 0, 16, 5, 8),
                     _sleepers(ni, nf, ps, 30, 40, 3, 8, 5.0, 6.0, 0.55, 0.8),
                     ps)[-1])([]), 60)))

    # idx7: TRAP -- quiet first stretch (lulling a reactive policy into
    # complacency) while fillers occupy bays and sleepers quietly climb
    # right behind them, then a dense urgent burst arrives, 2 bays
    specs.append(("trap_late_surge", 818, 46, 2, lambda ni, nf: _finalize_patients(
        (lambda ps: (_filler(ni, nf, ps, 0, 2, 2, 2),
                     _sleepers(ni, nf, ps, 2, 5, 4, 2, 4.0, 5.0, 1.0, 1.3),
                     _urgent_burst(ni, nf, ps, 20, 4, 4, 7.3, 8.4, 0.7, 0.95),
                     ps)[-1])([]), 46)))

    # idx8: held-out, large -- bigger P, more bays, two mild clusters plus
    # sleepers, generalization test
    specs.append(("held_out_large", 821, 62, 3, lambda ni, nf: _finalize_patients(
        (lambda ps: (_filler(ni, nf, ps, 0, 4, 3, 2),
                     _congestion_burst(ni, nf, ps, 6, 5, 2, 6.8, 7.8, 0.65, 0.9),
                     _sleepers(ni, nf, ps, 4, 8, 4, 3, 4.4, 5.4, 0.9, 1.2),
                     _congestion_burst(ni, nf, ps, 36, 5, 2, 6.8, 7.8, 0.65, 0.9),
                     _filler(ni, nf, ps, 46, 52, 3, 2),
                     ps)[-1])([]), 62)))

    # idx9: held-out, large + trap -- big scale AND dense multi-burst trap,
    # 2 bays for a bigger patient count
    specs.append(("held_out_trap", 822, 62, 2, lambda ni, nf: _finalize_patients(
        (lambda ps: (_filler(ni, nf, ps, 0, 3, 2, 2),
                     _sleepers(ni, nf, ps, 2, 5, 4, 2, 4.0, 5.0, 1.0, 1.3),
                     _congestion_burst(ni, nf, ps, 5, 4, 3, 6.9, 7.9, 0.65, 0.9),
                     _sleepers(ni, nf, ps, 26, 30, 3, 2, 4.0, 5.0, 1.0, 1.3),
                     _congestion_burst(ni, nf, ps, 34, 4, 3, 6.9, 7.9, 0.65, 0.9),
                     _filler(ni, nf, ps, 50, 56, 3, 2),
                     ps)[-1])([]), 62)))

    out = []
    for name, seed, T, N, builder in specs:
        ni, nf = _rng(seed)
        patients = builder(ni, nf)
        P = len(patients)
        arrival = [p["arrival"] for p in patients]
        L0 = [p["L0"] for p in patients]
        rate = [p["rate"] for p in patients]
        weight = [p["weight"] for p in patients]
        S = 4 if "held_out" in name else 3
        switch_penalty = 1.4 if "trap" in name else 1.0
        out.append({
            "name": name, "T": T, "N": N, "P": P,
            "arrival": arrival, "L0": L0, "rate": rate, "weight": weight,
            "C": 9.0, "D": 16.0, "base_need": 4, "slope": 0.8,
            "S": S, "switch_penalty": switch_penalty,
        })
    return out


# ----------------------------- simulation / scoring -------------------------
def _need(a, base_need, slope, C):
    if a <= C + 1e-9:
        return base_need
    v = base_need + slope * (a - C)
    return max(base_need, int(math.ceil(v - 1e-9)))


def _simulate(inst, answer):
    T, N, P = inst["T"], inst["N"], inst["P"]
    arrival, L0, rate, weight = inst["arrival"], inst["L0"], inst["rate"], inst["weight"]
    C, D = inst["C"], inst["D"]
    base_need, slope = inst["base_need"], inst["slope"]
    S, switch_penalty = inst["S"], inst["switch_penalty"]

    if not isinstance(answer, dict):
        return False, None
    assign = answer.get("assign")
    if not isinstance(assign, list) or len(assign) != T:
        return False, None
    grid = []
    for row in assign:
        if not isinstance(row, list) or len(row) != N:
            return False, None
        r = []
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int):
                return False, None
            if v < 0 or v > P:
                return False, None
            r.append(v)
        grid.append(r)

    status = ["not_arrived"] * P     # not_arrived | waiting | treating | done | dead
    acuity = [0.0] * P
    reward = [0.0] * P

    bay_status = ["idle"] * N        # idle | treating | setup
    bay_patient = [None] * N
    bay_progress = [0] * N
    bay_start_acuity = [0.0] * N
    bay_setup_remaining = [0] * N

    switch_events = 0

    def finalize(p, start_a):
        mult = 1.0 if start_a <= C + 1e-9 else 0.35
        reward[p] = weight[p] * mult
        status[p] = "done"

    for t in range(T):
        for p in range(P):
            if status[p] == "not_arrived" and arrival[p] == t:
                status[p] = "waiting"
                acuity[p] = L0[p]
        for p in range(P):
            if status[p] == "waiting" and arrival[p] < t:
                acuity[p] += rate[p]
                if acuity[p] > D + 1e-9:
                    status[p] = "dead"
                    reward[p] = -weight[p]

        for b in range(N):
            req = grid[t][b]
            if bay_status[b] == "setup":
                bay_setup_remaining[b] -= 1
                if bay_setup_remaining[b] <= 0:
                    bay_status[b] = "idle"
                continue
            if req == 0:
                if bay_status[b] == "treating":
                    y = bay_patient[b]
                    status[y] = "waiting"
                    bay_status[b] = "idle"
                    bay_patient[b] = None
                    bay_progress[b] = 0
                continue
            p0 = req - 1
            if bay_status[b] == "treating" and bay_patient[b] == p0:
                bay_progress[b] += 1
                need = _need(bay_start_acuity[b], base_need, slope, C)
                if bay_progress[b] >= need:
                    finalize(p0, bay_start_acuity[b])
                    bay_status[b] = "idle"
                    bay_patient[b] = None
                    bay_progress[b] = 0
                continue
            if status[p0] != "waiting":
                continue
            if bay_status[b] == "idle":
                bay_status[b] = "treating"
                bay_patient[b] = p0
                bay_progress[b] = 1
                bay_start_acuity[b] = acuity[p0]
                status[p0] = "treating"
                need = _need(bay_start_acuity[b], base_need, slope, C)
                if bay_progress[b] >= need:
                    finalize(p0, bay_start_acuity[b])
                    bay_status[b] = "idle"
                    bay_patient[b] = None
                    bay_progress[b] = 0
            else:
                y = bay_patient[b]
                status[y] = "waiting"
                switch_events += 1
                bay_status[b] = "setup"
                bay_patient[b] = None
                bay_progress[b] = 0
                bay_setup_remaining[b] = S - 1
                if bay_setup_remaining[b] <= 0:
                    bay_status[b] = "idle"

    obj = sum(reward) - switch_penalty * switch_events
    return True, obj


def _ideal_bound(inst):
    return max(sum(inst["weight"]), 1e-6)


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = {k: inst[k] for k in
                  ("name", "T", "N", "P", "arrival", "L0", "rate", "weight",
                   "C", "D", "base_need", "slope", "S", "switch_penalty")}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            feasible, obj = _simulate(inst, ans)
        except Exception:
            feasible, obj = False, None
        if not feasible or obj is None:
            vec.append(0.0)
            continue
        q_ideal = _ideal_bound(inst)
        r = 0.1 + 0.9 * obj / q_ideal
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
