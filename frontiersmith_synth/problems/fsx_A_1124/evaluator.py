#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_1124 -- "Counterdesk Commitments: Repair Shop Turnaround Quotes"
(family: quote-and-balk-counterdesk; format B, quality-metric).

THEME.  A repair shop's front desk takes jobs at a single counter.  Each walk-in job
arrives with a *value* (revenue if fixed and delivered on time), a *service time*
(bench-minutes actually needed), and a *patience* (the longest turnaround the customer
will wait before walking to a competitor instead).  The desk must, for every job, either
REJECT it outright or post a BINDING delay quote d: "ready in d time units."  A quote is
a real commitment, not small talk:
  - the customer's best response is mechanical: they join iff d <= patience (balking
    "admission-control-policy" + "balking-best-response" mechanisms);
  - once joined, the shop MUST actually finish the job by (arrival + d) on pain of a
    heavy fine ("binding-delay-quotes" mechanism) -- the desk cannot renegotiate later.

Mechanically, C identical repair benches process joined jobs by EARLIEST-PROMISED-DEADLINE
FIRST (EDF): whichever joined job has the tightest outstanding promise gets the next free
bench.  This is what makes a quote an OPTION SOLD ON FUTURE CAPACITY: promising a job for
"day 2" reserves it high scheduling priority ahead of time, and a sudden rush of jobs with
even tighter promises can bump an already-promised job out of its slot regardless of when it
arrived.  The seeded traces plant BURSTS of near-simultaneous, high-value, impatient jobs;
a desk that treats a quote as an honest *forecast* of when it expects to finish (rather than
the loosest *commitment* the customer will still accept, and rather than reasoning about
which jobs the burst will force it to decline) racks up violation fines exactly there.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "capacity": C (int), "n": N (int),
             "jobs": [{"t": arrival (int), "value": v (number), "service": s (int),
                       "patience": p (int)}, ...]}   # sorted by t ascending (ties allowed)
  stdout: ONE JSON object:
            {"decisions": [ {"action": "reject"} | {"action": "quote", "delay": d}, ... ]}
          decisions[i] is the desk's call for jobs[i], in the SAME order.  `delay` must be
          a plain (non-bool) integer with `service_i <= delay <= 100000`.

  A job with action "quote" JOINS iff delay <= patience (else it balks: 0 contribution,
  no scheduling, no penalty).  A joined job is scheduled by the frozen EDF-with-capacity-C
  simulation below; if its actual completion exceeds (arrival + delay) it contributes
  `-1.5 * value` (a broken binding promise); otherwise it contributes `+value`.  Rejected /
  balked jobs contribute 0.

  The `decisions` list must have exactly N entries, each well-typed (see above); any
  malformed entry, wrong length, non-JSON output, crash, or timeout scores that instance
  0.0.  There is no per-job partial credit within an instance -- validity is checked once
  up front and, if it holds, EVERY job's outcome (join/balk/finish/violate) feeds the score.

SCORING (deterministic; no wall-time).  Per instance:
    b       = value realized by the evaluator's own naive forecast-based desk policy
              (an M/c mean-field wait estimate; quotes exactly its forecast, no slack)
    U       = sum of all job values (an unreachable clairvoyant ceiling -- capacity can
              never actually serve every job in a burst, so this leaves real headroom)
    cand    = value realized by the candidate's decisions, replayed through the SAME
              frozen EDF simulation used for every policy
    r = clamp( 0.1 + 0.9 * (cand - b) / max(1e-9, U - b), 0, 1 )
  Reproducing the naive baseline scores ~0.1; beating it scores above; doing worse scores
  below (can clamp to 0.0 under heavy fines).

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance (jobs' t/value/service/
patience).  The EDF simulator, the naive baseline policy, and the fine bookkeeping all run
in THIS parent process -- a frame-walking candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, heapq, math
import isorun

PENALTY_MULT = 1.5
MAX_DELAY = 100000


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- instance family -----------------------------
def _calm_job(ni, t):
    value = ni(10, 70)
    svc = ni(2, 6)
    slack = ni(3, 10)
    return {"t": t, "value": float(value), "service": svc, "patience": svc + slack}


def _burst_job(ni, t):
    value = ni(55, 140)
    svc = ni(2, 6)
    slack = ni(0, 3)
    return {"t": t, "value": float(value), "service": svc, "patience": svc + slack}


def _build_jobs(seed, segments):
    """segments: list of (count, mode) with mode in {'calm','burst'}."""
    ni = _rng(seed)
    jobs = []
    t = 0
    for count, mode in segments:
        for _ in range(count):
            if mode == "burst":
                t += ni(0, 1)
                jobs.append(_burst_job(ni, t))
            else:
                t += ni(2, 5)
                jobs.append(_calm_job(ni, t))
    return jobs


def _build_instances():
    specs = [
        # (seed, C, segments, name)
        (301, 2, [(40, "calm")], "steady-a"),
        (302, 2, [(45, "calm")], "steady-b"),
        (303, 3, [(20, "calm"), (6, "burst"), (20, "calm")], "mild-burst"),
        (304, 2, [(15, "calm"), (10, "burst"), (15, "calm")], "burst-a"),
        (305, 2, [(18, "calm"), (11, "burst"), (17, "calm")], "burst-b"),
        (306, 3, [(15, "calm"), (9, "burst"), (12, "calm"), (9, "burst"), (10, "calm")], "double-burst-a"),
        (307, 2, [(44, "calm")], "steady-c"),
        (308, 3, [(12, "calm"), (10, "burst"), (12, "calm"), (10, "burst"), (12, "calm"), (10, "burst"), (10, "calm")], "burst-heavy-tail"),
        (309, 4, [(20, "calm"), (10, "burst"), (15, "calm"), (10, "burst"), (10, "calm")], "mixed-held-out"),
        (310, 3, [(20, "calm"), (12, "burst"), (18, "calm"), (12, "burst"), (18, "calm")], "double-burst-held-out"),
    ]
    out = []
    for seed, C, segments, name in specs:
        jobs = _build_jobs(seed, segments)
        out.append({"name": name, "capacity": C, "n": len(jobs), "jobs": jobs})
    return out


# ----------------------------- ground-truth dispatcher ----------------------
def simulate_schedule(admitted, C):
    """admitted: list of dicts with idx,t,service,deadline (=t+delay). Returns
    {idx: completion_time} under EARLIEST-PROMISED-DEADLINE-FIRST dispatch across
    C identical benches. Deterministic tie-break: (deadline, t, idx)."""
    order = sorted(admitted, key=lambda j: (j["t"], j["idx"]))
    by_idx = {j["idx"]: j for j in admitted}
    n = len(order)
    bench_free = [0] * C
    heapq.heapify(bench_free)
    waiting = []
    i = 0
    completion = {}
    while i < n or waiting:
        next_arrival_t = order[i]["t"] if i < n else None
        if waiting:
            next_free_t = bench_free[0]
            if next_arrival_t is None or next_free_t <= next_arrival_t:
                T = next_free_t
            else:
                T = next_arrival_t
        else:
            T = next_arrival_t
        while i < n and order[i]["t"] <= T:
            j = order[i]
            heapq.heappush(waiting, (j["deadline"], j["t"], j["idx"]))
            i += 1
        while waiting and bench_free[0] <= T:
            heapq.heappop(bench_free)
            _, _, idx = heapq.heappop(waiting)
            j = by_idx[idx]
            start = T
            fin = start + j["service"]
            completion[idx] = fin
            heapq.heappush(bench_free, fin)
    return completion


def _realize(jobs, decisions):
    """decisions: list of (action, delay|None) already type/range-validated, aligned
    with jobs. Returns total realized value under simulate_schedule. Never raises."""
    admitted = []
    joined_meta = {}
    for idx, (job, (action, delay)) in enumerate(zip(jobs, decisions)):
        if action != "quote":
            continue
        if delay > job["patience"]:
            continue  # balks
        admitted.append({"idx": idx, "t": job["t"], "service": job["service"],
                          "deadline": job["t"] + delay})
        joined_meta[idx] = (job["value"], job["t"] + delay)
    completion = simulate_schedule(admitted, _CUR_C[0])
    total = 0.0
    for idx, (value, deadline) in joined_meta.items():
        fin = completion.get(idx)
        if fin is None or fin > deadline:
            total -= PENALTY_MULT * value
        else:
            total += value
    return total


_CUR_C = [2]  # set per-instance before calling _realize / baseline policies


# ----------------------------- naive baseline policy ------------------------
def _baseline_decisions(jobs, C):
    """Mean-field M/c forecast: track scalar backlog W, drained at rate C per unit
    time since the last update; quote = ceil(W/C) + service (the forecast itself,
    with NO strategic slack). Admit iff quote <= patience."""
    decisions = []
    W = 0.0
    t_ref = 0
    for job in jobs:
        elapsed = job["t"] - t_ref
        W = max(0.0, W - elapsed * C)
        t_ref = job["t"]
        wait = math.ceil(W / C) if C > 0 else 0
        delay = wait + job["service"]
        if delay <= job["patience"]:
            decisions.append(("quote", delay))
            W += job["service"]
        else:
            decisions.append(("reject", None))
    return decisions


def baseline(inst):
    jobs = inst["jobs"]
    C = inst["capacity"]
    _CUR_C[0] = C
    decisions = _baseline_decisions(jobs, C)
    return _realize(jobs, decisions)


# ----------------------------- answer validation -----------------------------
def _validate(inst, answer):
    jobs = inst["jobs"]
    n = inst["n"]
    if not isinstance(answer, dict):
        return None
    dec = answer.get("decisions")
    if not isinstance(dec, list) or len(dec) != n:
        return None
    out = []
    for job, d in zip(jobs, dec):
        if not isinstance(d, dict):
            return None
        action = d.get("action")
        if action == "reject":
            out.append(("reject", None))
        elif action == "quote":
            delay = d.get("delay")
            if isinstance(delay, bool) or not isinstance(delay, int):
                return None
            if delay != delay or delay < job["service"] or delay > MAX_DELAY:
                return None
            out.append(("quote", delay))
        else:
            return None
    return out


def score(inst, answer):
    decisions = _validate(inst, answer)
    if decisions is None:
        return False, 0.0
    _CUR_C[0] = inst["capacity"]
    total = _realize(inst["jobs"], decisions)
    if total != total or total in (float("inf"), float("-inf")):
        return False, 0.0
    return True, total


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        b = baseline(inst)
        U = sum(j["value"] for j in inst["jobs"])
        denom = U - b
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "capacity": inst["capacity"], "n": inst["n"],
                  "jobs": [{"t": j["t"], "value": j["value"], "service": j["service"],
                            "patience": j["patience"]} for j in inst["jobs"]]}
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
        r = 0.1 + 0.9 * (obj - b) / denom
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
