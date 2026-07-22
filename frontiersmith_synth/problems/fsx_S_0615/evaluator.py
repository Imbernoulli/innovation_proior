#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0615 -- "Headroom Gate: Admission Control for a Shared
Compute Cluster Under Self-Inflicted Congestion" (family: latency-headroom-admission;
format B, quality-metric; theme: admission control for a shared compute cluster).

THEME.  A single shared cluster is modeled as one work-conserving server that runs
admitted jobs NON-PREEMPTIVELY in arrival order (FIFO by arrival time, ties by index).
A deterministic, SEEDED arrival stream of N jobs hits the gate.  Job i is
(a_i arrival, s_i service size, v_i value, d_i deadline).  The WHOLE stream is known
in advance (it is the public instance): future arrivals are ANTICIPATED, not a
surprise.  Your program is the admission controller: for every job you output
admit(1) or drop(0).  You do NOT reorder the server -- admission is your only lever.

QUEUE-LATENCY-BUILDUP FEEDBACK (mechanism 1).  The server is work-conserving: it
starts each admitted job when it is both present and the server is free, and holds the
server for the job's full size s_i.  So admitting a job now deterministically RAISES
the finish time of every admitted job behind it in arrival order -- congestion you
inflict on yourself.  Formally, over admitted jobs sorted by (a_i, i):
    free_0 = 0;  start_i = max(free_prev, a_i);  finish_i = start_i + s_i;  free = finish_i.

SLA DEADLINE-PENALTY RAMP (mechanism 2).  An admitted job that FINISHES by its
deadline earns its value v_i.  An admitted job that misses earns NOTHING and instead
pays a penalty that RAMPS with lateness:  beta * (finish_i - d_i).  A dropped job earns
0 and costs 0.

OBJECTIVE (maximize) per instance:
    J = sum_{admitted, finish<=d} v_i  -  beta * sum_{admitted, finish>d} (finish_i - d_i)
Admitting nothing yields J = 0.

INNOVATION HOOK (what `strong` exploits).  Because the stream is public, the gate can
RESERVE LATENCY HEADROOM: refuse a marginal low-value job now whose service time would
push the FIFO tail past the deadline of a premium job known to arrive soon.  The trap
family plants exactly this: a burst of small low-value jobs (each individually able to
meet its own loose deadline, so "admit anything positive" takes them all) arrives just
before a high-value job with a tight deadline; taking the burst busies the server so the
premium job starts late, misses, and is BOTH forfeited and penalized.  The insight is to
select jobs by value under a GLOBAL all-on-time feasibility check -- protecting the
premium set and dropping the low-value backlog that would starve it of headroom.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "N": int, "beta": float,
             "a": [a_0 ... a_{N-1}],   # arrival times (non-decreasing)
             "s": [s_0 ... s_{N-1}],   # service sizes (>0)
             "v": [v_0 ... v_{N-1}],   # values (>0)
             "d": [d_0 ... d_{N-1}]}   # absolute deadlines (finish-by)
  stdout: ONE JSON object:
            {"admit": [x_0 ... x_{N-1}]}   # each x_i in {0,1}
  VALID iff "admit" is a list of exactly N entries each equal to 0 or 1 (bools ok;
  no NaN/inf/other numbers).  Any violation, crash, timeout, or non-JSON -> 0.0.
  NOTE: admitting a set that misses deadlines is LEGAL but penalized -- feasibility is
  the solver's job, not a validity gate.

SCORING (deterministic; no wall-time).  Per instance:
    J_cand = objective above for the candidate's admit vector.
    A loose, UNREACHABLE upper bound is the total value if EVERY job were served on
    time: hi = GAIN * sum_i v_i.  Then
       r = clamp( 0.1 + 0.9 * J_cand / hi , 0, 1 ).
  Admitting nothing scores exactly 0.1; the slack in hi (and the fact that not all jobs
  can be on time under one server) keeps even an oracle below ~0.90, so a strong gate
  leaves headroom.  The final score is the mean of r over 10 fixed seeded instances
  (headroom-trap single-premium cases, mildly congested mixed cases, uncongested calm
  cases, and a held-out twin-premium regime).

ISOLATION.  The candidate is untrusted and runs in a FRESH bwrap-SANDBOXED SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC instance.  The objective and
the normalization are computed by THIS parent process.

CLI:  python3 evaluator.py <solution.py>
"""
import sys, json
import isorun

GAIN = 1.20   # normalization slack: larger -> lower ratios / more headroom


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


# ----------------------------- transition / objective ----------------------
def objective(inst, admit):
    """Replay the FIFO-by-arrival server for the admitted set; return the objective J."""
    N = inst["N"]
    a = inst["a"]; s = inst["s"]; v = inst["v"]; d = inst["d"]
    beta = inst["beta"]
    order = sorted((i for i in range(N) if admit[i] == 1), key=lambda i: (a[i], i))
    free = 0.0
    J = 0.0
    for i in order:
        start = a[i] if a[i] > free else free
        finish = start + s[i]
        free = finish
        if finish <= d[i] + 1e-9:
            J += v[i]
        else:
            J -= beta * (finish - d[i])
    return J


def _hi(inst):
    tot = 0.0
    for x in inst["v"]:
        tot += x
    return GAIN * tot


# ----------------------------- instance family -----------------------------
def _mk(seed, name, prof):
    """Build one instance.  A profile dict declares:
       burst  = (K, sL_lo, sL_hi, vL_lo, vL_hi)  low-value early jobs at t=0..K-1
       prem   = list of (offset_after_burst, sP, vP, slack)  tight-deadline premium jobs
       extra  = (M, gap, s_lo, s_hi, v_lo, v_hi, slack_lo, slack_hi) spread-out jobs
       beta   = penalty ramp slope
    All randomness is seeded; deadlines are absolute finish-by times."""
    ni = _rng(seed)
    a = []; s = []; v = []; d = []
    K, sLl, sLh, vLl, vLh = prof["burst"]
    # --- burst of low-value jobs, each with a LOOSE self-deadline (so a myopic
    #     'admit anything that can meet its own deadline' also takes them all) ---
    for t in range(K):
        sz = ni(sLl, sLh)
        a.append(t)
        s.append(float(sz))
        v.append(float(ni(vLl, vLh)))
        d.append(float(t + sz + ni(30, 50)))     # generous individual deadline
    # --- premium jobs arriving right after the burst with TIGHT deadlines ---
    for (off, sP, vP, slack) in prof["prem"]:
        ap = K + off
        a.append(ap)
        s.append(float(sP))
        v.append(float(vP + ni(-2, 2)))
        d.append(float(ap + sP + slack))          # just enough IF the server is free
    # --- extra spread-out medium jobs (fill the tail; add mild congestion) ---
    if prof.get("extra"):
        M, gap, sl, sh, vl, vh, kl, kh = prof["extra"]
        t = K + 2
        for _ in range(M):
            t += gap + ni(0, 1)
            sz = ni(sl, sh)
            a.append(float(t))
            s.append(float(sz))
            v.append(float(ni(vl, vh)))
            d.append(float(t + sz + ni(kl, kh)))
    N = len(a)
    return {"name": name, "N": N, "beta": float(prof["beta"]),
            "a": a, "s": s, "v": v, "d": d}


def _build_instances():
    specs = []
    # --- headroom-trap single-premium cases (the obvious 'admit all' is far from strong) ---
    specs.append(("trap1", 70301, {
        "burst": (6, 3, 3, 8, 12), "prem": [(0, 4, 74, 2)],
        "extra": (3, 6, 2, 4, 6, 10, 6, 12), "beta": 4.0}))
    specs.append(("trap2", 70302, {
        "burst": (8, 2, 3, 7, 11), "prem": [(0, 5, 82, 3)],
        "extra": (3, 7, 2, 4, 5, 9, 6, 12), "beta": 5.0}))
    specs.append(("trap3", 70303, {
        "burst": (7, 3, 4, 9, 13), "prem": [(1, 4, 68, 3)],
        "extra": (2, 8, 2, 4, 6, 10, 6, 14), "beta": 3.5}))
    # --- mildly congested mixed cases (partial penalties for admit-all; strong wins less) ---
    specs.append(("mix1", 70311, {
        "burst": (4, 3, 4, 12, 20), "prem": [(0, 4, 40, 5)],
        "extra": (5, 4, 3, 6, 10, 24, 5, 16), "beta": 3.0}))
    specs.append(("mix2", 70312, {
        "burst": (5, 2, 3, 14, 22), "prem": [(1, 3, 34, 6)],
        "extra": (5, 5, 2, 5, 12, 26, 6, 18), "beta": 2.5}))
    specs.append(("mix3", 70313, {
        "burst": (3, 3, 5, 16, 26), "prem": [(0, 5, 44, 6)],
        "extra": (6, 4, 3, 6, 10, 22, 6, 18), "beta": 3.0}))
    # --- calm / uncongested cases (everything fits; admit-all ~ strong, keeps greedy high) ---
    specs.append(("calm1", 70321, {
        "burst": (0, 0, 0, 0, 0), "prem": [],
        "extra": (10, 8, 2, 5, 10, 30, 20, 40), "beta": 2.0}))
    specs.append(("calm2", 70322, {
        "burst": (0, 0, 0, 0, 0), "prem": [],
        "extra": (11, 9, 2, 4, 12, 28, 22, 44), "beta": 2.0}))
    specs.append(("calm3", 70323, {
        "burst": (2, 2, 3, 10, 16), "prem": [],
        "extra": (10, 9, 2, 5, 10, 30, 22, 42), "beta": 2.0}))
    # --- held-out twin-premium regime (two tight premiums; even a good gate is imperfect) ---
    specs.append(("twin1", 70331, {
        "burst": (6, 3, 3, 8, 12), "prem": [(0, 4, 60, 2), (2, 4, 66, 6)],
        "extra": (3, 6, 2, 4, 6, 10, 6, 12), "beta": 4.0}))
    out = []
    for name, seed, prof in specs:
        out.append(_mk(seed, name, prof))
    return out


# ----------------------------- validation ----------------------------------
def _valid_admit(inst, answer):
    if not isinstance(answer, dict):
        return None
    v = answer.get("admit")
    N = inst["N"]
    if not isinstance(v, list) or len(v) != N:
        return None
    out = []
    for x in v:
        if isinstance(x, bool):
            out.append(1 if x else 0)
            continue
        if not isinstance(x, (int, float)):
            return None
        if x != x or x in (float("inf"), float("-inf")):
            return None
        if x == 0:
            out.append(0)
        elif x == 1:
            out.append(1)
        else:
            return None
    return out


# ----------------------------- scoring driver ------------------------------
def _public(inst):
    return {"name": inst["name"], "N": inst["N"], "beta": inst["beta"],
            "a": list(inst["a"]), "s": list(inst["s"]),
            "v": list(inst["v"]), "d": list(inst["d"])}


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        hi = _hi(inst)
        if hi <= 1e-9:
            hi = 1e-9
        ans, st = isorun.run_candidate(cand, _public(inst), timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            admit = _valid_admit(inst, ans)
            if admit is None:
                vec.append(0.0)
                continue
            J = objective(inst, admit)
        except Exception:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * J / hi
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
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
