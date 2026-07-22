#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0954 -- "Job Router for a Shop of Aging Machines"
(family: state-matched-dispatch-policy; format B, quality-metric).

THEME.  A small job shop runs `M` heterogeneous machines. Jobs of two kinds
arrive one at a time in a fixed stream: ABRASIVE ("A") jobs are rough
material-removal work, FINISHING ("F") jobs are precision polishing work.
Every machine has its own fixed abrasion sensitivity `a` and polish
sensitivity `q` (heterogeneous-wear-machines). Each machine also has a
current SPEED state `spd` (starts at 1.0). Running an abrasive job on a
machine multiplies that machine's speed by `(1 - a)` (it dulls); running a
finishing job multiplies its speed by `(1 + q)` (it self-sharpens) --
wear-job-affinity: the SAME machine can be a poor choice for one job type and
a great choice for the other. Speed is clamped to [SPD_MIN, SPD_MAX] so no
machine ever fully stalls or diverges to infinity.

Jobs are revealed ONE AT A TIME from a fixed, pre-generated stream and must
be assigned to a machine IMMEDIATELY and IRREVOCABLY (sequential-irrevocable-
assignment) -- there is no reassignment, no lookahead at future jobs, and no
undo. Each machine processes the jobs assigned to it strictly in the order
they were assigned (FCFS); all machines run in parallel, independently.

SCORE (minimize): total weight-times-completion-time, summed over every job
in the stream, using the REAL speed trajectory induced by the assignment
sequence. Assigning a job is therefore an INVESTMENT in that machine's future
condition, not just a queueing decision -- the innovation hook this problem
is built to reward is maintaining a deliberate PORTFOLIO of specialized
machine conditions (sacrifice one machine's speed on cheap abrasive filler,
protect/build another machine's speed for the valuable finishing work)
rather than spreading load (and hence wear) evenly across every machine.

CANDIDATE CONTRACT (isolated stdin -> stdout program, called ONCE PER JOB).
The evaluator interacts with the candidate program N times per instance (a
FRESH sandboxed subprocess each call -- the candidate is STATELESS between
calls; the public JSON it receives already contains every machine's current
observable state, so no history replay is required to act well).

  stdin (one call = one job arrival): ONE JSON object
    { "step": int, "n_jobs": int, "n_machines": int,
      "job": {"type": "A"|"F", "size": float, "weight": float},
      "seen_so_far": {"A": int, "F": int},   # counts of past job types
      "machines": [ {"id": int, "a": float, "q": float,
                      "spd": float, "free_at": float}, ... ] }

  stdout: ONE JSON object   { "assign": <int machine id> }
    - Must be an integer (or integer-valued float) in [0, n_machines).
    - A missing/out-of-range/non-numeric id, a crash, a timeout, or
      non-JSON output at ANY step voids the ENTIRE instance (score 0.0).

SCORING (deterministic; no wall-time). Per instance the evaluator computes,
by direct simulation of the N-step interaction:
    obj  = sum over all jobs of weight * completion_time, using the machine's
           REAL speed at the moment each job starts (free_at[m] before the
           job, plus size/spd[m]; then spd[m] updates per the job's type).
    weak = obj achieved by a FIXED "round robin across machines, ignore
           everything" reference policy (simulated directly, no candidate
           call) -- the naive non-adaptive recipe.
    lb   = a valid LOWER bound: replace every job's duration with its
           best-CASE value (size / SPD_MAX, the fastest any machine could
           ever run, since spd never exceeds SPD_MAX) and POOL the M
           machines into one preemptive machine of combined capacity M
           (strictly more powerful than M separate non-preemptive machines,
           since it can also "help" the same job with idle capacity that a
           real machine cannot borrow from a sibling); solved EXACTLY by
           Smith's ratio rule (WSPT). lb <= any online-achievable objective.
  normalized with an affine anchor (reproduce the round-robin recipe -> 0.1,
  reach the pooled relaxation -> 1.0):
    r = clamp( 0.1 + 0.9 * (weak - obj) / max(weak - lb, 1e-6), 0, 1 )

ISOLATION. Every job's candidate call runs in a FRESH sandboxed subprocess
via `isorun.run_candidate`; it only ever sees that job's public JSON. The
hidden job stream (future jobs), the weak baseline, and the pooled-relaxation
bound are all computed only in this parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun

SPD_MIN = 0.15
SPD_MAX = 6.0
STEP_TIMEOUT = 6      # seconds per per-job isorun call


# ----------------------------- deterministic RNG ----------------------------
def _rng(seed):
    state = seed & ((1 << 64) - 1)

    def unit():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)

    return unit


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


# ----------------------------- instance family -------------------------------
def _build_instances():
    """Deterministic instance family. Each machine's (a, q) pair is chosen so
    a HIGH abrasion sensitivity always comes paired with a HIGH polish
    sensitivity on the SAME machine (a "volatile" machine: dulls fast on
    abrasive, self-sharpens fast on finishing) versus a "stable" machine with
    LOW a and LOW q (barely changes either way) -- so no single machine is
    ever simultaneously the best abrasive-sink AND the best finishing-target,
    forcing a genuine multi-machine portfolio decision.

    shop301/302/303/304/305/321/322 are TRAP instances: the stream is either
    skewed so most jobs are cheap low-weight abrasive filler with a minority
    of expensive high-weight finishing jobs ("trap_lowweightA_highweightF"),
    or alternates in short bursts of several abrasive jobs then a couple of
    finishing jobs ("trap_bursts"). A dispatcher that reacts only to current
    queue length (shortest-queue / load-balancing, ignoring wear-affinity)
    spreads BOTH job types across every machine roughly evenly over time --
    no machine ever accumulates enough finishing history to become genuinely
    fast, and the abrasive filler dulls every machine a little instead of
    being concentrated on one machine that can absorb it cheaply. shop321 and
    shop322 are larger held-out variants of the same trap shape.
    shop311/312/313 are plain/mixed instances (uniform random 50/50 job-type
    stream; shop312 uses two IDENTICAL machines, where no specialization is
    possible at all) -- a sanity check that the ladder still orders
    sensibly when the trap structure is absent or degenerate.
    """
    specs = [
        dict(name="shop301", seed=301, M=2, N=22,
             machines=[dict(a=0.05, q=0.05), dict(a=0.24, q=0.24)],
             plan="trap_lowweightA_highweightF"),
        dict(name="shop302", seed=302, M=2, N=24,
             machines=[dict(a=0.06, q=0.06), dict(a=0.26, q=0.22)],
             plan="trap_lowweightA_highweightF"),
        dict(name="shop303", seed=303, M=3, N=26,
             machines=[dict(a=0.04, q=0.04), dict(a=0.24, q=0.24), dict(a=0.12, q=0.12)],
             plan="trap_lowweightA_highweightF"),
        dict(name="shop304", seed=304, M=2, N=24,
             machines=[dict(a=0.06, q=0.06), dict(a=0.22, q=0.22)],
             plan="trap_bursts"),
        dict(name="shop305", seed=305, M=3, N=27,
             machines=[dict(a=0.05, q=0.05), dict(a=0.20, q=0.22), dict(a=0.11, q=0.11)],
             plan="trap_bursts"),
        dict(name="shop311", seed=311, M=3, N=24,
             machines=[dict(a=0.08, q=0.08), dict(a=0.18, q=0.18), dict(a=0.12, q=0.12)],
             plan="mixed"),
        dict(name="shop312", seed=312, M=2, N=20,
             machines=[dict(a=0.10, q=0.10), dict(a=0.10, q=0.10)],
             plan="mixed"),
        dict(name="shop313", seed=313, M=3, N=22,
             machines=[dict(a=0.06, q=0.06), dict(a=0.20, q=0.19), dict(a=0.12, q=0.12)],
             plan="mixed"),
        dict(name="shop321", seed=321, M=3, N=30,
             machines=[dict(a=0.05, q=0.05), dict(a=0.24, q=0.24), dict(a=0.13, q=0.13)],
             plan="trap_lowweightA_highweightF"),
        dict(name="shop322", seed=322, M=2, N=28,
             machines=[dict(a=0.06, q=0.06), dict(a=0.35, q=0.35)],
             plan="trap_bursts"),
    ]
    out = []
    for sp in specs:
        u = _rng(sp["seed"])
        jobs = []
        N = sp["N"]
        plan = sp["plan"]
        if plan == "trap_lowweightA_highweightF":
            for _ in range(N):
                if u() < 0.65:
                    jobs.append(dict(type="A", size=round(2.0 + 3.0 * u(), 4),
                                     weight=round(1.0 + 1.0 * u(), 4)))
                else:
                    jobs.append(dict(type="F", size=round(4.0 + 4.0 * u(), 4),
                                     weight=round(6.0 + 6.0 * u(), 4)))
        elif plan == "trap_bursts":
            cycle = ["A", "A", "A", "A", "F", "F"]
            while len(jobs) < N:
                for t in cycle:
                    if len(jobs) >= N:
                        break
                    if t == "A":
                        jobs.append(dict(type="A", size=round(2.0 + 3.0 * u(), 4),
                                         weight=round(1.0 + 1.5 * u(), 4)))
                    else:
                        jobs.append(dict(type="F", size=round(4.0 + 5.0 * u(), 4),
                                         weight=round(5.0 + 7.0 * u(), 4)))
        else:  # mixed
            for _ in range(N):
                t = "A" if u() < 0.5 else "F"
                jobs.append(dict(type=t, size=round(2.0 + 5.0 * u(), 4),
                                 weight=round(1.0 + 6.0 * u(), 4)))
        out.append(dict(name=sp["name"], M=sp["M"], N=N, machines=sp["machines"], jobs=jobs))
    return out


# ----------------------------- simulation ------------------------------------
def _make_public(inst, spd, free, seen, step):
    return {"step": step, "n_jobs": inst["N"], "n_machines": inst["M"],
            "job": dict(inst["jobs"][step]),
            "seen_so_far": dict(seen),
            "machines": [{"id": m, "a": inst["machines"][m]["a"], "q": inst["machines"][m]["q"],
                          "spd": round(spd[m], 6), "free_at": round(free[m], 6)}
                         for m in range(inst["M"])]}


def _simulate(inst, get_answer):
    """Run the N-job interaction. `get_answer(public) -> (answer, ok)`.
    Returns the total weighted completion time, or None if ever invalid."""
    M, N = inst["M"], inst["N"]
    spd = [1.0] * M
    free = [0.0] * M
    seen = {"A": 0, "F": 0}
    total = 0.0
    for i in range(N):
        job = inst["jobs"][i]
        public = _make_public(inst, spd, free, seen, i)
        answer, ok = get_answer(public)
        if not ok or not isinstance(answer, dict):
            return None
        mid = answer.get("assign")
        if isinstance(mid, bool):
            return None
        if isinstance(mid, float):
            if mid != mid or mid in (float("inf"), float("-inf")) or not mid.is_integer():
                return None
            mid = int(mid)
        if not isinstance(mid, int):
            return None
        if mid < 0 or mid >= M:
            return None
        p, w, typ = job["size"], job["weight"], job["type"]
        proc = p / spd[mid]
        completion = free[mid] + proc
        total += w * completion
        free[mid] = completion
        if typ == "A":
            spd[mid] = _clip(spd[mid] * (1.0 - inst["machines"][mid]["a"]), SPD_MIN, SPD_MAX)
        else:
            spd[mid] = _clip(spd[mid] * (1.0 + inst["machines"][mid]["q"]), SPD_MIN, SPD_MAX)
        seen[typ] += 1
    return total


def _weak_get_answer(public):
    return {"assign": public["step"] % public["n_machines"]}, True


def _candidate_get_answer(cand):
    def fn(public):
        ans, st = isorun.run_candidate(cand, public, timeout=STEP_TIMEOUT)
        return ans, (st == "OK")
    return fn


def _lb_pooled(inst):
    """Valid lower bound: pool the M machines into one preemptive machine of
    combined capacity M, every job at its best-case duration size/SPD_MAX.
    Exactly optimal via Smith's ratio rule (WSPT)."""
    M = inst["M"]
    items = []
    for j in inst["jobs"]:
        e = (j["size"] / SPD_MAX) / M
        items.append((e, j["weight"]))
    items.sort(key=lambda x: (x[0] / x[1]))
    cum = 0.0
    total = 0.0
    for e, w in items:
        cum += e
        total += w * cum
    return total


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        weak = _simulate(inst, _weak_get_answer)
        if weak is None:
            weak = 0.0
        lb = _lb_pooled(inst)
        obj = _simulate(inst, _candidate_get_answer(cand))
        if obj is None:
            vec.append(0.0)
            continue
        denom = max(weak - lb, 1e-6)
        r = 0.1 + 0.9 * (weak - obj) / denom
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
