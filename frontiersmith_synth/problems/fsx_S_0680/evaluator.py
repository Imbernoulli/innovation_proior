#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0680 -- "Shifting Bottleneck: Sequence Every Machine in a Job
Shop" (family: shifting-bottleneck-dispatch; format B, quality-metric; theme: sequence
operations per machine in a job shop where the truly critical machine depends on the
whole routing).

THEME.  M machines, N jobs.  Each job is a fixed route of operations (each bound to one
machine, fixed duration); operations of a job run in route order (job precedence).
Operations sharing a machine run one at a time in an order the solver chooses.  The
solver outputs, for every machine, the processing order of its operations; the evaluator
combines this with the fixed job routes to build the full schedule and scores makespan.

OBJECTIVE (minimize, deterministic).  Given machine_order (one permutation per machine)
and the fixed job routes, build the combined precedence graph (job edges + given
machine-order edges).  If it has a cycle, or machine_order is malformed, score 0.
Otherwise every operation's start = max(finish of job-predecessor, finish of
machine-predecessor) (0 if neither), finish = start + dur; makespan = max finish.

WHY THE THREE MECHANISMS ARE FORCED.
  * one-machine-bottleneck-identify -- for machine m, given each of its operations'
    HEAD (earliest possible start under everything fixed so far) and TAIL (processing
    still required downstream after it), the one-machine relaxation (minimize
    max_i(completion_i + tail_i), i.e. 1|r_i|L_max) has a genuine bound: solve it (Schrage:
    among released ops, always run the largest-tail one next) for EVERY still-unsequenced
    machine; the machine with the LARGEST one-machine bound is the true bottleneck --
    total workload is a poor proxy (a lightly loaded machine can still be the bottleneck if
    its few operations are squeezed between long chains).
  * shifting-bottleneck-priority -- fix the identified bottleneck's order first (using its
    Schrage sequence), which changes every other machine's heads/tails, so RE-IDENTIFY the
    bottleneck among what remains and repeat; after all machines are fixed once, revisit
    each machine's order against the now-complete graph (the bottleneck ranking can shift
    again once later machines lock in).
  * critical-path-swap -- after sequencing, find the current longest chain through the
    schedule (the critical path); swapping two adjacent operations that sit on the SAME
    machine within that path can shorten it without touching any other machine's decisions.

INNOVATION HOOK (what `strong` exploits).  Solve each machine's one-machine subproblem
under current heads/tails, sequence that machine first, and reoptimize as the bottleneck
shifts.  TRAP instances hide the true bottleneck on a LOW total-load machine whose few
operations sit deep inside long, tightly-timed job chains, while decoy/filler machines
carry most of the total processing time without being tight anywhere.  A dispatch rule
that reacts to processing time alone (shortest-first list scheduling -- the standard
"obvious" JSP recipe, Giffler-Thompson active-schedule generation with an SPT tie-break)
has no notion of head/tail slack and frequently sequences the pivotal machine's operations
in the wrong order, inflating makespan on exactly these instances.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n_jobs": J, "n_machines": M, "n_ops": K,
             "ops": [{"id","job","pos","machine","dur"}, ...],
             "job_ops": [[id,...], ...], "machine_ops": [[id,...], ...]}
  stdout: ONE JSON object:
            {"machine_order": [[id,...], ...]}   # M lists

  VALID iff machine_order has length M, each machine_order[m] is a permutation of
  machine_ops[m] (no missing/duplicate/foreign ids), and the combined precedence graph
  (job edges + machine-order edges) is acyclic.  Any violation, crash, timeout, or
  non-JSON -> 0.0 on that instance.

SCORING (deterministic; no wall-time).  Per instance:
    q_base = makespan when every machine orders its ops by ascending id (job index, then
             position) -- a naive fixed-priority reference, computed by THIS evaluator.
    q_lb   = max(busiest machine's total load, longest job's total duration)
    q_cand = candidate's makespan (0 penalty if infeasible)
    r = clamp( 0.1 + 0.75 * (q_base - q_cand) / max(q_base - q_lb, 1e-9), 0, 1 )
  The id-order schedule scores exactly 0.1.  q_lb is a relaxation that ignores all
  cross-machine contention and is never reachable, so headroom stays open above any
  solution.  Final score = mean of r over 10 fixed seeded instances (4 bottleneck traps +
  3 mixed random job-shops + 3 dense/held-out instances with two competing bottlenecks).

ISOLATION.  The candidate is untrusted and runs in a FRESH bwrap-SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  q_base, q_lb and q_cand are
all computed by THIS parent process.

CLI:  python3 evaluator.py <solution.py>
"""
import sys, json
from collections import deque
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


# ----------------------------- instance assembly ----------------------------
def _assemble(name, job_routes):
    """job_routes: list of jobs, each a list of (machine, dur).  Assigns global op ids
    job-major (job, then position) -- this order is also the `trivial` reference."""
    ops = []
    job_ops = []
    oid = 0
    for j, route in enumerate(job_routes):
        ids = []
        for p, (m, d) in enumerate(route):
            ops.append({"id": oid, "job": j, "pos": p, "machine": m, "dur": int(d)})
            ids.append(oid)
            oid += 1
        job_ops.append(ids)
    n_machines = 1 + max(o["machine"] for o in ops)
    machine_ops = [[] for _ in range(n_machines)]
    for o in ops:
        machine_ops[o["machine"]].append(o["id"])
    return {"name": name, "n_jobs": len(job_routes), "n_machines": n_machines,
            "n_ops": len(ops), "ops": ops, "job_ops": job_ops, "machine_ops": machine_ops}


def _build_trap(seed, n_pairs, n_filler_jobs, filler_machines,
                 hdur, dA, dB, tailA, tailB1, tailB2):
    """TRAP: n_pairs conflict pairs share a single BOTTLENECK machine (index 0).  Job A of
    a pair reaches the bottleneck with head H and a SHORT tail after it; job B reaches it
    with the SAME head H but a LONGER bottleneck duration and a LONG tail after it.
    Shortest-processing-time dispatch runs A first (it is shorter), delaying B and
    stretching B's long tail; the largest-tail-first rule runs B first instead.  Filler
    jobs live only on separate decoy machines and inflate their total load far above the
    bottleneck's, so "busiest machine" never points at machine 0."""
    nx = _rng(seed)
    job_routes = []
    next_machine = 1
    bott = 0
    for _p in range(n_pairs):
        H = nx(*hdur)
        mHA, mTA, mHB, mTB1, mTB2 = range(next_machine, next_machine + 5)
        next_machine += 5
        da, db = nx(*dA), nx(*dB)
        ta, tb1, tb2 = nx(*tailA), nx(*tailB1), nx(*tailB2)
        job_routes.append([(mHA, H), (bott, da), (mTA, ta)])
        job_routes.append([(mHB, H), (bott, db), (mTB1, tb1), (mTB2, tb2)])
    fm0 = next_machine
    for _f in range(n_filler_jobs):
        k = nx(2, 4)
        route = [(fm0 + nx(0, filler_machines - 1), nx(8, 20)) for _ in range(k)]
        job_routes.append(route)
    return _assemble("trap", job_routes)


def _build_mixed(seed, n_jobs, n_machines, ops_min, ops_max, dmin, dmax):
    """MIXED: classic random job-shop (each job visits a random subset of machines in
    random order, random durations) -- no planted trap, keeps the ladder honest."""
    nx = _rng(seed)
    job_routes = []
    for _j in range(n_jobs):
        k = min(nx(ops_min, ops_max), n_machines)
        avail = list(range(n_machines))
        route = []
        for _ in range(k):
            idx = nx(0, len(avail) - 1)
            m = avail.pop(idx)
            route.append((m, nx(dmin, dmax)))
        job_routes.append(route)
    return _assemble("mixed", job_routes)


def _build_dense(seed, n_pairs, n_filler_jobs, filler_machines):
    """DENSE / held-out: TWO independent bottleneck machines (0 and 1), each with its own
    conflict pairs, plus filler noise -- tests that the shift keeps re-identifying the
    right machine across more than one round, not just once."""
    nx = _rng(seed)
    job_routes = []
    next_machine = 2
    for bott in (0, 1):
        for _p in range(n_pairs):
            H = nx(18, 30)
            mHA, mTA, mHB, mTB1, mTB2 = range(next_machine, next_machine + 5)
            next_machine += 5
            da, db = nx(3, 7), nx(14, 22)
            ta, tb1, tb2 = nx(2, 5), nx(10, 16), nx(10, 16)
            job_routes.append([(mHA, H), (bott, da), (mTA, ta)])
            job_routes.append([(mHB, H), (bott, db), (mTB1, tb1), (mTB2, tb2)])
    fm0 = next_machine
    for _f in range(n_filler_jobs):
        k = nx(2, 4)
        route = [(fm0 + nx(0, filler_machines - 1), nx(8, 20)) for _ in range(k)]
        job_routes.append(route)
    return _assemble("dense", job_routes)


def _build_instances():
    specs = [
        ("trap1", _build_trap, dict(seed=90000, n_pairs=3, n_filler_jobs=5, filler_machines=3,
                                     hdur=(18, 30), dA=(3, 7), dB=(14, 22),
                                     tailA=(2, 5), tailB1=(10, 16), tailB2=(10, 16))),
        ("trap2", _build_trap, dict(seed=90001, n_pairs=3, n_filler_jobs=5, filler_machines=3,
                                     hdur=(18, 30), dA=(3, 7), dB=(14, 22),
                                     tailA=(2, 5), tailB1=(10, 16), tailB2=(10, 16))),
        ("trap3", _build_trap, dict(seed=90002, n_pairs=3, n_filler_jobs=5, filler_machines=3,
                                     hdur=(18, 30), dA=(3, 7), dB=(14, 22),
                                     tailA=(2, 5), tailB1=(10, 16), tailB2=(10, 16))),
        ("trap4", _build_trap, dict(seed=90003, n_pairs=3, n_filler_jobs=5, filler_machines=3,
                                     hdur=(18, 30), dA=(3, 7), dB=(14, 22),
                                     tailA=(2, 5), tailB1=(10, 16), tailB2=(10, 16))),
        ("mixed1", _build_mixed, dict(seed=91000, n_jobs=8, n_machines=5, ops_min=3, ops_max=5,
                                       dmin=5, dmax=25)),
        ("mixed2", _build_mixed, dict(seed=91001, n_jobs=8, n_machines=5, ops_min=3, ops_max=5,
                                       dmin=5, dmax=25)),
        ("mixed3", _build_mixed, dict(seed=91002, n_jobs=8, n_machines=5, ops_min=3, ops_max=5,
                                       dmin=5, dmax=25)),
        ("dense1", _build_dense, dict(seed=92001, n_pairs=2, n_filler_jobs=6, filler_machines=4)),
        ("dense2", _build_dense, dict(seed=92004, n_pairs=2, n_filler_jobs=6, filler_machines=4)),
        ("dense3", _build_dense, dict(seed=92008, n_pairs=2, n_filler_jobs=6, filler_machines=4)),
    ]
    out = []
    for name, fn, kw in specs:
        inst = fn(**kw)
        inst["name"] = name
        out.append(inst)
    return out


# ----------------------------- simulation -----------------------------------
def _simulate(inst, machine_order):
    """Strict feasibility check + forward (Kahn's toposort) simulation.  Returns
    (feasible: bool, makespan: float|None)."""
    n = inst["n_ops"]
    if not isinstance(machine_order, list) or len(machine_order) != inst["n_machines"]:
        return False, None
    dur = [o["dur"] for o in inst["ops"]]
    job_ops = inst["job_ops"]
    machine_ops = inst["machine_ops"]
    pred = [[] for _ in range(n)]
    for ids in job_ops:
        for a, b in zip(ids, ids[1:]):
            pred[b].append(a)
    for m in range(inst["n_machines"]):
        seq = machine_order[m]
        if not isinstance(seq, list):
            return False, None
        for x in seq:
            if isinstance(x, bool) or not isinstance(x, int) or x < 0 or x >= n:
                return False, None
        if sorted(seq) != sorted(machine_ops[m]):
            return False, None
        for a, b in zip(seq, seq[1:]):
            pred[b].append(a)
    succ = [[] for _ in range(n)]
    for v in range(n):
        for p in pred[v]:
            succ[p].append(v)
    indeg = [len(pred[i]) for i in range(n)]
    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != n:
        return False, None  # cycle: machine order contradicts job routes
    finish = [0.0] * n
    for u in order:
        st = max((finish[p] for p in pred[u]), default=0.0)
        finish[u] = st + dur[u]
    return True, max(finish)


def _reference_order(inst):
    return [sorted(inst["machine_ops"][m]) for m in range(inst["n_machines"])]


def _lower_bound(inst):
    n_m, n_j = inst["n_machines"], inst["n_jobs"]
    mload = [0] * n_m
    jload = [0] * n_j
    for o in inst["ops"]:
        mload[o["machine"]] += o["dur"]
        jload[o["job"]] += o["dur"]
    return float(max(max(mload), max(jload)))


GAIN = 0.75


def _score(inst, answer):
    if not isinstance(answer, dict):
        return False, 0.0
    ok, ms = _simulate(inst, answer.get("machine_order"))
    if not ok:
        return False, 0.0
    return True, float(ms)


# ----------------------------- driver -------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        ok0, q_base = _simulate(inst, _reference_order(inst))
        assert ok0, "reference construction must always be feasible"
        q_lb = _lower_bound(inst)
        denom = max(q_base - q_lb, 1e-9)

        public = {"name": inst["name"], "n_jobs": inst["n_jobs"], "n_machines": inst["n_machines"],
                  "n_ops": inst["n_ops"], "ops": [dict(o) for o in inst["ops"]],
                  "job_ops": [list(x) for x in inst["job_ops"]],
                  "machine_ops": [list(x) for x in inst["machine_ops"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, q_cand = _score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        r = 0.1 + GAIN * (q_base - q_cand) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        vec.append(max(0.0, min(1.0, r)))

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
