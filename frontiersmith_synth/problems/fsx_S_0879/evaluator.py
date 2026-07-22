#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0879 -- "Foundry Order Book: Deadline-Weighted Jobs with
Family Setup Switching" (family: family-setup-batch-scheduler; format B, quality-metric;
theme: order deadline-weighted jobs with family setup times to maximize on-time
weighted throughput).

THEME.  A single machine processes jobs one at a time.  Every job i has a processing
time p[i], a deadline d[i], a weight w[i], and a family tag fam[i].  Whenever the
machine switches from a job of family a to a job of a DIFFERENT family b, it pays a
family-PAIR-dependent setup time setup[a][b] (the matrix is not necessarily symmetric
and lives only in the instance -- a "hidden" cost table the solver must read, not a
constant the statement can spell out).  Consecutive jobs of the SAME family pay no
setup at all.  A job is admitted (and earns its weight) only if the machine's clock,
after that job's setup (if any) and its own processing time, is still <= its deadline;
otherwise the job is simply never started (skipped, no time charged) -- there is no
partial credit and no penalty beyond forfeiting its weight.

MECHANISM 1 -- batch-aware sequencing.  Because same-family runs are setup-free, the
achievable throughput of an ordering depends on HOW MUCH the family sequence
fragments: an ordering that keeps a family's jobs together ("batches") amortizes one
setup over many jobs, while an ordering that hops families every step pays a setup on
almost every job.

MECHANISM 2 -- family-clustering diagnosis.  Which families are cheap to sit next to,
and which family's jobs are collectively worth visiting early vs. late, is instance-
specific (encoded in setup[][], fam[], and the deadline distribution) -- a solver has
to DIAGNOSE this from the input, not assume any fixed family order.

MECHANISM 3 -- reinsertion local move.  Clustering an ENTIRE family as one contiguous
block is sometimes itself a trap: a family can contain one very high-weight job with a
tight deadline stranded among many low-weight, loose-deadline packmates. Visiting the
whole family block early rescues the stranded job but wastes time on its low-value
packmates that could have waited; visiting the block at its "natural" slot loses the
stranded job entirely. The fix is to PULL that single job OUT of its family's block,
insert it elsewhere in the sequence (paying one extra setup) to catch its deadline, and
revisit the rest of the family later -- a local reinsertion move, not a block decision.

INNOVATION HOOK (what `strong` exploits).  Cluster by family to amortize setups, then
interleave the family batches by a deadline-versus-setup trade-off (visit whichever
family's next pending job is most urgent, batch-consume as much of that family's queue
as still meets its deadlines, then switch), and finally run single-job reinsertion
moves to rescue high-weight jobs stranded between families.  A pure earliest-deadline-
first (EDF) pass ignores family structure entirely and thrashes setups; a pure "cluster
every family into one block" pass cannot rescue a job stranded inside a low-priority
family without wasting time on that family's low-value packmates.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": int, "F": int,
             "p":   [p_0 ... p_{n-1}],           # processing times (positive ints)
             "d":   [d_0 ... d_{n-1}],           # deadlines (positive ints)
             "w":   [w_0 ... w_{n-1}],           # weights (positive numbers)
             "fam": [fam_0 ... fam_{n-1}],       # family id in [0, F) per job
             "setup": [[setup[a][b] for b in range(F)] for a in range(F)]}  # setup[a][a]=0
  stdout: ONE JSON object:
            {"order": [i_0, i_1, ...]}           # a sequence of DISTINCT job indices in
                                                  # [0, n) -- the processing order. May
                                                  # be a proper subset of {0..n-1}.

  VALID iff "order" is a list of distinct integers (or integer-valued numbers), each in
  [0, n).  Any violation (wrong type, duplicate, out-of-range, NaN/inf, bool, crash,
  timeout, non-JSON) -> 0.0 on that instance.

TRANSITION (deterministic admission control).  t=0, prev_family=None, gained=0.
For each index idx in "order", in order:
  su = 0 if prev_family is None or prev_family == fam[idx] else setup[prev_family][fam[idx]]
  tentative = t + su + p[idx]
  if tentative <= d[idx]: t = tentative; prev_family = fam[idx]; gained += w[idx]   # admit
  else: skip entirely -- t and prev_family are UNCHANGED, job earns nothing.
A job is never partially started and a missed job never advances the clock -- so a
solver can freely propose jobs it is unsure about; only genuinely-infeasible-right-now
jobs are auto-skipped.

SCORING (deterministic; no wall-time).  Per instance the evaluator computes:
    relax = sum(w[i] for i in range(n) if p[i] <= d[i])
  This is a mathematically valid UPPER BOUND on any achievable admitted weight (any
  admitted job's completion time is >= its own processing time, so p[i] <= d[i] is
  NECESSARY for admission under ANY order/subset) -- and it deliberately over-counts:
  every instance plants at least two "conflict pairs" of jobs (different families, each
  individually feasible, i.e. each with p[i] <= d[i]) that are mutually exclusive --
  scheduling one after the other in either order blows the second one's deadline -- so
  `relax` counts both while no real order can ever bank both, guaranteeing headroom
  above any achievable schedule.
    r = clamp(gained / relax, 0, 1)
  The final score is the mean of r over 10 fixed seeded instances (interleaved-deadline
  traps that punish family-blind EDF, stranded-high-value traps that punish whole-
  family blocking, and larger mixed/held-out instances for generalization).

ISOLATION.  The candidate is untrusted and runs in a FRESH bwrap-SANDBOXED SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC instance above.

CLI:  python3 evaluator.py <solution.py>
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


def _make_setup(F, seed, lo=7, hi=12):
    nxt = _rng(seed)
    M = [[0.0] * F for _ in range(F)]
    for a in range(F):
        for b in range(F):
            if a != b:
                M[a][b] = float(nxt(lo, hi))
    return M


def _add_conflict_pairs(jobs, F, setup, nxt, n_pairs=2):
    """Append pairs of jobs (different families) that are each individually feasible
    (p<=d) but jointly infeasible in either processing order -- guarantees the
    relaxation bound over-counts achievable weight, so `strong` cannot saturate."""
    for _ in range(n_pairs):
        fa = nxt(0, F - 1)
        fb = fa
        while fb == fa:
            fb = nxt(0, F - 1)
        pa = nxt(4, 7)
        pb = nxt(4, 7)
        su_ab = setup[fa][fb]
        su_ba = setup[fb][fa]
        # Use the CHEAPER direction's setup so BOTH processing orders are infeasible
        # for the pair (min, not max: whichever order is cheaper must still miss the
        # shared deadline, which automatically makes the costlier order miss it too).
        tightD = pa + min(su_ab, su_ba) + pb - 1
        tightD = max(tightD, pa + 1, pb + 1)
        wa = nxt(30, 50)
        wb = nxt(30, 50)
        jobs.append({"fam": fa, "p": pa, "d": tightD, "w": wa})
        jobs.append({"fam": fb, "p": pb, "d": tightD, "w": wb})


def _pack_and_shuffle(jobs, F, setup, seed):
    n = len(jobs)
    nxt = _rng(seed)
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = nxt(0, i)
        perm[i], perm[j] = perm[j], perm[i]
    fam = [jobs[perm[i]]["fam"] for i in range(n)]
    p = [jobs[perm[i]]["p"] for i in range(n)]
    d = [jobs[perm[i]]["d"] for i in range(n)]
    w = [jobs[perm[i]]["w"] for i in range(n)]
    return {"n": n, "F": F, "fam": fam, "p": p, "d": d, "w": w, "setup": setup}


def _build_interleave(seed, F=4, m=6, step=5, base=20):
    """Trap topology: family assignment cycles 0..F-1 while deadlines strictly
    increase with rank, so earliest-deadline-first (EDF) visits a different family on
    almost every step and pays a setup almost every step. Clustering by family instead
    amortizes to F-1 setups total."""
    nxt = _rng(seed)
    jobs = []
    for k in range(F * m):
        jobs.append({"fam": k % F, "p": nxt(3, 6), "d": base + k * step, "w": nxt(5, 15)})
    setup = _make_setup(F, seed * 7 + 1)
    _add_conflict_pairs(jobs, F, setup, _rng(seed * 11 + 9), n_pairs=2)
    return _pack_and_shuffle(jobs, F, setup, seed * 13 + 5)


def _build_stranded(seed, F=4, m=5, horizon=110):
    """Trap topology: most jobs have loose deadlines clustered near `horizon`, but one
    high-weight job is stranded with a tight deadline inside an otherwise low-priority
    family, and 3 low-weight DECOY jobs (one per other family) have even tighter
    deadlines. EDF chases the decoys across families first (thrashing setups) and
    reaches the stranded job too late; a family-block-only scheduler that visits the
    stranded job's whole family early rescues it but wastes time on its low-value
    packmates. Only pulling the single stranded job out via reinsertion is efficient."""
    nxt = _rng(seed)
    jobs = []
    for i in range(F * m):
        jobs.append({"fam": i // m, "p": nxt(3, 6), "d": horizon - nxt(0, 20), "w": nxt(5, 15)})
    setup = _make_setup(F, seed * 7 + 2)
    strand_fam = min(2, F - 1)
    strand_idx = strand_fam * m
    jobs[strand_idx]["d"] = 30
    jobs[strand_idx]["w"] = 60
    decoy_fams = [f for f in range(F) if f != strand_fam]
    decoy_deadlines = [14, 18, 22]
    for k, f in enumerate(decoy_fams[:3]):
        job_i = f * m + 1
        jobs[job_i]["d"] = decoy_deadlines[k % len(decoy_deadlines)]
        jobs[job_i]["w"] = 4
    _add_conflict_pairs(jobs, F, setup, _rng(seed * 11 + 10), n_pairs=2)
    return _pack_and_shuffle(jobs, F, setup, seed * 13 + 6)


def _build_mixed(seed, F=5, m=6, step=6):
    """Held-out / generalization topology: family assignment is UNCORRELATED with
    deadline order (random per job), larger n and F, plus two extra random high-weight
    tight-deadline jobs -- a broader stress test of the same trade-off."""
    nxt = _rng(seed)
    jobs = []
    for k in range(F * m):
        jobs.append({"fam": nxt(0, F - 1), "p": nxt(3, 7), "d": 30 + k * step + nxt(-5, 5), "w": nxt(4, 20)})
    for _ in range(2):
        j = nxt(0, len(jobs) - 1)
        jobs[j]["d"] = nxt(15, 40)
        jobs[j]["w"] = nxt(45, 65)
    setup = _make_setup(F, seed * 7 + 3)
    _add_conflict_pairs(jobs, F, setup, _rng(seed * 11 + 11), n_pairs=2)
    return _pack_and_shuffle(jobs, F, setup, seed * 13 + 7)


def _build_instances():
    out = []
    specs = [
        ("interleave1", "interleave", 100, 4, 6, 5, 20),
        ("interleave2", "interleave", 101, 4, 6, 5, 20),
        ("interleave3", "interleave", 102, 4, 6, 5, 20),
        ("interleave4", "interleave", 103, 4, 6, 5, 20),
        ("stranded1", "stranded", 200, 4, 5, 110, None),
        ("stranded2", "stranded", 201, 4, 5, 110, None),
        ("stranded3", "stranded", 202, 4, 5, 110, None),
        ("mixed1", "mixed", 300, 5, 6, 6, None),
        ("mixed2", "mixed", 301, 5, 6, 6, None),
        ("mixed3", "mixed", 302, 5, 6, 6, None),
    ]
    for name, kind, seed, F, m, a, b in specs:
        if kind == "interleave":
            inst = _build_interleave(seed, F=F, m=m, step=a, base=b)
        elif kind == "stranded":
            inst = _build_stranded(seed, F=F, m=m, horizon=a)
        else:
            inst = _build_mixed(seed, F=F, m=m, step=a)
        inst["name"] = name
        out.append(inst)
    return out


# ----------------------------- transition / scoring -------------------------
def simulate(inst, order):
    """Replay the deterministic admission-control transition. Return total admitted
    weight, or None if `order` is malformed (out-of-range / duplicate index)."""
    n = inst["n"]
    fam = inst["fam"]
    p = inst["p"]
    d = inst["d"]
    w = inst["w"]
    setup = inst["setup"]
    seen = set()
    t = 0.0
    prev = None
    gained = 0.0
    for idx in order:
        if not (0 <= idx < n) or idx in seen:
            return None
        seen.add(idx)
        f = fam[idx]
        su = 0.0 if prev is None or prev == f else setup[prev][f]
        tentative = t + su + p[idx]
        if tentative <= d[idx] + 1e-9:
            t = tentative
            prev = f
            gained += w[idx]
    return gained


def relax_bound(inst):
    return sum(inst["w"][i] for i in range(inst["n"]) if inst["p"][i] <= inst["d"][i] + 1e-9)


# ----------------------------- validation ------------------------------------
def _valid_order(inst, answer):
    """Return a list of distinct valid job indices, or None if malformed."""
    if not isinstance(answer, dict):
        return None
    order = answer.get("order")
    if not isinstance(order, list):
        return None
    n = inst["n"]
    out = []
    seen = set()
    for v in order:
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            iv = v
        elif isinstance(v, float):
            if v != v or v in (float("inf"), float("-inf")) or v != int(v):
                return None
            iv = int(v)
        else:
            return None
        if not (0 <= iv < n) or iv in seen:
            return None
        seen.add(iv)
        out.append(iv)
    return out


# ----------------------------- scoring driver ---------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        rb = relax_bound(inst)
        if rb <= 1e-9:
            rb = 1e-9
        public = {"name": inst["name"], "n": inst["n"], "F": inst["F"],
                  "p": list(inst["p"]), "d": list(inst["d"]), "w": list(inst["w"]),
                  "fam": list(inst["fam"]), "setup": [list(row) for row in inst["setup"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            order = _valid_order(inst, ans)
            if order is None:
                vec.append(0.0)
                continue
            gained = simulate(inst, order)
        except Exception:
            vec.append(0.0)
            continue
        if gained is None:
            vec.append(0.0)
            continue
        r = gained / rb
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
