#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0471 -- "Staged Trainer: Curriculum Ordering for
Fastest Convergence"  (family: ml-curriculum-order; format B, quality-metric).

THEME.  A fixed toy "student" model is trained on a corpus of labelled training
examples.  Every example teaches ONE atomic *concept*; a concept only becomes
learnable once its *prerequisite* concepts are already partly mastered (a concept
dependency DAG -- roots have no prerequisites, deeper concepts stack on shallower
ones).  You, the trainer, decide the CURRICULUM: the exact sequence in which
examples are shown to the model (repetition allowed -- you may re-show an example
many times, like re-visiting a topic across epochs).  The model runs one
gradient-style update per shown example.  Training STOPS the moment the model's
loss first drops to the target.  You want to reach the target in as FEW updates
as possible.  Show a deep-concept example before its prerequisites are mastered
and the update is nearly wasted; front-load the foundations and the whole
dependency chain lights up quickly.  This is curriculum learning as a scheduling
problem.

DETERMINISTIC TRAINING DYNAMICS (the frozen "student").  There are K concepts,
each with a mastery m_k in [0,1], all starting at 0.  Showing an example that
teaches concept c with prerequisite set P applies ONE update:

    readiness = 1.0                     if P is empty
              = min(m_p for p in P)     otherwise
    m_c      += LR * readiness * (1 - m_c)

so an update on a concept whose prerequisites are unmastered barely moves m_c
(readiness ~ 0), while a well-prepared concept jumps toward mastery with
diminishing returns.  The training LOSS after each update is the mean immaturity:

    loss = (1/K) * sum_k (1 - m_k)

Training halts at the first update index t (1-based) with loss <= target; that
index is the candidate's convergence step count q_cand.  If the schedule runs out
(or the cap is hit) before reaching the target, the instance scores 0.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "K": int, "n_examples": N, "LR": float,
             "target": float, "cap": int,
             "examples": [ {"concept": c, "prereqs": [ ... ], "layer": l},
                           ...   # example id == list index, length N ]}
  stdout: ONE JSON object:
            {"schedule": [e_0, e_1, ...]}    # each e_t is an example id in [0,N-1];
                                             # length 1..cap; repetition allowed.
  The model is shown examples[e_0], examples[e_1], ...  A schedule is VALID iff it
  is a non-empty list of at most `cap` integers, each a valid example id in
  [0, N-1].  A non-list, wrong element type/range, over-length, crash, timeout, or
  non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance the evaluator itself computes
three convergence-step references under the SAME frozen dynamics:
    q_lb   = optimistic lower bound: ignore prerequisites (readiness == 1 always)
             and greedily update the least-mastered concept each step.  Since real
             readiness <= 1 and this picks the largest possible per-step gain, NO
             real curriculum can beat it -> an unreachable ideal.
    q_base = the natural "as-shipped" order cycled: show examples 0,1,...,N-1,
             0,1,... in the (shuffled) order they arrive.  A weak reference.
    q_cand = the candidate schedule's convergence step count.
  and normalize with an affine anchor (weak baseline -> 0.1, ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
  Reproducing the as-shipped order scores ~0.1; a schedule reaching the (generally
  unreachable) optimistic bound scores 1.0; doing worse than the natural order
  scores < 0.1.  Because q_lb ignores the dependency tax that every real schedule
  must pay, even strong simulation-driven curricula stay well below 1.0 -> headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(q_lb, q_base) and the authoritative training simulation are computed by THIS
parent process, so a frame-walking / introspecting candidate learns nothing it
could not compute itself -- and cannot inject a fake step count.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
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
def _build(seed, L, W, reps):
    """Deterministic staged-trainer instance: L layers x W concepts, `reps`
    training examples per concept.  Returns (K, examples) where examples is the
    already-shuffled 'as-shipped' corpus (example id == list index)."""
    ni = _rng(seed)
    K = L * W
    layer = [l for l in range(L) for _ in range(W)]
    parents = [[] for _ in range(K)]
    for c in range(K):
        l = layer[c]
        if l == 0:
            continue
        prev = [x for x in range(K) if layer[x] == l - 1]
        npar = 1 + (1 if ni(0, 99) < 45 else 0)          # sometimes 2 prerequisites
        chosen = set()
        for _ in range(npar):
            chosen.add(prev[ni(0, len(prev) - 1)])
        parents[c] = sorted(chosen)
    examples = []
    for c in range(K):
        for _ in range(reps):
            examples.append({"concept": c, "prereqs": parents[c], "layer": layer[c]})
    N = len(examples)
    idx = list(range(N))
    for i in range(N - 1, 0, -1):                        # deterministic shuffle
        j = ni(0, i)
        idx[i], idx[j] = idx[j], idx[i]
    examples = [examples[i] for i in idx]
    return K, examples


def _build_instances():
    """Deterministic instance family. (seed, L, W, reps).  Deeper L = harder /
    held-out (longer prerequisite chains reward foundation-first curricula more)."""
    specs = [
        (101, 4, 5, 3),
        (102, 5, 4, 3),
        (103, 3, 6, 3),
        (104, 6, 4, 2),
        (105, 4, 6, 3),
        (106, 5, 5, 2),
        # harder / held-out: deeper prerequisite chains
        (107, 6, 5, 2),
        (108, 7, 4, 2),
        (109, 5, 6, 2),
        (110, 8, 4, 2),
    ]
    LR = 0.30
    TARGET = 0.15
    out = []
    for seed, L, W, reps in specs:
        K, examples = _build(seed, L, W, reps)
        N = len(examples)
        cap = 45 * N
        out.append({"name": f"trainer{seed}", "K": K, "n_examples": N,
                    "LR": LR, "target": TARGET, "cap": cap, "examples": examples})
    return out


# ----------------------------- frozen dynamics -----------------------------
def _simulate(K, examples, schedule, LR, target, cap):
    """Run the frozen student on `schedule`; return 1-based convergence step, or
    None if the target is never reached within the schedule/cap."""
    m = [0.0] * K
    for t, idx in enumerate(schedule):
        if t >= cap:
            break
        ex = examples[idx]
        c = ex["concept"]
        P = ex["prereqs"]
        readiness = 1.0 if not P else min(m[p] for p in P)
        m[c] += LR * readiness * (1.0 - m[c])
        loss = sum(1.0 - x for x in m) / K
        if loss <= target:
            return t + 1
    return None


def _q_base(K, examples, LR, target, cap):
    N = len(examples)
    sched = [i % N for i in range(cap)]
    return _simulate(K, examples, sched, LR, target, cap)


def _q_lb(K, examples, LR, target, cap):
    """Optimistic lower bound: ignore prerequisites (readiness == 1) and always
    update the least-mastered concept.  Unreachable by any real schedule."""
    m = [0.0] * K
    for t in range(cap):
        c = min(range(K), key=lambda k: m[k])
        m[c] += LR * (1.0 - m[c])
        loss = sum(1.0 - x for x in m) / K
        if loss <= target:
            return t + 1
    return None


# ----------------------------- validation ----------------------------------
def _q_cand(inst, answer):
    """Validate the candidate schedule, then simulate it. Return convergence step
    count, or None on any invalidity / non-convergence."""
    if not isinstance(answer, dict):
        return None
    sched = answer.get("schedule")
    if not isinstance(sched, list) or not sched:
        return None
    N = inst["n_examples"]
    cap = inst["cap"]
    if len(sched) > cap:
        return None
    for e in sched:
        if isinstance(e, bool) or not isinstance(e, int):
            return None
        if e < 0 or e >= N:
            return None
    return _simulate(inst["K"], inst["examples"], sched, inst["LR"],
                     inst["target"], cap)


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        K = inst["K"]
        examples = inst["examples"]
        LR = inst["LR"]
        target = inst["target"]
        cap = inst["cap"]
        q_lb = _q_lb(K, examples, LR, target, cap)
        q_base = _q_base(K, examples, LR, target, cap)
        if q_lb is None or q_base is None:
            # should not happen for this family; guard anyway
            vec.append(0.0)
            continue
        denom = q_base - q_lb
        if denom < 1e-9:
            denom = 1e-9

        public = {"name": inst["name"], "K": K, "n_examples": inst["n_examples"],
                  "LR": LR, "target": target, "cap": cap,
                  "examples": [{"concept": e["concept"],
                                "prereqs": list(e["prereqs"]),
                                "layer": e["layer"]} for e in examples]}

        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            qc = _q_cand(inst, ans)
        except Exception:
            qc = None
        if qc is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (q_base - qc) / denom
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
