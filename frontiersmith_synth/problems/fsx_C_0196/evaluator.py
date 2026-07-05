#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0196 -- "Rooftop Garden Trellis: Inducing a Nesting Rule
that Generalizes to Taller Trellises"
(family: synthetic-algorithmic-generalization; format B, correctness/exact-match).

THEME.  A vertical-farming co-op grows plants on rooftop *trellises*.  A trellis is a
stack of planter MODULES read bottom-to-top; each module is one of `n_types` symbol
kinds (think: which bracket / which planter shape).  Every plant VARIETY thrives only
on trellises whose module arrangement obeys a hidden *nesting rule* -- a regular
(finite-state) predicate over the module string.  The co-op has measured, for a batch
of SHORT trellises, whether the variety thrived (label 1) or wilted (label 0).  They
now want to predict thriving on TALLER trellises they have not yet built.

THE ALGORITHMIC-GENERALIZATION TASK.  The hidden rule is a randomly generated
deterministic finite automaton (DFA) over the module alphabet.  You are given labeled
SHORT trellises (in-distribution, ID).  You must output a DFA (your induced rule) that
the evaluator runs on HIDDEN test trellises: half are fresh SHORT trellises (ID
generalization) and half are much TALLER trellises (length-OOD generalization).  A rule
that merely MEMORIZES the training trellises scores near the majority baseline; a rule
that recovers the underlying finite-state STRUCTURE generalizes to the taller,
never-seen lengths.  This is the ID-vs-length-OOD split that rewards true generalization
over memorization.  The hidden rule is randomized per instance, so it cannot be
hand-coded -- it must be learned from the examples.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "n_types": D (int >=2),          # module alphabet: symbols 0..D-1
             "max_states": M (int),           # your DFA may use at most M states
             "train": [[seq, label], ...]}    # seq = list of ints in [0,D); label in {0,1}
  stdout: ONE JSON object describing a COMPLETE deterministic finite automaton:
            {"start": s0,                      # int in [0, K)
             "accept": [a_0,...,a_{K-1}],      # each 0/1 ; a_i=1 => state i is accepting
             "trans": [[t_{0,0},...,t_{0,D-1}],# K x D ; t_{s,c} in [0,K) = next state
                       ... K rows ...]}
          Let K = len(trans) = len(accept) be the number of states.  The DFA reads a
          trellis left-to-right from `start`, following trans[state][symbol]; the
          trellis is predicted "thrives" (1) iff the final state is accepting.

  A submission is VALID iff: 1 <= K <= max_states; `accept` is a length-K list of 0/1;
  `trans` is a K-row list, every row length D, every entry an int in [0,K); `start` is
  an int in [0,K).  Any malformed submission, a crash, a timeout, or non-JSON output ->
  that instance scores 0.0.

SCORING (deterministic; no wall-time, no training).  For each instance the evaluator
holds two HIDDEN labeled test sets: `id_test` (fresh short trellises, disjoint from
train) and `ood_test` (much taller trellises).  It runs the submitted DFA on both:
    id_acc  = exact-match accuracy on id_test
    ood_acc = exact-match accuracy on ood_test
    obj     = gmean(id_acc, ood_acc) = sqrt(id_acc * ood_acc)
The geometric mean forces BOTH regimes to be good: a memorizer with high id_acc but
chance ood_acc gets a low gmean.  We normalize against a weak baseline and the ideal:
    base    = gmean of the MAJORITY-label classifier (predict the train majority label
              on every test trellis) -- the evaluator computes this itself
    oracle  = 1.0 (the true hidden DFA labels every test trellis correctly)
    r = clamp( 0.1 + 0.9 * (obj - base) / max(1e-9, oracle - base), 0, 1 )
Matching the majority baseline scores ~0.1; perfectly recovering the hidden rule on all
lengths scores 1.0.  Because exact DFA identification from finite data is
under-determined (and hard on the sparser / larger instances), even good grammar-
induction learners stay below 1.0 on much of the family -> genuine headroom.

  The reported Ratio is the mean of r over all instances; Vector lists per-instance r.

ISOLATION.  The candidate runs in a FRESH SUBPROCESS via `isorun.run_candidate`; it
only ever sees the PUBLIC instance (train examples).  The hidden test trellises, their
labels, the true DFA, and the majority baseline are computed only in THIS parent
process, so a frame-walking / introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math, random
import isorun


# ----------------------------- hidden DFA family ---------------------------
def _random_dfa(rng, m, D):
    """A random complete DFA over alphabet D with m states, start=0."""
    trans = [[rng.randrange(m) for _ in range(D)] for _ in range(m)]
    accept = [rng.randrange(2) for _ in range(m)]
    return {"start": 0, "trans": trans, "accept": accept}


def _run(dfa, seq):
    s = dfa["start"]
    tr = dfa["trans"]
    for c in seq:
        s = tr[s][c]
    return dfa["accept"][s]


def _sample_seq(rng, D, lo, hi):
    L = rng.randint(lo, hi)
    return [rng.randrange(D) for _ in range(L)]


def _accept_rate(dfa, seqs):
    if not seqs:
        return 0.0
    return sum(_run(dfa, s) for s in seqs) / len(seqs)


def _build_instance(seed, D, m, max_states, n_train,
                    id_lo, id_hi, ood_lo, ood_hi, n_test):
    """Deterministically build one instance. Rejection-samples the hidden DFA so the
    label distribution is reasonably balanced on BOTH test regimes (keeps the majority
    baseline well below the oracle -> real headroom / discrimination)."""
    for attempt in range(200):
        rng = random.Random((seed * 1_000_003) ^ (attempt * 97 + 13))
        dfa = _random_dfa(rng, m, D)

        # sample disjoint train / id_test / ood_test string sets
        seen = set()
        train = []
        while len(train) < n_train:
            sq = _sample_seq(rng, D, id_lo, id_hi)
            key = tuple(sq)
            if key in seen:
                continue
            seen.add(key)
            train.append(sq)

        id_test = []
        while len(id_test) < n_test:
            sq = _sample_seq(rng, D, id_lo, id_hi)
            key = tuple(sq)
            if key in seen:
                continue
            seen.add(key)
            id_test.append(sq)

        ood_seen = set()
        ood_test = []
        while len(ood_test) < n_test:
            sq = _sample_seq(rng, D, ood_lo, ood_hi)
            key = tuple(sq)
            if key in ood_seen:
                continue
            ood_seen.add(key)
            ood_test.append(sq)

        tr_rate = _accept_rate(dfa, train)
        id_rate = _accept_rate(dfa, id_test)
        ood_rate = _accept_rate(dfa, ood_test)
        if all(0.30 <= r <= 0.70 for r in (tr_rate, id_rate, ood_rate)):
            train_lab = [[sq, int(_run(dfa, sq))] for sq in train]
            id_lab = [(sq, int(_run(dfa, sq))) for sq in id_test]
            ood_lab = [(sq, int(_run(dfa, sq))) for sq in ood_test]
            return {
                "name": f"trellis{seed}",
                "n_types": D,
                "max_states": max_states,
                "train": train_lab,
                "id_test": id_lab,
                "ood_test": ood_lab,
            }
    raise RuntimeError(f"could not balance instance seed={seed}")


def _build_instances():
    """Deterministic instance family.
    (seed, D, m, max_states, n_train, id_lo, id_hi, ood_lo, ood_hi, n_test)."""
    specs = [
        # smaller rule / ample-ish data -> mostly learnable (upper-mid scores)
        (101, 2, 6, 64, 220, 3, 12, 16, 30, 200),
        (102, 2, 7, 64, 200, 3, 12, 16, 30, 200),
        (103, 2, 8, 64, 200, 3, 13, 17, 32, 200),
        (104, 3, 6, 64, 220, 3, 11, 15, 28, 200),
        # moderate rule / data
        (205, 2, 8, 64, 160, 3, 12, 16, 32, 200),
        (206, 2, 9, 64, 180, 4, 13, 18, 34, 200),
        (207, 3, 7, 64, 180, 3, 12, 16, 30, 200),
        (208, 2, 10, 64, 170, 4, 14, 18, 36, 200),
        # bigger rule / sparser data -> under-determined (real headroom)
        (309, 2, 10, 64, 130, 4, 13, 18, 36, 200),
        (310, 2, 12, 64, 150, 4, 14, 20, 40, 200),
        (311, 3, 9, 64, 150, 3, 12, 16, 34, 200),
        (312, 2, 12, 64, 130, 4, 15, 22, 42, 200),
    ]
    out = []
    for sp in specs:
        out.append(_build_instance(*sp))
    return out


# ----------------------------- submission validation -----------------------
def _parse_dfa(answer, D, M):
    """Validate a submitted DFA. Return a normalized dfa dict or None."""
    if not isinstance(answer, dict):
        return None
    start = answer.get("start")
    accept = answer.get("accept")
    trans = answer.get("trans")
    if isinstance(start, bool) or not isinstance(start, int):
        return None
    if not isinstance(accept, list) or not isinstance(trans, list):
        return None
    K = len(trans)
    if K < 1 or K > M:
        return None
    if len(accept) != K:
        return None
    if not (0 <= start < K):
        return None
    acc = []
    for a in accept:
        if isinstance(a, bool):
            acc.append(1 if a else 0)
        elif isinstance(a, int) and a in (0, 1):
            acc.append(a)
        else:
            return None
    tr = []
    for row in trans:
        if not isinstance(row, list) or len(row) != D:
            return None
        r2 = []
        for t in row:
            if isinstance(t, bool) or not isinstance(t, int):
                return None
            if not (0 <= t < K):
                return None
            r2.append(t)
        tr.append(r2)
    return {"start": start, "accept": acc, "trans": tr}


def _dfa_acc(dfa, labeled):
    good = 0
    for seq, y in labeled:
        if _run(dfa, seq) == y:
            good += 1
    return good / len(labeled) if labeled else 0.0


# ----------------------------- scoring driver ------------------------------
def _baseline_gmean(inst):
    ones = sum(1 for _, y in inst["train"] if y == 1)
    maj = 1 if ones * 2 >= len(inst["train"]) else 0
    id_acc = sum(1 for _, y in inst["id_test"] if y == maj) / len(inst["id_test"])
    ood_acc = sum(1 for _, y in inst["ood_test"] if y == maj) / len(inst["ood_test"])
    return math.sqrt(id_acc * ood_acc)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        D = inst["n_types"]
        M = inst["max_states"]
        base = _baseline_gmean(inst)
        denom = 1.0 - base
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "n_types": D, "max_states": M,
                  "train": [[list(sq), int(y)] for sq, y in inst["train"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            dfa = _parse_dfa(ans, D, M)
        except Exception:
            dfa = None
        if dfa is None:
            vec.append(0.0)
            continue
        try:
            id_acc = _dfa_acc(dfa, inst["id_test"])
            ood_acc = _dfa_acc(dfa, inst["ood_test"])
            obj = math.sqrt(id_acc * ood_acc)
        except Exception:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (obj - base) / denom
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
