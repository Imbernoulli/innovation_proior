#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0400 -- "Wildlife Corridor Flow-Balance: Length Generalization"
(family: synthetic-algorithmic-generalization; format B, correctness/exact-match).

THEME.  Field ecologists instrument a wildlife corridor with a chain of gates.  Each
crossing event is tagged with a GATE TAG (an integer symbol 0..A-1) recording which
kind of gate an animal passed.  Every tag carries a hidden NET-FLOW weight w[tag] in
{-3..+3}: a positive weight means that gate tends to route animals INTO the protected
core, a negative weight OUT of it.  Reading a whole corridor log (a sequence of tags),
the running total = sum of the weights of its tags is the segment's NET FLOW.  Given a
tolerance band T the segment is classified:
    label 0  ("sink")     net flow  <  -T
    label 1  ("balanced") -T <= net flow <= T
    label 2  ("source")   net flow  >   T
The weights w[.] are HIDDEN.  A survey delivers a batch of SHORT, already-classified
corridor logs (the TRAIN split, "in-distribution" lengths) plus a batch of UNLABELED
logs to classify (the TEST split).  The test split mixes IN-DISTRIBUTION lengths with
much LONGER out-of-distribution logs.  The tolerance T is disclosed; the per-tag
weights are not.

This is a purest length-generalization task (MLS-Bench synth-* shape): a model that
merely MEMORIZES the short training logs matches their surface statistics but cannot
extrapolate to long logs, whereas a model that RECOVERS the underlying per-tag flow
rule classifies any length exactly.  The in-distribution-vs-length-OOD gap is what the
score rewards.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "A": int, "T": int,
             "train": [ {"seq": [int,...], "label": int}, ... ],   # labelled, SHORT
             "test":  [ [int,...], ... ] }                         # UNLABELLED logs to classify
          Every symbol is an integer in [0, A-1]; every train label is in {0,1,2}.
  stdout: ONE JSON object:
            {"labels": [l_0, ..., l_{M-1}]}
          exactly M = len(test) integers, each in {0,1,2}, l_i = the predicted class of
          test[i].  Wrong length, an out-of-range/boolean/non-integer label, a crash, a
          timeout, or non-JSON  ->  that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance the parent computes the hidden
true labels and two references:
    acc_cand = fraction of test logs the candidate classifies correctly
    acc_base = accuracy of the MAJORITY-OF-TRAIN constant predictor (predict the most
               frequent train label, ties -> smallest label, for every test log)
  and normalizes with an affine anchor (majority baseline -> 0.1, perfect -> 1.0):
    r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / max(1e-9, 1 - acc_base), 0, 1 )
  A candidate that just echoes the majority train label scores ~0.1; a candidate that
  classifies every log exactly scores 1.0; doing worse than the majority baseline scores
  < 0.1.  Because the SHORT train logs only pin the weights down to a band (the tolerance
  T leaves integer slack, and short logs rarely exercise every tag heavily), even a
  principled weight-recovery solver disagrees with the truth on some long OOD logs ->
  headroom below 1.0.

ISOLATION.  The candidate is untrusted and runs in a FRESH SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance (train + UNLABELLED test).
The hidden weights and the true test labels are computed by THIS parent process, so a
frame-walking / filesystem-scraping candidate learns nothing useful.

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

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- rule / label --------------------------------
def _label(total, T):
    if total < -T:
        return 0
    if total > T:
        return 2
    return 1


def _seq_total(seq, w):
    t = 0
    for s in seq:
        t += w[s]
    return t


# ----------------------------- instance family -----------------------------
def _build_weights(ni, A):
    """Hidden per-tag net-flow weights in {-3..3}, guaranteed non-degenerate
    (at least two nonzero and not all the same sign)."""
    while True:
        w = [ni(-3, 3) for _ in range(A)]
        nz = [x for x in w if x != 0]
        if len(nz) >= 2 and any(x > 0 for x in nz) and any(x < 0 for x in nz):
            return w


def _gen_seq(ni, A, L):
    return [ni(0, A - 1) for _ in range(L)]


def _build_one(seed, A, T, n_train, n_id, n_ood, id_lo, id_hi, ood_lo, ood_hi):
    ni = _rng(seed)
    w = _build_weights(ni, A)

    # Train + ID-test logs share the SHORT length band; OOD logs are much longer.
    train = []
    for _ in range(n_train):
        L = ni(id_lo, id_hi)
        seq = _gen_seq(ni, A, L)
        train.append({"seq": seq, "label": _label(_seq_total(seq, w), T)})

    test_seqs = []
    test_labels = []
    for _ in range(n_id):
        L = ni(id_lo, id_hi)
        seq = _gen_seq(ni, A, L)
        test_seqs.append(seq)
        test_labels.append(_label(_seq_total(seq, w), T))
    for _ in range(n_ood):
        L = ni(ood_lo, ood_hi)
        seq = _gen_seq(ni, A, L)
        test_seqs.append(seq)
        test_labels.append(_label(_seq_total(seq, w), T))

    return {
        "name": f"corridor{seed}",
        "A": A, "T": T,
        "train": train,
        "test": test_seqs,
        "true": test_labels,
    }


def _build_instances():
    # (seed, A, T, n_train, n_id, n_ood, id_lo, id_hi, ood_lo, ood_hi)
    specs = [
        (10101, 4, 2, 36, 18, 22, 6, 14, 30, 48),
        (10202, 4, 2, 40, 18, 22, 6, 14, 32, 52),
        (10303, 5, 2, 44, 18, 22, 6, 14, 30, 50),
        (10404, 3, 1, 30, 18, 22, 5, 12, 28, 46),
        (10505, 5, 3, 44, 18, 22, 7, 15, 34, 56),
        (10606, 4, 2, 38, 18, 22, 6, 13, 30, 48),
        (10707, 5, 2, 46, 18, 22, 6, 14, 32, 54),
        (10808, 4, 3, 40, 18, 22, 7, 14, 30, 50),
        # harder / larger held-out instances (more tags, wider OOD gap, bigger band)
        (10909, 5, 3, 42, 18, 22, 7, 15, 40, 64),
        (11010, 5, 2, 40, 18, 22, 6, 13, 42, 66),
        (11111, 4, 3, 36, 18, 22, 6, 14, 44, 68),
        (11212, 5, 3, 44, 18, 22, 7, 15, 46, 70),
    ]
    return [_build_one(*s) for s in specs]


# ----------------------------- references ----------------------------------
def _majority_label(train):
    cnt = {0: 0, 1: 0, 2: 0}
    for ex in train:
        cnt[ex["label"]] += 1
    # ties -> smallest label
    best = 0
    for lab in (0, 1, 2):
        if cnt[lab] > cnt[best]:
            best = lab
    return best


def _accuracy(pred, true):
    return sum(1 for p, t in zip(pred, true) if p == t) / len(true)


# ----------------------------- validation ----------------------------------
def _validate(answer, m):
    """Return a clean list of M labels in {0,1,2}, or None."""
    if not isinstance(answer, dict):
        return None
    labels = answer.get("labels")
    if not isinstance(labels, list) or len(labels) != m:
        return None
    out = []
    for x in labels:
        if isinstance(x, bool) or not isinstance(x, int):
            return None
        if x not in (0, 1, 2):
            return None
        out.append(x)
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        true = inst["true"]
        m = len(true)
        acc_base = _accuracy([_majority_label(inst["train"])] * m, true)
        public = {
            "name": inst["name"], "A": inst["A"], "T": inst["T"],
            "train": [{"seq": list(ex["seq"]), "label": ex["label"]} for ex in inst["train"]],
            "test": [list(s) for s in inst["test"]],
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            pred = _validate(ans, m)
        except Exception:
            pred = None
        if pred is None:
            vec.append(0.0)
            continue
        acc_cand = _accuracy(pred, true)
        denom = 1.0 - acc_base
        if denom < 1e-9:
            denom = 1e-9
        r = 0.1 + 0.9 * (acc_cand - acc_base) / denom
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
