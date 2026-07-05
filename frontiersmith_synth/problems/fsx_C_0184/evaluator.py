#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0184 -- "Drone Delivery Swarm: Mission-Validity Rule Induction"
(family: synthetic-algorithmic-generalization; eval_form: correctness; MLS-Bench synth-* shape).

A drone-delivery swarm executes MISSIONS.  A mission is a sequence of atomic
commands over an alphabet of 2*K tokens:

    token 2*i      = LAUNCH a drone toward depot i   ("open" of type i)
    token 2*i + 1  = RETURN a drone from depot i      ("close" of type i)

A mission is VALID iff BOTH of the following hold (this is the HIDDEN rule the
candidate must INDUCE from labelled examples):

  (1) it is *well-nested* -- every RETURN matches the most recently LAUNCHed,
      still-airborne drone of the SAME depot (a Dyck-K word: LIFO, type-matched,
      and the swarm is empty at the end), and
  (2) its maximum simultaneous airborne count (the maximum nesting depth) never
      exceeds a per-swarm capacity  D  (the airspace can only hold D drones aloft
      at once).  D is NOT revealed and differs per instance.

The candidate is shown a TRAIN split (missions WITH their 0/1 labels) and must
predict the labels of a QUERY split (missions WITHOUT labels).  Crucially, some
instances are LENGTH-OOD: the queries are far LONGER than any training mission,
so a memoriser that keys on surface patterns collapses while a solver that
recovers the abstract rule (Dyck matching + a capacity threshold) generalises.
This ID-vs-length-OOD gap is the generalization signal.

The candidate runs as an ISOLATED subprocess (isorun): it reads ONE JSON "public
instance" from stdin and writes ONE JSON answer (a length-Q list of 0/1 labels)
to stdout.  It never sees the query labels, D, or this evaluator's memory.

Public instance JSON (stdin):
    {
      "num_types": int,                       # K
      "train":   [{"seq": [int,...], "label": 0|1}, ...],   # labelled missions
      "queries": [[int,...], ...],            # UNlabelled missions to classify
      "regime":  "ID" | "OOD_LENGTH",         # informational
      "seed":    int                          # per-instance seed the candidate MAY use
    }

Answer JSON (stdout):
    [0, 1, 0, 1, ...]                         # length == len(queries); label[i] for query i

Quality of one instance = classification accuracy against the HIDDEN query
labels, affinely anchored against the evaluator's own baseline (predict the
majority TRAIN label for every query):

    r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / max(1 - acc_base, MIN_DENOM), 0, 1 )

so reproducing the baseline maps to ~0.1 and a perfect classifier maps to ~1.0.
Valid instances are floored to a small positive value so the geometric mean stays
defined; an instance where the candidate raises, returns the wrong shape, or
emits non-0/1 labels scores exactly 0.0 (dragging the geometric mean to 0).  The
final Ratio is the GEOMETRIC MEAN over instances, so a method that nails the ID
instances but collapses on the length-OOD ones is heavily penalised.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <geometric mean of per-instance r, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys
import json
import math
import random

import isorun

VALID_FLOOR = 0.02     # floor for VALID instances so gmean stays defined
MIN_DENOM = 0.15       # cap on 1/(1-acc_base) so headroom stays sane
CAND_TIMEOUT = 20


# ============================ mission mechanics ============================
def dyck_ok_and_depth(seq, K):
    """Return (well_nested, max_depth). max_depth is meaningful only if well_nested."""
    stack = []
    mx = 0
    for t in seq:
        if not isinstance(t, int) or t < 0 or t >= 2 * K:
            return False, 0
        if t % 2 == 0:                     # LAUNCH of type t//2
            stack.append(t // 2)
            if len(stack) > mx:
                mx = len(stack)
        else:                              # RETURN of type (t-1)//2
            typ = (t - 1) // 2
            if not stack or stack[-1] != typ:
                return False, 0
            stack.pop()
    return (len(stack) == 0), mx


def gen_wellnested(rng, pairs, depth, K):
    """A well-nested Dyck-K word with exactly `pairs` pairs and max nesting == depth.
    Requires pairs >= depth."""
    seq = []
    stack = []
    # Phase 1: descend to the target depth so the maximum is exactly `depth`.
    for _ in range(depth):
        typ = rng.randrange(K)
        seq.append(2 * typ)
        stack.append(typ)
    opens_left = pairs - depth
    # Phase 2: random walk capped at `depth`, never exceeding it.
    while opens_left > 0 or stack:
        can_open = opens_left > 0 and len(stack) < depth
        can_close = len(stack) > 0
        if can_open and (not can_close or rng.random() < 0.5):
            typ = rng.randrange(K)
            seq.append(2 * typ)
            stack.append(typ)
            opens_left -= 1
        else:
            typ = stack.pop()
            seq.append(2 * typ + 1)
    return seq


def gen_structural_invalid(rng, K, pairs, depth):
    """A NOT-well-nested mission (broken matching), verified invalid."""
    base = gen_wellnested(rng, pairs, depth, K)
    for _ in range(40):
        s = list(base)
        mode = rng.randrange(3)
        if mode == 0 and len(s) >= 2:                    # swap two tokens
            i = rng.randrange(len(s))
            j = rng.randrange(len(s))
            s[i], s[j] = s[j], s[i]
        elif mode == 1:                                  # flip a launch<->return (imbalance)
            i = rng.randrange(len(s))
            s[i] = s[i] + 1 if s[i] % 2 == 0 else s[i] - 1
        else:                                            # retype a RETURN (mismatch)
            idxs = [i for i, t in enumerate(s) if t % 2 == 1]
            if idxs and K > 1:
                i = rng.choice(idxs)
                typ = (s[i] - 1) // 2
                s[i] = 2 * ((typ + 1) % K) + 1
        ok, _ = dyck_ok_and_depth(s, K)
        if not ok:
            return s
    return base + [1]                                    # guaranteed prefix underflow


def _pairs_short(rng, depth):
    return depth + rng.randint(0, 4)


def _pairs_long(rng, depth):
    return depth + rng.randint(9, 16)


def _make_split(rng, K, D, n_valid, n_struct, n_deep, pairs_fn, valid_depth_hi):
    """Build a labelled mission set.

    valid missions: well-nested, depth in [1, valid_depth_hi]  (label 1)
    struct-invalid: not well-nested                            (label 0)
    deep-invalid:   well-nested, depth in [D+1, D+2]           (label 0)
    """
    items = []
    for i in range(n_valid):
        d = 1 + (i % valid_depth_hi)                 # cycles 1..valid_depth_hi (guarantees coverage)
        p = max(pairs_fn(rng, d), d)
        items.append((gen_wellnested(rng, p, d, K), 1))
    for i in range(n_struct):
        d = 1 + (i % D)
        p = max(pairs_fn(rng, d), d)
        items.append((gen_structural_invalid(rng, K, p, d), 0))
    for i in range(n_deep):
        d = D + 1 + (i % 2)                          # D+1 or D+2
        p = max(pairs_fn(rng, d), d)
        items.append((gen_wellnested(rng, p, d, K), 0))
    rng.shuffle(items)
    return items


def _build_instances():
    """Deterministic instance family.  Train missions are SHORT everywhere; for
    OOD_LENGTH instances the QUERIES are LONG (length generalization)."""
    specs = [
        dict(seed=101, K=2, D=3, regime="ID"),
        dict(seed=102, K=2, D=4, regime="ID"),
        dict(seed=103, K=3, D=4, regime="ID"),
        dict(seed=104, K=2, D=3, regime="OOD_LENGTH"),
        dict(seed=105, K=2, D=4, regime="OOD_LENGTH"),
        dict(seed=106, K=3, D=5, regime="OOD_LENGTH"),
        dict(seed=107, K=3, D=5, regime="ID"),
        dict(seed=108, K=4, D=4, regime="OOD_LENGTH"),
        dict(seed=109, K=2, D=5, regime="ID"),
    ]
    out = []
    for sp in specs:
        rng = random.Random(sp["seed"])
        K, D = sp["K"], sp["D"]
        # TRAIN: valid depths only reach D-1 (the capacity boundary D is UNSEEN in
        # training) -> the exact threshold must be extrapolated, not memorised.
        train = _make_split(rng, K, D, n_valid=66, n_struct=37, n_deep=37,
                            pairs_fn=_pairs_short, valid_depth_hi=D - 1)
        q_pairs_fn = _pairs_long if sp["regime"] == "OOD_LENGTH" else _pairs_short
        # QUERIES: valid depths reach D (includes the unseen boundary).
        query_items = _make_split(rng, K, D, n_valid=48, n_struct=26, n_deep=26,
                                 pairs_fn=q_pairs_fn, valid_depth_hi=D)
        out.append({
            "name": "s%d_K%d_D%d_%s" % (sp["seed"], K, D, sp["regime"]),
            "K": K,
            "train": train,
            "queries": [seq for seq, _ in query_items],
            "labels": [lab for _, lab in query_items],
            "regime": sp["regime"],
            "seed": sp["seed"],
        })
    return out


# ============================ baseline + scoring ===========================
def _baseline_pred(train):
    """Majority TRAIN label (ties -> 0)."""
    ones = sum(1 for _, lab in train if lab == 1)
    zeros = len(train) - ones
    return 1 if ones > zeros else 0


def _accuracy(pred, truth):
    return sum(1 for a, b in zip(pred, truth) if a == b) / len(truth)


def _valid_labels(ans, q):
    """Return a length-q list of ints in {0,1}, or None if the answer is invalid."""
    if isinstance(ans, dict):
        ans = ans.get("labels", None)
    if not isinstance(ans, list) or len(ans) != q:
        return None
    out = []
    for x in ans:
        if isinstance(x, bool):
            out.append(1 if x else 0)
        elif isinstance(x, int) and x in (0, 1):
            out.append(x)
        elif isinstance(x, float) and x in (0.0, 1.0):
            out.append(int(x))
        else:
            return None
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        truth = inst["labels"]
        q = len(truth)
        base_lab = _baseline_pred(inst["train"])
        acc_base = _accuracy([base_lab] * q, truth)
        denom = max(1.0 - acc_base, MIN_DENOM)

        public = {
            "num_types": inst["K"],
            "train": [{"seq": seq, "label": lab} for seq, lab in inst["train"]],
            "queries": inst["queries"],
            "regime": inst["regime"],
            "seed": int(1000000 + inst["seed"]),
        }
        ans, st = isorun.run_candidate(cand, public, timeout=CAND_TIMEOUT)
        if st != "OK":
            vec.append(0.0)
            continue

        pred = _valid_labels(ans, q)
        if pred is None:
            vec.append(0.0)
            continue

        acc_cand = _accuracy(pred, truth)
        r = 0.1 + 0.9 * (acc_cand - acc_base) / denom
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        if r < VALID_FLOOR:
            r = VALID_FLOOR
        vec.append(float(r))

    if any(v <= 0.0 for v in vec):
        ratio = 0.0
    else:
        ratio = math.exp(sum(math.log(v) for v in vec) / len(vec))

    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(v, 6) for v in vec]))


if __name__ == "__main__":
    main()
