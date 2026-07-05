#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0463 -- "Credit-Scoring Decision Tree: Design the Split"
(family: ml-decision-tree-split; format B, quality-metric).

THEME.  A consumer-credit lender must decide, from an applicant's financial
profile, whether the applicant will DEFAULT (label 1) or REPAY (label 0).  The
lender's model is a binary decision tree: each internal node tests ONE feature
against a threshold (go LEFT if feature <= threshold, else RIGHT) and each leaf
predicts a class.  The whole modelling choice is the SPLIT CRITERION and the
stopping rule that a greedy top-down tree builder uses -- that is what the
candidate designs.

The candidate receives a labelled TRAINING sample of applicants and must return
a fully-built decision tree (the result of running whatever greedy split
criterion it designed).  The evaluator then scores that tree by its accuracy on
a HELD-OUT test set of applicants drawn from the SAME lending population that the
candidate never sees.  The tension is classic bias/variance: a tree that is too
shallow underfits the several distinct risk regions of the population, while a
tree grown too deep memorises the training noise and generalises worse -- so
neither "always predict the common class" nor "grow until pure" is optimal.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "n_features": F (int),               # number of numeric features
             "feature_names": [str, ...],         # length F, human labels
             "X_train": [[float, ...], ...],       # M training rows, each length F
             "y_train": [0/1, ...]}                # M training labels
  stdout: ONE JSON object describing a decision tree:
            {"nodes": [node0, node1, ...]}
          node 0 is the ROOT.  Each node is either
            LEAF     : {"leaf": 0}  or  {"leaf": 1}
            INTERNAL : {"feature": j, "threshold": t, "left": a, "right": b}
          To classify a row x, start at node 0; at an INTERNAL node go to child
          `left` if x[feature] <= threshold else child `right`; stop at a LEAF and
          predict its class.

  A tree is VALID iff: `nodes` is a non-empty list of at most MAX_NODES dicts;
  every LEAF's class is 0 or 1; every INTERNAL node has an integer feature in
  [0,F), a FINITE numeric threshold, and integer child indices in [0, len(nodes));
  and the graph reachable from the root is ACYCLIC (so classification always
  terminates at a leaf).  Wrong shape, out-of-range index, non-finite threshold, a
  cycle, a crash, a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance:
    acc_cand = accuracy of the candidate's tree on the held-out test set.
    acc_base = accuracy of the majority-class rule (predict the more common TEST
               class); this is the trivial "no split" baseline.
    acc_ub   = 1.0, a loose (Bayes-unreachable, because labels are noisy) ceiling
               that leaves headroom so even strong trees stay below 1.0.
  normalized with an affine anchor (majority rule -> 0.1, perfect -> 1.0):
    r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / max(1e-9, acc_ub - acc_base), 0, 1 )
  Reproducing the majority rule scores ~0.1; a tree that carves the population's
  risk regions well scores higher; a tree worse than the majority rule scores < 0.1.

ISOLATION.  The candidate is untrusted and runs in a FRESH SANDBOXED SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC instance (training data).
The held-out test set, the majority baseline, and all validation live in THIS
parent process, so a frame-walking / introspecting candidate learns nothing that
helps it and cannot read the held-out labels.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun

MAX_NODES = 4096
N_FEATURES = 8
FEATURE_NAMES = [
    "age", "annual_income", "debt_to_income", "credit_utilization",
    "num_prior_defaults", "employment_years", "num_open_accounts",
    "recent_inquiries",
]


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def u01():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / (1 << 53)

    return u01


# ----------------------------- credit population ---------------------------
def _gen(seed, n, pnoise):
    """Deterministic labelled sample from a region-structured lending population.

    Default risk is driven by several DISJOINT risk regions (AND-clauses of two
    or three features), so the Bayes-optimal boundary is genuinely non-linear and
    a greedy tree needs several splits to carve it.  Labels are Bernoulli with
    probability (1-pnoise) inside a risk region and pnoise outside -> irreducible
    noise `pnoise` that both punishes over-deep memorisation and keeps accuracy
    below 1.0.
    """
    u = _rng(seed)
    X = []
    y = []
    for _ in range(n):
        age = 18.0 + u() * 57.0
        income = 10000.0 + (u() ** 2) * 190000.0
        dti = u()
        util = u()
        defaults = float(int(u() ** 2 * 6.999))
        emp = u() * 40.0
        accounts = float(int(u() * 20))
        inq = float(int(u() ** 1.5 * 15))
        f1 = (dti > 0.55 and util > 0.55)
        f2 = (defaults >= 3)
        f3 = (income < 45000.0 and inq > 6)
        f4 = (util > 0.85)
        f5 = (age < 28.0 and emp < 2.0 and dti > 0.5)
        risky = (f1 or f2 or f3 or f4 or f5)
        p = (1.0 - pnoise) if risky else pnoise
        lab = 1 if u() < p else 0
        X.append([age, income, dti, util, defaults, emp, accounts, inq])
        y.append(lab)
    return X, y


def _build_instances():
    """Deterministic instance family: (seed, n, pnoise)."""
    specs = [
        (101, 400, 0.10),
        (102, 420, 0.12),
        (103, 380, 0.08),
        (104, 440, 0.14),
        (105, 400, 0.10),
        # harder / larger held-out instances (more noise, larger populations)
        (206, 460, 0.12),
        (207, 420, 0.09),
        (208, 480, 0.15),
        (209, 440, 0.07),
        (210, 500, 0.12),
    ]
    out = []
    for (seed, n, pnoise) in specs:
        Xtr, ytr = _gen(seed, n, pnoise)
        Xte, yte = _gen(seed + 9000, n, pnoise)
        out.append({"name": f"portfolio{seed}", "seed": seed,
                    "X_train": Xtr, "y_train": ytr,
                    "X_test": Xte, "y_test": yte})
    return out


# ----------------------------- validation ----------------------------------
def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool))


def _validate(answer):
    """Return the validated `nodes` list (as plain dicts) or None."""
    if not isinstance(answer, dict):
        return None
    nodes = answer.get("nodes")
    if not isinstance(nodes, list):
        return None
    m = len(nodes)
    if m < 1 or m > MAX_NODES:
        return None
    norm = []
    for nd in nodes:
        if not isinstance(nd, dict):
            return None
        if "leaf" in nd:
            lv = nd["leaf"]
            if not _is_int(lv) or lv not in (0, 1):
                return None
            norm.append(("L", lv))
        else:
            f = nd.get("feature"); t = nd.get("threshold")
            a = nd.get("left"); b = nd.get("right")
            if not _is_int(f) or f < 0 or f >= N_FEATURES:
                return None
            if not _is_num(t):
                return None
            tf = float(t)
            if tf != tf or tf in (float("inf"), float("-inf")):
                return None
            if not _is_int(a) or a < 0 or a >= m:
                return None
            if not _is_int(b) or b < 0 or b >= m:
                return None
            norm.append(("I", f, tf, a, b))
    # acyclicity check on the graph reachable from the root (node 0)
    state = [0] * m          # 0=unvisited, 1=in-stack, 2=done
    stack = [(0, False)]
    while stack:
        i, closing = stack.pop()
        if closing:
            state[i] = 2
            continue
        if state[i] == 2:
            continue
        if state[i] == 1:
            return None       # back-edge -> cycle
        state[i] = 1
        stack.append((i, True))
        nd = norm[i]
        if nd[0] == "I":
            stack.append((nd[3], False))
            stack.append((nd[4], False))
    return norm


def _predict(norm, x):
    i = 0
    for _ in range(len(norm) + 1):
        nd = norm[i]
        if nd[0] == "L":
            return nd[1]
        i = nd[3] if x[nd[1]] <= nd[2] else nd[4]
    return 0      # unreachable given acyclicity, kept as safety


def _accuracy(norm, X, y):
    good = 0
    for xi, yi in zip(X, y):
        if _predict(norm, xi) == yi:
            good += 1
    return good / len(y)


def _majority_acc(y):
    c1 = sum(y); c0 = len(y) - c1
    maj = 0 if c0 >= c1 else 1
    return (c0 if maj == 0 else c1) / len(y)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        Xte = inst["X_test"]; yte = inst["y_test"]
        acc_base = _majority_acc(yte)
        denom = 1.0 - acc_base
        if denom < 1e-9:
            denom = 1e-9
        public = {
            "name": inst["name"],
            "n_features": N_FEATURES,
            "feature_names": list(FEATURE_NAMES),
            "X_train": [list(row) for row in inst["X_train"]],
            "y_train": list(inst["y_train"]),
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            norm = _validate(ans)
        except Exception:
            norm = None
        if norm is None:
            vec.append(0.0)
            continue
        try:
            acc_cand = _accuracy(norm, Xte, yte)
        except Exception:
            vec.append(0.0)
            continue
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
