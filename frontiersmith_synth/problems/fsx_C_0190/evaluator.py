#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0190 -- "Deep-Sea Cable Splice: Length-OOD Generalization"
(family: synthetic-algorithmic-generalization; format B, correctness).

THEME.  A deep-sea cable network is assembled from CONNECTOR TOKENS.  There are K
connector TYPES; each type has an OPENING token (a lowercase letter) and a matching
CLOSING token (the same letter uppercased).  A "cable" is a string of such tokens.

A cable is WELL-SPLICED iff it obeys a hidden deterministic rule:
    (1) the connectors are properly matched and nested -- reading left to right on a
        stack, every closing token must close the most-recently-opened, still-open
        connector OF THE SAME TYPE, and the stack must be empty at the end
        (i.e. the string is a valid Dyck word over K bracket types); AND
    (2) the nesting never runs deeper than a hidden splice-depth budget D
        (the maximum stack depth is <= D).
The rule (both K's alphabet and, crucially, D) is NOT disclosed to the solver.

THE TASK (a length-OOD generalization problem).  For each instance the solver is
handed a LABELLED training set of SHORT cables (in-distribution, ID) and must
predict the well-spliced label (1) / not (0) for a QUERY set that mixes ID-length
cables with much LONGER out-of-distribution (OOD) cables.  A learner that only
memorises training strings, or that leans on length-correlated features, collapses
on the OOD cables; only a learner that recovers the underlying matching algorithm
(and makes a sensible inductive-bias choice about the unseen depth budget)
generalises.  Because the training cables never probe the depth budget, D is
genuinely under-determined -- so even a strong learner leaves headroom.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "n_types": K,                         # number of connector types
             "open_symbols":  [str, ...],          # K opening tokens (lowercase)
             "close_symbols": [str, ...],          # K matching closing tokens
             "train":  [[cable_str, label], ...],  # label in {0,1}; the learning signal
             "queries":[cable_str, ...]}           # UNLABELLED; predict these
  stdout: ONE JSON object:
            {"labels": [0/1, 0/1, ...]}            # EXACTLY len(queries) ints in {0,1}

  A prediction is VALID iff `labels` is a list of exactly len(queries) items, each an
  int equal to 0 or 1.  Wrong length, non-integer / out-of-set entries, a crash, a
  timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance:
    acc_cand = fraction of queries whose predicted label matches the hidden truth.
    acc_base = accuracy of the best CONSTANT classifier on the hidden query labels
               (= max(valid_fraction, 1 - valid_fraction)); every instance is built
               so "not well-spliced" is the strict majority, so acc_base > 0.5.
  Normalised with an affine anchor (majority-constant -> 0.1, perfect -> 1.0):
    r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / (1.0 - acc_base), 0, 1 )
  Predicting the majority class scores ~0.1; doing worse scores < 0.1; correctly
  generalising the matching rule to the OOD cables scores higher.  A perfect score
  requires guessing the hidden depth budget D, which the ID training set does not
  reveal -> headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance (train labels + query
STRINGS).  The hidden query labels, the true rule, and the depth budget D live only
in THIS parent process, so a frame-walking / introspecting candidate learns nothing.

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


ALPHA = "abcdefgh"
D_TRUE = 6            # hidden splice-depth budget (same for every instance)
TRAIN_MAX_DEPTH = 4   # training valid cables never nest deeper than this


# ----------------------------- string construction -------------------------
def _build_valid(ni, L, d, K, opens, closes):
    """A well-matched cable of length L with maximum nesting depth EXACTLY d.

    core = a d-deep nested block (2d tokens); the rest = shallow top-level pairs."""
    seq = []
    stack = []
    for _ in range(d):
        t = ni(0, K - 1)
        stack.append(t)
        seq.append(opens[t])
    while stack:
        t = stack.pop()
        seq.append(closes[t])
    rem = L - 2 * d
    for _ in range(rem // 2):
        t = ni(0, K - 1)
        seq.append(opens[t])
        seq.append(closes[t])
    return "".join(seq)


def _make_unbalanced(ni, s):
    """Delete one token -> odd length -> counts can never balance -> not well-spliced."""
    i = ni(0, len(s) - 1)
    return s[:i] + s[i + 1:]


def _make_typemismatch(ni, s, K, opens, closes):
    """Retype one closing token (requires K>=2). Counts stay balanced, but the token
    no longer matches its opener -> not well-spliced, yet count-balance can't see it."""
    close_set = set(closes)
    positions = [i for i, ch in enumerate(s) if ch in close_set]
    i = positions[ni(0, len(positions) - 1)]
    cur = closes.index(s[i])
    newt = (cur + 1 + ni(0, K - 2)) % K        # any type != cur
    return s[:i] + closes[newt] + s[i + 1:]


def _oracle(s, opens, closes, dbudget):
    """Ground-truth well-spliced test with a given depth budget."""
    open_of = {c: i for i, c in enumerate(opens)}
    close_of = {c: i for i, c in enumerate(closes)}
    stack = []
    maxd = 0
    for ch in s:
        if ch in open_of:
            stack.append(open_of[ch])
            if len(stack) > maxd:
                maxd = len(stack)
        elif ch in close_of:
            if not stack or stack[-1] != close_of[ch]:
                return 0
            stack.pop()
        else:
            return 0
    if stack:
        return 0
    return 1 if maxd <= dbudget else 0


# ----------------------------- instance family -----------------------------
def _build_instance(seed, K):
    ni = _rng(seed)
    opens = list(ALPHA[:K])
    closes = [c.upper() for c in opens]

    def id_len():
        return 8 + 2 * ni(0, 6)          # even, 8..20
    def ood_len():
        return 40 + 2 * ni(0, 20)        # even, 40..80

    # ---------------- training set (labelled, ID, depth <= TRAIN_MAX_DEPTH) --------
    train = []
    # guarantee the deepest training valid reaches TRAIN_MAX_DEPTH so D_hat is inferable
    train_valids = [_build_valid(ni, max(id_len(), 2 * TRAIN_MAX_DEPTH),
                                 TRAIN_MAX_DEPTH, K, opens, closes)]
    for _ in range(19):
        d = ni(1, TRAIN_MAX_DEPTH)
        train_valids.append(_build_valid(ni, max(id_len(), 2 * d), d, K, opens, closes))
    for s in train_valids:
        train.append([s, 1])
    for _ in range(10):
        d = ni(1, TRAIN_MAX_DEPTH)
        base = _build_valid(ni, max(id_len(), 2 * d), d, K, opens, closes)
        train.append([_make_unbalanced(ni, base), 0])
    for _ in range(10):
        d = ni(1, TRAIN_MAX_DEPTH)
        base = _build_valid(ni, max(id_len(), 2 * d), d, K, opens, closes)
        train.append([_make_typemismatch(ni, base, K, opens, closes), 0])

    # ---------------- query set (unlabelled to the solver) ------------------------
    queries = []

    def add_valid_id(n):
        for _ in range(n):
            d = ni(1, TRAIN_MAX_DEPTH)
            queries.append(_build_valid(ni, max(id_len(), 2 * d), d, K, opens, closes))

    def add_valid_ood_shallow(n):
        for _ in range(n):
            d = ni(1, TRAIN_MAX_DEPTH)
            queries.append(_build_valid(ni, max(ood_len(), 2 * d), d, K, opens, closes))

    def add_valid_ood_deep(n):
        for _ in range(n):
            d = ni(TRAIN_MAX_DEPTH + 1, D_TRUE)      # depth 5..6, still <= D_TRUE => valid
            queries.append(_build_valid(ni, max(ood_len(), 2 * d), d, K, opens, closes))

    def add_invalid(n, ood, kind):
        for _ in range(n):
            d = ni(1, TRAIN_MAX_DEPTH)
            L = max(ood_len() if ood else id_len(), 2 * d)
            base = _build_valid(ni, L, d, K, opens, closes)
            if kind == "unbal":
                queries.append(_make_unbalanced(ni, base))
            else:
                queries.append(_make_typemismatch(ni, base, K, opens, closes))

    def add_invalid_depth(n):
        for _ in range(n):
            d = ni(D_TRUE + 1, D_TRUE + 3)           # depth 7..9 > D_TRUE => not well-spliced
            queries.append(_build_valid(ni, max(ood_len(), 2 * d), d, K, opens, closes))

    add_valid_id(8)                 # ID  well-spliced, shallow
    add_valid_ood_shallow(7)        # OOD well-spliced, shallow  (length extrapolation)
    add_valid_ood_deep(6)           # OOD well-spliced, deep     (depth-budget headroom)
    add_invalid(4, False, "unbal")  # ID  not-spliced, unbalanced
    add_invalid(4, False, "type")   # ID  not-spliced, type-mismatch
    add_invalid(6, True, "unbal")   # OOD not-spliced, unbalanced
    add_invalid(6, True, "type")    # OOD not-spliced, type-mismatch
    add_invalid_depth(9)            # OOD not-spliced, too deep

    labels = [_oracle(s, opens, closes, D_TRUE) for s in queries]

    # deterministic shuffle of both train and queries so labels aren't positional
    def shuffle(items):
        a = list(items)
        for i in range(len(a) - 1, 0, -1):
            j = ni(0, i)
            a[i], a[j] = a[j], a[i]
        return a

    train = shuffle(train)
    pairs = shuffle(list(zip(queries, labels)))
    queries = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]

    public = {"name": f"cable{seed}", "n_types": K,
              "open_symbols": opens, "close_symbols": closes,
              "train": train, "queries": queries}
    hidden = {"labels": labels}
    return {"public": public, "hidden": hidden}


def _build_instances():
    specs = [(7001, 2), (7002, 3), (7003, 4), (7004, 2), (7005, 3),
             (7006, 4), (7007, 2), (7008, 3), (7009, 4), (7010, 3)]
    return [_build_instance(seed, K) for (seed, K) in specs]


# ----------------------------- scoring ------------------------------------
def _validate(public, answer):
    if not isinstance(answer, dict):
        return None
    labels = answer.get("labels")
    if not isinstance(labels, list):
        return None
    q = public["queries"]
    if len(labels) != len(q):
        return None
    out = []
    for v in labels:
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        if v not in (0, 1):
            return None
        out.append(v)
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = inst["public"]
        truth = inst["hidden"]["labels"]
        m = len(truth)
        pos = sum(truth)
        valid_frac = pos / m
        acc_base = max(valid_frac, 1.0 - valid_frac)
        denom = 1.0 - acc_base
        if denom < 1e-9:
            denom = 1e-9

        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            pred = _validate(public, ans)
        except Exception:
            pred = None
        if pred is None:
            vec.append(0.0)
            continue

        correct = sum(1 for a, b in zip(pred, truth) if a == b)
        acc = correct / m
        r = 0.1 + 0.9 * (acc - acc_base) / denom
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
