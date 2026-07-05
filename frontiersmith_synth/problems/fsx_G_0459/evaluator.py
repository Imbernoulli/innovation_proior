#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0459 -- "Cold-Start Label Budget: Pool-Based
Active-Learning Query Order" (family: ml-active-learning-query; format B,
quality-metric; objective = MAXIMIZE).

THEME (labeling budget).  A data team has a big pool of UNLABELED examples and a
tiny labeling budget.  Every label costs money, so they must decide -- up front,
from the feature geometry ALONE (they have not seen any labels yet) -- the ORDER
in which to send pool examples to human annotators.  A fixed, deterministic
classifier (a nearest-centroid / Rocchio model) is retrained after each new label
and evaluated on a held-out test set.  Plotting test accuracy against the number
of labels acquired gives a LEARNING CURVE; the team is graded on the AREA UNDER
that learning curve (AULC).  A good query order reaches high accuracy with FEW
labels -- the classic pool-based active-learning objective, in the hard cold-start
regime where the strategy sees no labels and must rely on representativeness /
coverage / diversity of the pool.

Why it is genuinely open-ended.  The pool contains several well-separated clusters
(the latent classes) with IMBALANCED priors: a couple of common classes and
several rare ones.  The nearest-centroid model can only predict a class once at
least one of its members has been labeled, so early accuracy is dominated by how
fast the query order COVERS the distinct clusters -- especially the rare ones.
Uniform-random ordering wastes budget re-sampling the common clusters; a
diversity/coverage-aware order (farthest-first / k-center, dispersion, density
peaks, clustering representatives, ...) hits the rare clusters far sooner.  There
is no closed-form optimum: the immediate-jump ceiling is unreachable, and many
distinct strategies trade off coverage vs. representativeness differently.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "d": int, "n": int, "budget": int,
             "pool": [[x_0_0, ..., x_0_{d-1}], ...]}   # n unlabeled feature rows
          NO labels, NO class count, NO test set are ever revealed.
  stdout: ONE JSON object:
            {"order": [p_0, p_1, ..., p_{n-1}]}
          a PERMUTATION of 0..n-1 -- the order in which pool rows would be sent
          for labeling.  The evaluator acquires labels along this order and only
          the first `budget` entries affect the score, but a VALID answer must be
          a full permutation (exactly n distinct integers in [0, n)).

  Invalid output (not a permutation, wrong length, out of range, repeats, a crash,
  a timeout, or non-JSON) -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  The fixed model is nearest-centroid:
train = mean feature vector per labeled class; predict = nearest class centroid
(only classes seen so far are predictable).  For a query order o and budget B,
    AULC(o) = (1/B) * sum_{k=1..B} test_accuracy( model trained on o[:k] ).
Per instance we compute three references IN THIS PARENT PROCESS (never revealed):
    base   = mean AULC over several fixed seeded RANDOM permutations   (weak anchor)
    ceil   = test accuracy of the model trained on the WHOLE pool      (immediate-
             jump upper bound; unreachable because early budgets cannot cover all
             classes, so even the best order stays strictly below it)
    cand   = AULC of the candidate's order
and normalize with an affine anchor (random -> 0.1, immediate-jump -> 1.0):
    r = clamp( 0.1 + 0.9 * (cand - base) / max(1e-9, ceil - base), 0, 1 )
A random-quality order scores ~0.1; matching the (unreachable) instant-ceiling
scores 1.0; doing worse than random scores < 0.1.

ISOLATION.  The candidate is untrusted and runs in a FRESH OS-SANDBOXED subprocess
via `isorun.run_candidate`; it only ever sees the PUBLIC instance (pool features).
Labels, the test set, the references, and the model live only in THIS parent
process, so a frame-walking / filesystem-scraping candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    """64-bit LCG -> uniform floats in [0,1); fully deterministic."""
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def u():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    return u


def _gauss(u):
    """One standard-normal sample from a uniform generator via Box-Muller."""
    a = u()
    b = u()
    if a < 1e-12:
        a = 1e-12
    return math.sqrt(-2.0 * math.log(a)) * math.cos(2.0 * math.pi * b)


# ----------------------------- instance family -----------------------------
def _build_instance(seed, d, K, n, m, budget, sep, sigma):
    """Deterministic clustered classification instance with imbalanced priors.

    Returns pool features (public), pool labels, test features, test labels,
    and the budget.  Classes are well-separated Gaussian blobs; a few common
    classes and several rare ones (coverage of rare clusters is what matters)."""
    u = _rng(seed)

    # class priors: 2 common classes, the rest rare (coverage-sensitive)
    raw = []
    for c in range(K):
        raw.append(3.0 if c < 2 else 0.6)
    tot = sum(raw)
    priors = [w / tot for w in raw]
    cum = []
    acc = 0.0
    for p in priors:
        acc += p
        cum.append(acc)

    # cluster centers, well separated in R^d
    centers = []
    for c in range(K):
        centers.append([(u() * 2.0 - 1.0) * sep for _ in range(d)])

    def sample_class():
        r = u()
        for c in range(K):
            if r <= cum[c]:
                return c
        return K - 1

    def sample_point(c):
        ctr = centers[c]
        return [ctr[j] + sigma * _gauss(u) for j in range(d)]

    # pool: force at least one member of every class to be PRESENT (so a perfect
    # order COULD cover all classes), then fill the rest by prior, then shuffle.
    pool_y = list(range(K))
    while len(pool_y) < n:
        pool_y.append(sample_class())
    # deterministic shuffle (Fisher-Yates with the same RNG)
    for i in range(n - 1, 0, -1):
        j = int(u() * (i + 1))
        if j > i:
            j = i
        pool_y[i], pool_y[j] = pool_y[j], pool_y[i]
    pool_X = [sample_point(c) for c in pool_y]

    # test set drawn from the same distribution (held out)
    test_y = [sample_class() for _ in range(m)]
    test_X = [sample_point(c) for c in test_y]

    return {"name": f"pool{seed}", "d": d, "n": n, "budget": budget,
            "K": K, "pool_X": pool_X, "pool_y": pool_y,
            "test_X": test_X, "test_y": test_y}


def _instances():
    """Fixed instance distribution (medium scale). (seed,d,K,n,m,budget,sep,sigma)."""
    specs = [
        (1011, 8, 6, 90, 300, 18, 6.0, 1.0),
        (1027, 8, 6, 100, 300, 18, 6.0, 1.0),
        (1043, 8, 5, 90, 300, 16, 6.0, 1.0),
        (1061, 10, 6, 100, 300, 20, 6.5, 1.0),
        (1088, 8, 7, 110, 300, 20, 6.0, 1.0),
        (1103, 8, 6, 96, 300, 18, 5.5, 1.1),
        (1129, 10, 6, 100, 300, 18, 6.0, 1.0),
        (1147, 8, 5, 84, 300, 16, 6.0, 1.0),
        # harder / larger held-out instances (bigger pool, more classes,
        # smaller separation or tighter budget -> coverage is scarcer)
        (2011, 9, 7, 120, 350, 22, 5.5, 1.1),
        (2029, 10, 8, 130, 350, 22, 5.5, 1.1),
        (2053, 8, 7, 110, 350, 18, 5.0, 1.2),
        (2077, 12, 8, 130, 350, 24, 5.5, 1.1),
    ]
    return [_build_instance(*s) for s in specs]


# ----------------------------- fixed model ---------------------------------
def _sqdist(a, b):
    s = 0.0
    for j in range(len(a)):
        dd = a[j] - b[j]
        s += dd * dd
    return s


def _centroids(X, y, idxs):
    """Nearest-centroid training on the labeled subset given by idxs."""
    sums = {}
    cnts = {}
    d = len(X[0])
    for i in idxs:
        c = y[i]
        if c not in sums:
            sums[c] = [0.0] * d
            cnts[c] = 0
        row = X[i]
        acc = sums[c]
        for j in range(d):
            acc[j] += row[j]
        cnts[c] += 1
    cents = {}
    for c in sums:
        cents[c] = [v / cnts[c] for v in sums[c]]
    return cents


def _accuracy(cents, test_X, test_y):
    if not cents:
        return 0.0
    items = list(cents.items())
    correct = 0
    for t in range(len(test_X)):
        x = test_X[t]
        best_c = None
        best_d = None
        for c, ctr in items:
            dd = _sqdist(x, ctr)
            if best_d is None or dd < best_d:
                best_d = dd
                best_c = c
        if best_c == test_y[t]:
            correct += 1
    return correct / len(test_X)


def _aulc(order, inst, B):
    """Area under the learning curve for an ordering: mean test accuracy over
    budgets k=1..B, retraining nearest-centroid on order[:k]."""
    X = inst["pool_X"]
    y = inst["pool_y"]
    tX = inst["test_X"]
    tY = inst["test_y"]
    total = 0.0
    # incremental centroid maintenance
    sums = {}
    cnts = {}
    d = inst["d"]
    for k in range(1, B + 1):
        i = order[k - 1]
        c = y[i]
        if c not in sums:
            sums[c] = [0.0] * d
            cnts[c] = 0
        row = X[i]
        acc = sums[c]
        for j in range(d):
            acc[j] += row[j]
        cnts[c] += 1
        cents = {c2: [v / cnts[c2] for v in sums[c2]] for c2 in sums}
        total += _accuracy(cents, tX, tY)
    return total / B


def _baseline_aulc(inst, B):
    """Mean AULC over several fixed seeded random permutations (weak anchor)."""
    n = inst["n"]
    reps = 8
    acc = 0.0
    for r in range(reps):
        u = _rng(9176 + 31 * r + 7 * inst["n"] + 101 * inst["K"])
        perm = list(range(n))
        for i in range(n - 1, 0, -1):
            j = int(u() * (i + 1))
            if j > i:
                j = i
            perm[i], perm[j] = perm[j], perm[i]
        acc += _aulc(perm, inst, B)
    return acc / reps


def _ceiling(inst):
    """Immediate-jump upper bound: accuracy with the WHOLE pool labeled."""
    cents = _centroids(inst["pool_X"], inst["pool_y"], list(range(inst["n"])))
    return _accuracy(cents, inst["test_X"], inst["test_y"])


# ----------------------------- validation ----------------------------------
def _valid_order(answer, n):
    if not isinstance(answer, dict):
        return None
    order = answer.get("order")
    if not isinstance(order, list) or len(order) != n:
        return None
    seen = [False] * n
    out = []
    for v in order:
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        if v < 0 or v >= n or seen[v]:
            return None
        seen[v] = True
        out.append(v)
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _instances()

    vec = []
    for inst in instances:
        n = inst["n"]
        B = inst["budget"]
        base = _baseline_aulc(inst, B)
        ceil = _ceiling(inst)
        denom = ceil - base
        if denom < 1e-9:
            denom = 1e-9

        public = {"name": inst["name"], "d": inst["d"], "n": n,
                  "budget": B, "pool": [list(row) for row in inst["pool_X"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            order = _valid_order(ans, n)
        except Exception:
            order = None
        if order is None:
            vec.append(0.0)
            continue
        try:
            cand_aulc = _aulc(order, inst, B)
        except Exception:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (cand_aulc - base) / denom
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
