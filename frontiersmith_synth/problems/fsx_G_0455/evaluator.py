#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0455 -- "Fraud Feature Sieve: Selecting Signals for a
Frozen Scorer" (family: ml-feature-selection; format B, quality-metric).

THEME.  A payments company screens card transactions for fraud.  Its risk team logs
a WIDE table of engineered features per transaction (velocity counters, device
fingerprints, geo mismatches, merchant-category flags, ...), but most of them are
noise -- only a handful actually separate fraud from legitimate activity, and the
finite training log gives many *pure-noise* columns a deceptive in-sample
correlation with the fraud label.  Downstream, a FROZEN, fixed scoring model is
retrained nightly on WHATEVER feature subset the risk team hands it.  That frozen
model is a standardized nearest-centroid classifier (a.k.a. diagonal-covariance /
"mean-difference" scorer): it standardizes each chosen column on the training log,
computes a fraud centroid and a legitimate centroid, and labels a new transaction
by the nearer centroid in the selected subspace.  Because every extra irrelevant
column injects distance noise, feeding the frozen model ALL columns is badly hurt by
the curse of dimensionality -- yet dropping a genuinely informative column loses
signal.  The risk team's job (the CANDIDATE) is to pick the feature SUBSET, seeing
ONLY the training log.

TASK (the model writes a feature-SELECTION heuristic).  Given the public training
log (features + fraud labels) the candidate outputs a subset of column indices.  The
evaluator retrains the FROZEN nearest-centroid model on exactly that subset and
measures accuracy on a HELD-OUT transaction set the candidate never sees.  Score =
geometric mean of per-dataset normalized accuracy across a bank of fraud datasets.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "n_features": F (int),
             "n_train":   Ntr (int),
             "X_train": [[f_0,...,f_{F-1}], ...]   # Ntr rows of floats (the log)
             "y_train": [0/1, ...]}                # Ntr fraud labels (1 = fraud)
  stdout: ONE JSON object:
            {"features": [j_0, j_1, ...]}          # DISTINCT column indices, each
                                                     0 <= j < F, at least one, at
                                                     most F.  (Order is ignored.)

  A selection is VALID iff "features" is a non-empty list of distinct integers, every
  index in [0, F).  Anything else (crash, timeout, non-JSON, null, wrong type,
  out-of-range/duplicate indices, empty list) -> that dataset scores 0.0.

FROZEN MODEL (identical for every subset S; deterministic; no wall-time).
  Fit on the training log restricted to S:
    - per column j in S: mu_j, sd_j = mean, std over the TRAIN rows (sd<1e-9 -> 1).
    - centroid_c[j] = mean over train rows of class c of (x_j - mu_j)/sd_j, c in {0,1}.
  Predict a held-out row x:  choose the class c minimizing
        sum_{j in S} ( (x_j - mu_j)/sd_j - centroid_c[j] )^2
  (ties -> the majority training class).  Accuracy = fraction correct on the held-out
  transactions.

SCORING (deterministic).  Per dataset the evaluator computes two internal anchors on
the SAME held-out set:
    a_all    = frozen-model accuracy using ALL F columns        # weak baseline (0.1)
    a_oracle = frozen-model accuracy using the true planted     # strong anchor (1.0)
               informative columns
    a_cand   = frozen-model accuracy using the candidate subset
  and normalizes with an affine anchor (weak baseline -> 0.1, planted ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (a_cand - a_all) / max(a_oracle - a_all, 0.02), 0, 1 )
  A valid selection is floored at 0.02 so a merely-bad (but legal) subset stays
  distinguishable from an INVALID one (which scores exactly 0.0).  Reproducing the
  all-columns baseline scores ~0.1; matching the planted informative set scores ~1.0;
  the finite log's spurious correlations keep even good filters strictly below 1.0 on
  most datasets -> headroom.  The overall Ratio is the GEOMETRIC MEAN of the r's, so a
  heuristic must select well across the WHOLE dataset bank, not just the easy ones.

ISOLATION.  The candidate is untrusted and runs OS-sandboxed in a FRESH SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC training log.  The held-out
transactions, the planted-informative identities, and the anchors a_all / a_oracle are
computed by THIS parent process, so a frame-walking / filesystem-snooping candidate
learns nothing exploitable.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <geometric mean of r over all datasets, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)

    def u():
        nonlocal state
        state = (state * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)
        return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    return u


def _normals(u, k):
    """k i.i.d. standard normals via Box-Muller (deterministic given u)."""
    out = []
    while len(out) < k:
        a = u()
        b = u()
        if a < 1e-12:
            a = 1e-12
        r = math.sqrt(-2.0 * math.log(a))
        t = 2.0 * math.pi * b
        out.append(r * math.cos(t))
        out.append(r * math.sin(t))
    return out[:k]


# ----------------------------- instance family -----------------------------
# (seed, F, n_inf, n_train, n_test, fraud_rate, signal_scale)
_SPECS = [
    (102, 150, 7, 350, 400, 0.40, 0.50),
    (103, 120, 6, 350, 400, 0.45, 0.55),
    (104, 180, 8, 380, 400, 0.40, 0.50),
    (105, 150, 6, 320, 400, 0.40, 0.45),
    (106, 130, 6, 400, 400, 0.45, 0.55),
    (107, 200, 8, 360, 400, 0.40, 0.50),
    # harder / larger held-out datasets
    (108, 150, 7, 340, 500, 0.40, 0.50),
    (211, 170, 7, 420, 500, 0.40, 0.50),
    (212, 190, 8, 380, 500, 0.40, 0.45),
    (213, 150, 6, 340, 500, 0.45, 0.55),
    (214, 140, 6, 360, 500, 0.40, 0.55),
    (215, 160, 7, 360, 500, 0.40, 0.50),
]


def _build_one(seed, F, n_inf, n_train, n_test, fraud, sig):
    u = _rng(seed)
    # informative columns: evenly spread across the table
    inf = []
    step = max(1, F // n_inf)
    idx = 0
    while len(inf) < n_inf and idx < F:
        inf.append(idx)
        idx += step
    inf = inf[:n_inf]
    infset = set(inf)
    # per-informative class-mean shift (varied strength; some weak, some strong)
    deltas = {j: sig * (0.4 + 1.1 * u()) for j in inf}
    n = n_train + n_test
    X = []
    Y = []
    for _ in range(n):
        y = 1 if u() < fraud else 0
        z = _normals(u, F)
        row = [0.0] * F
        for j in range(F):
            row[j] = z[j] + (deltas[j] if (j in infset and y == 1) else 0.0)
        X.append(row)
        Y.append(y)
    return {
        "name": "fraud%d" % seed,
        "n_features": F,
        "n_train": n_train,
        "X_train": [X[i] for i in range(n_train)],
        "y_train": [Y[i] for i in range(n_train)],
        # hidden (never sent to the candidate):
        "X_test": [X[i] for i in range(n_train, n)],
        "y_test": [Y[i] for i in range(n_train, n)],
        "informative": inf,
    }


def _build_instances():
    return [_build_one(*spec) for spec in _SPECS]


# ----------------------------- FROZEN model --------------------------------
def _accuracy(Xtr, Ytr, Xte, Yte, sel):
    """Standardized nearest-centroid accuracy on (Xte,Yte) using columns `sel`."""
    if not sel:
        return 0.0
    ntr = len(Xtr)
    mu = {}
    sd = {}
    for j in sel:
        m = 0.0
        for i in range(ntr):
            m += Xtr[i][j]
        m /= ntr
        v = 0.0
        for i in range(ntr):
            d = Xtr[i][j] - m
            v += d * d
        v /= ntr
        mu[j] = m
        sd[j] = math.sqrt(v) if v > 1e-12 else 1.0
    cnt = {0: 0, 1: 0}
    sums = {0: {j: 0.0 for j in sel}, 1: {j: 0.0 for j in sel}}
    for i in range(ntr):
        c = Ytr[i]
        cnt[c] += 1
        for j in sel:
            sums[c][j] += (Xtr[i][j] - mu[j]) / sd[j]
    cen = {0: {}, 1: {}}
    for c in (0, 1):
        for j in sel:
            cen[c][j] = sums[c][j] / cnt[c] if cnt[c] > 0 else 0.0
    maj = 0 if cnt[0] >= cnt[1] else 1
    correct = 0
    for i in range(len(Xte)):
        d0 = 0.0
        d1 = 0.0
        for j in sel:
            z = (Xte[i][j] - mu[j]) / sd[j]
            e0 = z - cen[0][j]
            e1 = z - cen[1][j]
            d0 += e0 * e0
            d1 += e1 * e1
        if d0 < d1:
            p = 0
        elif d1 < d0:
            p = 1
        else:
            p = maj
        if p == Yte[i]:
            correct += 1
    return correct / len(Xte)


# ----------------------------- validation ----------------------------------
def _validate(answer, F):
    """Return a validated, de-duplicated list of column indices, or None."""
    if not isinstance(answer, dict):
        return None
    feats = answer.get("features")
    if not isinstance(feats, list) or len(feats) == 0 or len(feats) > F:
        return None
    seen = set()
    for j in feats:
        if isinstance(j, bool) or not isinstance(j, int):
            return None
        if j < 0 or j >= F:
            return None
        if j in seen:
            return None          # duplicates are not a valid subset
        seen.add(j)
    return sorted(seen)


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        Xtr = inst["X_train"]
        Ytr = inst["y_train"]
        Xte = inst["X_test"]
        Yte = inst["y_test"]
        F = inst["n_features"]

        a_all = _accuracy(Xtr, Ytr, Xte, Yte, list(range(F)))
        a_oracle = _accuracy(Xtr, Ytr, Xte, Yte, inst["informative"])
        denom = a_oracle - a_all
        if denom < 0.02:
            denom = 0.02

        public = {"name": inst["name"], "n_features": F,
                  "n_train": inst["n_train"],
                  "X_train": [list(row) for row in Xtr],
                  "y_train": list(Ytr)}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            sel = _validate(ans, F)
        except Exception:
            sel = None
        if sel is None:
            vec.append(0.0)
            continue

        a_cand = _accuracy(Xtr, Ytr, Xte, Yte, sel)
        r = 0.1 + 0.9 * (a_cand - a_all) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        if r < 0.02:        # valid-but-bad subset stays > 0 (distinct from INVALID)
            r = 0.02
        vec.append(r)

    # geometric mean over datasets (matches the "gmean held-out accuracy" objective)
    if vec and all(x > 0 for x in vec):
        ratio = math.exp(sum(math.log(x) for x in vec) / len(vec))
    else:
        ratio = 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
