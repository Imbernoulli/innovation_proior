#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0458 -- "Rare-Disease Screen: Resampling/Reweighting Policy"
(family: ml-imbalance-resampling; format B, quality-metric).

THEME.  A hospital runs a multi-class screening assay for a rare disease.  Each
patient is described by a vector of standardized biomarker readings.  The label
is one of THREE classes:
    0 = healthy / no finding      (the overwhelming majority),
    1 = rare disease subtype A,
    2 = rare disease subtype B.
Because the two disease subtypes are rare, a classifier trained on the raw cohort
collapses onto "predict healthy for everyone": it looks accurate but MISSES the
patients who actually need care.  The quality metric that matters here is the
MACRO-averaged F1 (each class weighted equally, so the rare subtypes count as much
as the healthy majority) on a balanced held-out validation cohort.

THE TASK the candidate solves.  A FIXED classifier -- a weighted multinomial
logistic-regression trained by a FIXED, deterministic full-batch gradient descent
-- is baked into THIS evaluator and never changes.  The only lever the candidate
controls is a RESAMPLING / REWEIGHTING POLICY: it assigns a non-negative training
WEIGHT to every training patient.  Duplicating (oversampling) a patient is the same
as doubling its weight, and dropping (undersampling) a patient is the same as
zeroing its weight, so per-example weights express the full family of
resampling/reweighting policies.  The evaluator trains the fixed classifier on the
supplied weights and reports macro-F1 on the HIDDEN validation cohort.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "n_classes": 3,
             "feature_dim": D (int),
             "n": N (int),                       # number of training patients
             "y": [c_0, ..., c_{N-1}],           # training labels in {0,1,2}
             "class_counts": [n0, n1, n2],       # counts of each class in the train set
             "X": [[...D floats...], ... N rows]} # standardized biomarker vectors
  stdout: ONE JSON object:
            {"weights": [w_0, ..., w_{N-1}]}      # w_i >= 0, finite, aligned to X/y order
          The weights are re-scaled to mean 1 before training (only their RELATIVE
          magnitudes matter), so the total training "budget" is fixed and a policy
          cannot win merely by inflating every weight.

  A policy is VALID iff `weights` is a list of exactly N finite non-negative reals
  whose sum is strictly positive.  Wrong length, a negative / NaN / Inf entry, an
  all-zero vector, a crash, a timeout, or non-JSON  ->  that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance we compute three references
with the SAME fixed classifier:
    f_base = macro-F1 of the classifier trained with UNIFORM weights (ignores the
             imbalance -- the weak, trivial baseline),
    f_ref  = macro-F1 of the classifier trained on a LARGE, perfectly BALANCED clean
             cohort drawn from the true generative model (an idealized ceiling that
             the small, imbalanced training set can never quite reach),
    f_cand = macro-F1 of the classifier trained with the candidate's weights,
  and normalize with an affine anchor (uniform baseline -> 0.1, ideal cohort -> 1.0):
    r = clamp( 0.1 + 0.9 * (f_cand - f_base) / max(1e-9, f_ref - f_base), 0, 1 )
  A policy reproducing uniform weights scores ~0.1; a policy matching the (generally
  unreachable) balanced-ideal ceiling scores 1.0; a policy that hurts the rare
  classes relative to uniform scores < 0.1.  Because f_ref is estimated from far more
  (and balanced) rare-class data than any reweighting of 5-22 rare patients can
  recover, even the textbook inverse-frequency policy stays below 1.0 -> headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH OS-SANDBOXED SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC instance.  The hidden
validation cohort and all references (f_base, f_ref) are computed by THIS parent
process, so a frame-walking / filesystem-probing candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import numpy as np
import isorun

K = 3                     # number of classes: healthy, subtype A, subtype B
GD_STEPS = 300            # fixed classifier: full-batch gradient-descent steps
GD_LR = 0.5              # fixed learning rate
GD_L2 = 2e-3            # fixed L2 regularization


# --------------------------- deterministic data model ----------------------
def _class_means(seed, seps, D):
    """Deterministic class-conditional means. Class 0 sits near the origin
    (healthy); the two rare subtypes are offset along random unit directions
    scaled by their separations `seps` (which control per-class difficulty)."""
    rs = np.random.RandomState(seed * 7 + 1)
    dirs = rs.randn(K, D)
    M = np.zeros((K, D))
    for k in range(K):
        M[k] = dirs[k] / (np.linalg.norm(dirs[k]) + 1e-12) * seps[k]
    M[0] = rs.randn(D) * 0.1
    return M


def _draw(seed, counts, M, noise, D):
    """Draw class-conditional Gaussian samples. Deterministic in `seed`."""
    rs = np.random.RandomState(seed)
    Xs, ys = [], []
    for k in range(K):
        Xs.append(rs.randn(counts[k], D) * noise + M[k])
        ys += [k] * counts[k]
    X = np.vstack(Xs)
    y = np.array(ys, dtype=int)
    perm = rs.permutation(len(y))
    return X[perm], y[perm]


# --------------------------- the FIXED classifier --------------------------
def _train(X, y, w):
    """Weighted multinomial logistic regression by fixed full-batch GD.
    `X` is already standardized. Weights are rescaled to mean 1. Deterministic."""
    n, d = X.shape
    W = np.zeros((K, d))
    b = np.zeros(K)
    wn = w / (w.mean() + 1e-12)
    Y = np.zeros((n, K))
    Y[np.arange(n), y] = 1.0
    for _ in range(GD_STEPS):
        Z = X @ W.T + b
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        G = (P - Y) * wn[:, None]
        W -= GD_LR * (G.T @ X / n + GD_L2 * W)
        b -= GD_LR * (G.sum(axis=0) / n)
    return W, b


def _macro_f1(yt, yp):
    fs = []
    for k in range(K):
        tp = int(np.sum((yp == k) & (yt == k)))
        fp = int(np.sum((yp == k) & (yt != k)))
        fn = int(np.sum((yp != k) & (yt == k)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        fs.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(fs))


def _score_with_weights(Xtr, ytr, Xte, yte, w):
    """Standardize with train stats, train fixed classifier, macro-F1 on test."""
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0) + 1e-9
    W, b = _train((Xtr - mean) / std, ytr, w)
    pred = (((Xte - mean) / std) @ W.T + b).argmax(axis=1)
    return _macro_f1(yte, pred)


# --------------------------- instance family -------------------------------
# (seed, [n_healthy, n_A, n_B], noise, [sep0, sepA, sepB])
_SPECS = [
    (201, [360, 16, 8], 1.20, [0.1, 2.8, 1.6]),
    (202, [320, 12, 7], 1.25, [0.1, 1.7, 2.8]),
    (203, [400, 18, 9], 1.15, [0.1, 2.6, 1.7]),
    (204, [300, 14, 7], 1.20, [0.1, 1.9, 2.5]),
    (205, [380, 20, 11], 1.20, [0.1, 2.2, 2.2]),
    (206, [340, 15, 9], 1.25, [0.1, 2.4, 2.0]),
    (207, [300, 13, 8], 1.30, [0.1, 2.1, 2.3]),
    (208, [420, 22, 10], 1.15, [0.1, 2.5, 2.1]),
    # harder held-out cohorts: fewer rare patients, noisier assay
    (301, [300, 10, 6], 1.35, [0.1, 1.8, 2.6]),
    (302, [280, 9, 6], 1.40, [0.1, 2.6, 1.8]),
    (303, [350, 11, 6], 1.35, [0.1, 2.0, 2.4]),
    (304, [260, 8, 5], 1.45, [0.1, 2.3, 1.9]),
]

D_FEATURES = 8
_TEST_PER_CLASS = 150
_IDEAL_PER_CLASS = 1200


def _build_instances():
    """Deterministic instances. public = train cohort seen by the candidate;
    hidden = held-out validation cohort + precomputed references."""
    out = []
    for seed, counts, noise, seps in _SPECS:
        M = _class_means(seed, seps, D_FEATURES)
        Xtr, ytr = _draw(seed, counts, M, noise, D_FEATURES)
        Xte, yte = _draw(seed + 9999, [_TEST_PER_CLASS] * K, M, noise, D_FEATURES)
        Xideal, yideal = _draw(seed + 5555, [_IDEAL_PER_CLASS] * K, M, noise, D_FEATURES)
        n = len(ytr)
        class_counts = [int(np.sum(ytr == k)) for k in range(K)]
        f_base = _score_with_weights(Xtr, ytr, Xte, yte, np.ones(n))
        f_ref = _score_with_weights(Xideal, yideal, Xte, yte, np.ones(len(yideal)))
        public = {
            "name": f"screen{seed}",
            "n_classes": K,
            "feature_dim": D_FEATURES,
            "n": n,
            "y": [int(v) for v in ytr],
            "class_counts": class_counts,
            "X": [[float(v) for v in row] for row in Xtr],
        }
        hidden = {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte,
                  "f_base": f_base, "f_ref": f_ref}
        out.append({"public": public, "hidden": hidden})
    return out


# --------------------------- answer validation -----------------------------
def _score(inst, answer):
    """Validate the policy strictly, then train + evaluate the fixed classifier.
    Returns (ok, f_cand)."""
    if not isinstance(answer, dict):
        return False, 0.0
    w = answer.get("weights")
    if not isinstance(w, list):
        return False, 0.0
    N = inst["public"]["n"]
    if len(w) != N:
        return False, 0.0
    ww = np.empty(N, dtype=float)
    for i, v in enumerate(w):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False, 0.0
        fv = float(v)
        if not math.isfinite(fv) or fv < 0.0:
            return False, 0.0
        ww[i] = fv
    if ww.sum() <= 0.0:
        return False, 0.0
    h = inst["hidden"]
    f_cand = _score_with_weights(h["Xtr"], h["ytr"], h["Xte"], h["yte"], ww)
    return True, f_cand


# --------------------------- scoring driver --------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        h = inst["hidden"]
        denom = h["f_ref"] - h["f_base"]
        if denom < 1e-9:
            denom = 1e-9
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, f_cand = _score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (f_cand - h["f_base"]) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
