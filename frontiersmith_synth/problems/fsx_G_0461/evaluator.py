#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0461 -- "Robust Regression Head: Fit a Loss for Corrupted Sensors"
(family: ml-loss-function-design; format B, quality-metric; objective = MINIMIZE held-out error).

THEME.  A calibration lab fits a LINEAR REGRESSION HEAD y ~ x.w + b that maps a handful of raw
sensor features to a target reading.  The catch: a chunk of the TRAINING labels are corrupted --
a fraction of the sensors log wild, gross outliers (spikes / dropouts) with no warning flag.  The
held-out validation set is CLEAN.  The job is to DESIGN THE LOSS / FITTING PROCEDURE for the head
so the learned weights generalize to the clean distribution despite the corrupted training labels.

This is the classic "design a (sub)differentiable loss, fixed optimizer/iterations, score = final
held-out error" task (MLS-Bench regression-head shape).  Plain least-squares (a squared/L2 loss)
chases the outliers and generalizes poorly; a robust loss (Huber / absolute / Tukey biweight /
trimmed / quantile ...) or an explicit outlier-rejection procedure recovers the true head.  There
is NO single best loss -- delta / breakdown / reweighting schedule all trade off -- so the space of
viable fitting strategies is genuinely open.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "d": int, "n_train": int,
             "X_train": [[float]*d]*n_train,   # training features
             "y_train": [float]*n_train}       # training targets (a fraction are CORRUPTED)
          NOTE: the held-out validation set (X_test, y_test) is NEVER shown to the candidate.
  stdout: ONE JSON object giving the learned regression head:
            {"w": [float]*d, "b": float}
          Prediction on a validation point x is  sum_j w[j]*x[j] + b.

  A submission is VALID iff `w` is a list of exactly d finite numbers and `b` is a finite number.
  Wrong length, a non-finite (NaN/Inf) weight, a crash, a timeout, or non-JSON -> that instance
  scores 0.0.

SCORING (deterministic; no wall-time).  Per instance the evaluator holds out a CLEAN validation set
and computes mean squared error on it:
    obj   = mean_i ( (pred_i - y_test_i)^2 )                 # candidate's held-out MSE (MINIMIZE)
    base  = mean_i ( (mean(y_train) - y_test_i)^2 )          # trivial "predict the training mean" MSE
  and normalizes (minimization anchor: trivial baseline -> ~0.1, headroom left above):
    r = min( 1.0, 0.1 * base / max(obj, 1e-12) )
  Predicting the (outlier-skewed) training mean scores ~0.1; a head that recovers the clean signal
  drives obj well below `base` and scores higher.  Because the clean validation noise floor is
  strictly positive (obj can never reach 0), even an excellent robust fit stays below 1.0 -> headroom.

ISOLATION.  The candidate is untrusted and runs OS-sandboxed in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The clean validation set (X_test,
y_test), the true weights, and the baseline are computed by THIS parent process, so a frame-walking /
introspecting / source-reading candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math, random
import isorun


# ----------------------------- instance family -----------------------------
def _gen_instance(seed, n_train, d, sigma, rho, out_scale, n_test=400):
    """Deterministic robust-regression instance.

    Training labels: y = X.w* + b* + N(0,sigma), then a fraction rho of them are
    replaced by gross symmetric outliers of magnitude ~out_scale.  Validation set
    is CLEAN (same signal, no outliers).  Returns (public, hidden)."""
    rng = random.Random(seed)

    # true head; scale chosen so Var(X.w*) ~ 6-7 (features are unit-variance)
    w_true = [rng.uniform(-1.8, 1.8) for _ in range(d)]
    b_true = rng.uniform(-1.0, 1.0)

    def _row():
        return [rng.gauss(0.0, 1.0) for _ in range(d)]

    def _signal(x):
        return sum(w_true[j] * x[j] for j in range(d)) + b_true

    # ---- training set (corrupted) ----
    X_train, y_train = [], []
    for _ in range(n_train):
        x = _row()
        y = _signal(x) + rng.gauss(0.0, sigma)
        X_train.append(x)
        y_train.append(y)

    # corrupt a deterministic fraction of the training labels with big symmetric spikes
    k = int(round(rho * n_train))
    idx = list(range(n_train))
    rng.shuffle(idx)
    for i in idx[:k]:
        spike = out_scale * rng.uniform(0.7, 1.4) * (1 if rng.random() < 0.5 else -1)
        y_train[i] = _signal(X_train[i]) + spike

    # ---- clean held-out validation set ----
    X_test, y_test = [], []
    for _ in range(n_test):
        x = _row()
        y = _signal(x) + rng.gauss(0.0, sigma)
        X_test.append(x)
        y_test.append(y)

    public = {"name": f"cal_{seed}", "d": d, "n_train": n_train,
              "X_train": X_train, "y_train": y_train}
    hidden = {"X_test": X_test, "y_test": y_test}
    return public, hidden


def _build_instances():
    """Fixed, seeded instance distribution. (seed, n_train, d, sigma, rho, out_scale)."""
    specs = [
        # --- core: moderate corruption, plenty of signal ---
        (1001, 60, 6, 1.0, 0.20, 10.0),
        (1002, 60, 6, 1.0, 0.25, 10.0),
        (1003, 70, 6, 1.0, 0.20, 9.0),
        (1004, 55, 6, 1.2, 0.25, 11.0),
        (1005, 65, 6, 1.0, 0.30, 10.0),
        (1006, 80, 6, 0.9, 0.20, 10.0),
        # --- more features (higher signal var -> larger sigma to keep headroom) ---
        (1007, 70, 8, 1.2, 0.25, 10.0),
        (1008, 75, 8, 1.2, 0.20, 11.0),
        # --- held-out HARD: heavy corruption + fewer samples ---
        (1009, 50, 6, 1.0, 0.35, 12.0),
        (1010, 48, 6, 1.2, 0.35, 13.0),
        (1011, 55, 8, 1.1, 0.38, 12.0),
        (1012, 45, 6, 1.3, 0.40, 14.0),
    ]
    out = []
    for s in specs:
        pub, hid = _gen_instance(*s)
        out.append({"public": pub, "hidden": hid})
    return out


# ----------------------------- references ----------------------------------
def _baseline(pub, hid):
    """Trivial construction: predict the (outlier-skewed) training-label mean."""
    y_tr = pub["y_train"]
    c = sum(y_tr) / len(y_tr)
    yt = hid["y_test"]
    return sum((c - v) ** 2 for v in yt) / len(yt)


# ----------------------------- validation ----------------------------------
def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite(v):
    return _is_num(v) and (v == v) and v not in (float("inf"), float("-inf"))


def _objective(pub, hid, answer):
    """Validate the head and return held-out MSE, or None if invalid."""
    if not isinstance(answer, dict):
        return None
    w = answer.get("w")
    b = answer.get("b")
    d = pub["d"]
    if not isinstance(w, list) or len(w) != d:
        return None
    for v in w:
        if not _finite(v):
            return None
    if not _finite(b):
        return None
    w = [float(v) for v in w]
    b = float(b)
    X = hid["X_test"]
    yt = hid["y_test"]
    se = 0.0
    for x, y in zip(X, yt):
        pred = b
        for j in range(d):
            pred += w[j] * x[j]
        e = pred - y
        se += e * e
    obj = se / len(yt)
    if not _finite(obj):
        return None
    return obj


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        pub, hid = inst["public"], inst["hidden"]
        base = _baseline(pub, hid)
        ans, st = isorun.run_candidate(cand, pub, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            obj = _objective(pub, hid, ans)
        except Exception:
            obj = None
        if obj is None:
            vec.append(0.0)
            continue
        r = 0.1 * base / max(obj, 1e-12)
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
