#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0191 -- "Data-Center Cooling: A Drop-In Thermal Response
Nonlinearity for Cross-Hall Load Classifiers"
(family: modular-component-cross-setting; eval_form: quality-metric; MLS-Bench dl-*/
llm-* shape, CPU proxies only).

THEME.  An operator runs many cooling halls.  In each hall a *tiny* neural classifier
reads a handful of thermal/airflow sensor features and predicts the hall's discrete
operating regime (which cooling response is appropriate / whether a rack cluster is
hotspotting).  The classifiers are architecturally identical up to size, but each uses
ONE shared, hand-designed nonlinearity -- a "thermal response curve" -- as its hidden
activation.  We want a SINGLE drop-in activation function that makes these small
CPU-trainable classifiers accurate across a DIVERSE family of halls, including a hidden
held-out hall the designer never sees.

WHAT THE CANDIDATE DESIGNS.  A drop-in scalar activation phi: R -> R, supplied as the
y-values of a piecewise-linear curve over a FIXED knot grid xs (linear interpolation
between knots; linear extrapolation with the boundary segment's slope outside the grid).
The evaluator plugs phi into a one-hidden-layer MLP and trains it FROM SCRATCH with
deterministic full-batch gradient descent (fixed seed / init / data / epochs) on each
hall, then measures held-out test accuracy.  The candidate is a drop-in *component*: it
edits ONLY the activation; the architecture, optimizer, data and training budget are
frozen by the evaluator.

ISOLATION.  The candidate is UNTRUSTED and runs as an isolated stdin->stdout subprocess
(isorun).  It reads ONE JSON "public instance" (a description of the hall's classifier
-- NOT the labelled data) and writes ONE JSON answer (the activation knot values).  The
sensor data, the labels, the trained weights and this evaluator's memory are never
exposed to the candidate.  All the candidate can do is propose a curve.

Public instance JSON (candidate stdin):
    {
      "setting":    str,          # hall family name (for optional adaptation)
      "n_features": int,          # input dimension d
      "n_classes":  int,          # number of regimes C
      "hidden":     int,          # hidden width H (frozen)
      "grid":       [float,...],  # K fixed knot x-positions (linspace, symmetric)
      "n_knots":    int,          # K
      "epochs":     int,          # training budget (frozen; informational)
      "seed":       int           # a per-instance seed the candidate MAY use
    }

Answer JSON (candidate stdout) -- EITHER of:
    [float, ..., float]                 # length K activation values phi(grid[k])
    {"activation": [float, ..., float]} # same, wrapped

SCORING.  For each hall the evaluator trains the SAME network twice with the SAME seed,
data and budget, differing only in the activation:
  * a reference baseline activation = the IDENTITY curve (phi(x)=x), which makes the
    one-hidden-layer net collapse to a linear classifier;
  * the candidate's curve.
Let acc_cand / acc_base be the resulting test accuracies.  The per-hall normalized score
anchors the candidate against the evaluator's own baseline:

    r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / max(1 - acc_base, MIN_DENOM), 0, 1 )

so reproducing the baseline (the identity curve) maps to ~0.1 and a much-better curve
climbs toward 1.0, with headroom kept by the noisy, imperfectly-separable halls (no
curve reaches 1.0 test accuracy).  The final Ratio is the GEOMETRIC MEAN of the per-hall
r, so a curve that helps some halls but destabilises another is heavily penalised --
this rewards a nonlinearity that GENERALISES across settings, not one tuned to a single
hall.  An instance where the candidate errors, returns the wrong shape / non-finite
values, or makes training diverge (non-finite loss/weights) scores exactly 0.0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <geometric mean of per-hall r, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import os
# Cap BLAS/OMP threads so numpy imports cleanly inside isorun's memory-capped child and
# so the parent's training is deterministic regardless of core count.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys, json, math
import numpy as np
import isorun

# ----------------------------- fixed configuration -------------------------
K_KNOTS = 21
GRID = np.linspace(-4.0, 4.0, K_KNOTS)          # symmetric knot grid
MAX_ABS_Y = 1.0e3                               # reject absurd activation magnitudes
VALID_FLOOR = 0.02                              # floor for VALID halls so gmean stays defined
MIN_DENOM = 0.20                                # keeps headroom when baseline is already strong
CAND_TIMEOUT = 20

# frozen training hyper-parameters (shared by baseline + candidate)
LR = 0.30
WD = 1.0e-4
GRAD_CLIP = 5.0                                 # global-norm clip -> stable across curves


# ============================ hall family (instances) ======================
def _standardize_split(rng, X, y, test_frac=0.4, noise=0.0):
    """Shuffle, add optional feature noise, split, standardize by TRAIN stats."""
    n = X.shape[0]
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    if noise > 0.0:
        X = X + rng.normal(0.0, noise, size=X.shape)
    n_test = int(round(n * test_frac))
    Xte, yte = X[:n_test], y[:n_test]
    Xtr, ytr = X[n_test:], y[n_test:]
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    return (Xtr.astype(np.float64), ytr.astype(np.int64),
            Xte.astype(np.float64), yte.astype(np.int64))


def _gen_xor(seed, n, d, noise):
    """XOR-style: 2 informative axes (4 clusters, class = quadrant parity), plus
    (d-2) pure-noise axes.  Not linearly separable -> a linear model is at chance."""
    rng = np.random.default_rng(seed)
    m = n
    a = rng.integers(0, 2, size=m) * 2 - 1
    b = rng.integers(0, 2, size=m) * 2 - 1
    X = np.zeros((m, d))
    X[:, 0] = a * 1.6 + rng.normal(0, 0.5, m)
    X[:, 1] = b * 1.6 + rng.normal(0, 0.5, m)
    for j in range(2, d):
        X[:, j] = rng.normal(0, 1.0, m)
    y = ((a * b) > 0).astype(np.int64)
    return _standardize_split(rng, X, y, noise=noise)


def _gen_rings(seed, n, d, noise):
    """Concentric shells: inner blob vs outer ring (radial structure).  Linear model
    cannot separate an annulus from its centre."""
    rng = np.random.default_rng(seed)
    m = n
    r_in = rng.uniform(0.0, 1.1, m // 2)
    r_out = rng.uniform(2.4, 3.4, m - m // 2)
    r = np.concatenate([r_in, r_out])
    y = np.array([0] * (m // 2) + [1] * (m - m // 2), dtype=np.int64)
    theta = rng.uniform(0, 2 * np.pi, m)
    X = np.zeros((m, d))
    X[:, 0] = r * np.cos(theta)
    X[:, 1] = r * np.sin(theta)
    for j in range(2, d):
        X[:, j] = rng.normal(0, 1.0, m)
    return _standardize_split(rng, X, y, noise=noise)


def _gen_spiral(seed, n, d, classes, noise):
    """Interleaved spirals (one per class).  Strongly nonlinear."""
    rng = np.random.default_rng(seed)
    per = n // classes
    Xs, ys = [], []
    for c in range(classes):
        t = np.linspace(0.15, 1.0, per) * 3.2
        rad = t
        ang = t * 2.4 + c * (2 * np.pi / classes)
        x0 = rad * np.cos(ang) + rng.normal(0, 0.12, per)
        x1 = rad * np.sin(ang) + rng.normal(0, 0.12, per)
        blk = np.zeros((per, d))
        blk[:, 0] = x0
        blk[:, 1] = x1
        for j in range(2, d):
            blk[:, j] = rng.normal(0, 1.0, per)
        Xs.append(blk)
        ys.append(np.full(per, c, dtype=np.int64))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return _standardize_split(rng, X, y, noise=noise)


def _gen_moons(seed, n, d, noise):
    """Two interleaving half-moons."""
    rng = np.random.default_rng(seed)
    m = n
    h = m // 2
    t0 = np.linspace(0, np.pi, h)
    t1 = np.linspace(0, np.pi, m - h)
    x0 = np.concatenate([np.cos(t0), 1 - np.cos(t1)])
    x1 = np.concatenate([np.sin(t0), 0.5 - np.sin(t1)])
    y = np.array([0] * h + [1] * (m - h), dtype=np.int64)
    X = np.zeros((m, d))
    X[:, 0] = x0 * 2.0 + rng.normal(0, 0.18, m)
    X[:, 1] = x1 * 2.0 + rng.normal(0, 0.18, m)
    for j in range(2, d):
        X[:, j] = rng.normal(0, 1.0, m)
    return _standardize_split(rng, X, y, noise=noise)


def _gen_blobs(seed, n, d, classes, noise):
    """Well-separated Gaussian regimes -- nearly linearly separable, so even the
    identity baseline scores high.  Here the activation barely helps: a source of
    per-hall DIVERGENCE that punishes overfitting to nonlinear halls."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.0, size=(classes, d))
    centers /= (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-9)
    centers *= 4.5
    per = n // classes
    Xs, ys = [], []
    for c in range(classes):
        blk = centers[c] + rng.normal(0, 0.9, size=(per, d))
        Xs.append(blk)
        ys.append(np.full(per, c, dtype=np.int64))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return _standardize_split(rng, X, y, noise=noise)


def _gen_checker(seed, n, d, noise):
    """2x2-per-axis checkerboard parity: highly nonlinear XOR-of-bins."""
    rng = np.random.default_rng(seed)
    m = n
    X = np.zeros((m, d))
    X[:, 0] = rng.uniform(-2.4, 2.4, m)
    X[:, 1] = rng.uniform(-2.4, 2.4, m)
    for j in range(2, d):
        X[:, j] = rng.normal(0, 1.0, m)
    bx = np.floor(X[:, 0] / 1.2).astype(np.int64)
    by = np.floor(X[:, 1] / 1.2).astype(np.int64)
    y = ((bx + by) % 2).astype(np.int64)
    return _standardize_split(rng, X, y, noise=noise)


def _build_instances():
    """Nine halls: several strongly-nonlinear settings (curve matters a lot), one
    near-linear blobs hall (curve barely matters), and three held-out variants
    (higher input dim / more classes / harder noise) the designer cannot tune to."""
    specs = [
        ("xor",     dict(gen="xor",    seed=1101, n=360, d=2, classes=2, noise=0.38)),
        ("rings",   dict(gen="rings",  seed=1102, n=360, d=2, classes=2, noise=0.30)),
        ("spiral2", dict(gen="spiral", seed=1103, n=360, d=2, classes=2, noise=0.10)),
        ("moons",   dict(gen="moons",  seed=1104, n=360, d=2, classes=2, noise=0.12)),
        ("checker", dict(gen="checker",seed=1105, n=400, d=2, classes=2, noise=0.08)),
        ("blobs",   dict(gen="blobs",  seed=1106, n=360, d=3, classes=3, noise=0.15)),
        # ---- held-out variants (higher d / more classes / harder) ----
        ("xor4d",   dict(gen="xor",    seed=1107, n=400, d=4, classes=2, noise=0.35)),
        ("spiral3", dict(gen="spiral", seed=1108, n=420, d=2, classes=3, noise=0.10)),
        ("rings4d", dict(gen="rings",  seed=1109, n=400, d=4, classes=2, noise=0.35)),
    ]
    gens = {"xor": _gen_xor, "rings": _gen_rings, "spiral": _gen_spiral,
            "moons": _gen_moons, "blobs": _gen_blobs, "checker": _gen_checker}
    out = []
    for name, p in specs:
        g = gens[p["gen"]]
        if p["gen"] in ("spiral", "blobs"):
            data = g(p["seed"], p["n"], p["d"], p["classes"], p["noise"])
        else:
            data = g(p["seed"], p["n"], p["d"], p["noise"])
        Xtr, ytr, Xte, yte = data
        out.append({
            "name": name,
            "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte,
            "d": int(Xtr.shape[1]),
            "classes": int(p["classes"]),
            "hidden": 6,
            "epochs": 60,
            "seed": p["seed"],
        })
    return out


# ============================ activation from knots ========================
def _make_activation(ys):
    """Piecewise-linear phi from knot y-values over the fixed GRID; linear
    extrapolation with the boundary slope outside the grid.  Returns (phi, dphi)."""
    xs = GRID
    ys = np.asarray(ys, dtype=np.float64)
    slopes = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])

    def phi(z):
        idx = np.searchsorted(xs, z, side="right") - 1
        idx = np.clip(idx, 0, xs.shape[0] - 2)
        return ys[idx] + slopes[idx] * (z - xs[idx])

    def dphi(z):
        idx = np.searchsorted(xs, z, side="right") - 1
        idx = np.clip(idx, 0, xs.shape[0] - 2)
        return slopes[idx]

    return phi, dphi


# ============================ deterministic MLP trainer ====================
def _train_test_acc(inst, ys):
    """Train a 1-hidden-layer MLP with activation defined by knot values ys on this
    hall; return test accuracy, or None if training diverges (non-finite)."""
    Xtr, ytr = inst["Xtr"], inst["ytr"]
    Xte, yte = inst["Xte"], inst["yte"]
    d, H, C = inst["d"], inst["hidden"], inst["classes"]
    N = Xtr.shape[0]
    phi, dphi = _make_activation(ys)

    rng = np.random.default_rng(inst["seed"] * 7919 + 13)
    W1 = rng.normal(0, 1.0, size=(d, H)) * math.sqrt(2.0 / d)
    b1 = np.zeros(H)
    W2 = rng.normal(0, 1.0, size=(H, C)) * math.sqrt(2.0 / H)
    b2 = np.zeros(C)

    idx_tr = np.arange(N)
    for _ in range(inst["epochs"]):
        Z1 = Xtr @ W1 + b1
        A1 = phi(Z1)
        Z2 = A1 @ W2 + b2
        Z2 = Z2 - Z2.max(axis=1, keepdims=True)
        expZ = np.exp(Z2)
        P = expZ / (expZ.sum(axis=1, keepdims=True) + 1e-12)
        # gradient of cross-entropy
        dZ2 = P.copy()
        dZ2[idx_tr, ytr] -= 1.0
        dZ2 /= N
        dW2 = A1.T @ dZ2 + WD * W2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * dphi(Z1)
        dW1 = Xtr.T @ dZ1 + WD * W1
        db1 = dZ1.sum(axis=0)
        # global-norm gradient clip for stability across arbitrary curves
        gn = math.sqrt(float((dW1**2).sum() + (db1**2).sum()
                             + (dW2**2).sum() + (db2**2).sum()))
        scale = GRAD_CLIP / gn if gn > GRAD_CLIP else 1.0
        W1 -= LR * scale * dW1
        b1 -= LR * scale * db1
        W2 -= LR * scale * dW2
        b2 -= LR * scale * db2
        if not (np.isfinite(W1).all() and np.isfinite(W2).all()
                and np.isfinite(b1).all() and np.isfinite(b2).all()):
            return None

    Z1 = Xte @ W1 + b1
    A1 = phi(Z1)
    Z2 = A1 @ W2 + b2
    if not np.isfinite(Z2).all():
        return None
    pred = np.argmax(Z2, axis=1)
    return float((pred == yte).mean())


# ============================ candidate answer handling ====================
def _valid_knots(ans):
    """Return a float64 length-K array of activation values, or None if invalid."""
    if isinstance(ans, dict):
        ans = ans.get("activation", None)
    if ans is None:
        return None
    try:
        y = np.asarray(ans, dtype=np.float64)
    except Exception:
        return None
    if y.ndim != 1 or y.shape[0] != K_KNOTS:
        return None
    if not np.all(np.isfinite(y)):
        return None
    if np.max(np.abs(y)) > MAX_ABS_Y:
        return None
    return y


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    # baseline test accuracies: identity curve (phi(x)=x), computed BY the evaluator.
    identity_ys = GRID.copy()

    vec = []
    for inst in instances:
        acc_base = _train_test_acc(inst, identity_ys)
        if acc_base is None:
            # baseline should never diverge, but guard anyway
            vec.append(0.0)
            continue
        denom = max(1.0 - acc_base, MIN_DENOM)

        public = {
            "setting": inst["name"],
            "n_features": inst["d"],
            "n_classes": inst["classes"],
            "hidden": inst["hidden"],
            "grid": GRID.tolist(),
            "n_knots": K_KNOTS,
            "epochs": inst["epochs"],
            "seed": int(20240191 + inst["seed"]),
        }
        ans, st = isorun.run_candidate(cand, public, timeout=CAND_TIMEOUT)
        if st != "OK":
            vec.append(0.0)
            continue
        ys = _valid_knots(ans)
        if ys is None:
            vec.append(0.0)
            continue
        try:
            acc_cand = _train_test_acc(inst, ys)
        except Exception:
            acc_cand = None
        if acc_cand is None:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (acc_cand - acc_base) / denom
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        if 0.0 < r < VALID_FLOOR:
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
