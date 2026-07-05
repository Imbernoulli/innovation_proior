#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0476 -- "Drop-in Activation Design: the MLP Battery"
(family: ml-activation-design; format B, quality-metric).

THEME.  You are handed a fixed *battery* of tiny CPU multilayer perceptrons (MLPs)
that must each learn a small 2-D non-linearly-separable classification task
(XOR, concentric circles, spirals, interleaved moons, Gaussian blobs, checkerboard).
Every net has the SAME shape: input -> one hidden layer of 16 units -> softmax, and
is trained with the SAME seeded full-batch gradient descent.  The ONLY thing you get
to choose is the hidden-layer *activation function* phi: R -> R, which is dropped in
identically across the whole battery.  A good activation lets the nets carve curved
decision boundaries; a poor one (e.g. the identity, which collapses the net to a
linear model) cannot.  Objective: MAXIMIZE the geometric-mean test accuracy of the
battery.

HOW phi IS SUPPLIED.  You cannot ship code (the evaluator would have to trust it).
Instead you ship phi as a table of values on a FIXED dense grid of x-coordinates
(GRID, length M, spanning [x_min, x_max]).  The evaluator rebuilds phi as the
piecewise-LINEAR interpolant of your table: for a pre-activation z it uses
phi(z) = interp(z; GRID, y) and phi'(z) = the slope of the enclosing segment
(clamped to the grid ends, with zero derivative outside [x_min, x_max]).  So your
table both defines the forward activation AND, through its segment slopes, the
gradient the nets train with.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "x_min": -8.0, "x_max": 8.0, "n_grid": M,
             "grid": [x_0, ..., x_{M-1}],       # the fixed x-coordinates (ascending)
             "n_nets": K,                        # nets in this battery
             "announced_kinds": [str, ...]}      # dataset kinds of the ANNOUNCED nets
                                                 # (one extra HELD-OUT net of an
                                                 #  undisclosed kind is also scored)
  stdout: ONE JSON object:
            {"y": [phi(x_0), ..., phi(x_{M-1})]} # M finite numbers, |y_i| <= 1e4

  A table is VALID iff "y" is a list of exactly M real, finite numbers of magnitude
  <= 1e4.  Wrong length, a non-number, NaN/Inf, an over-large value, a crash, a
  timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  For each battery (instance) we compute the
geometric-mean test accuracy two ways:
    g_base = gmean accuracy with phi = IDENTITY (y = GRID) -> linear-collapse baseline
    g_cand = gmean accuracy with the candidate's phi
  and normalize with an affine anchor (identity baseline -> 0.1, perfect gmean 1.0 -> 1.0):
    r = clamp( 0.1 + 0.9 * (g_cand - g_base) / (1.0 - g_base), 0, 1 )
  Reproducing the identity scores ~0.1; a genuinely non-linear activation scores
  higher.  Datasets carry ~7% label noise so perfect accuracy is unreachable and even
  the best activations stay strictly below 1.0 on every battery -> headroom.

ISOLATION.  The candidate is untrusted and runs OS-sandboxed in a FRESH SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC instance (grid + metadata).
The datasets, the held-out net, the seeded net initialisations and the identity
reference are computed ONLY in THIS parent, so a frame-walking / filesystem-scraping
candidate learns nothing usable.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all batteries, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"          # force single-thread BLAS -> bit-reproducible

import sys, json, math
import numpy as np
import isorun

# ----------------------------- fixed activation grid -----------------------
X_MIN, X_MAX = -8.0, 8.0
M = 161
GRID = np.linspace(X_MIN, X_MAX, M)
Y_MAG_CAP = 1e4

# ----------------------------- dataset family ------------------------------
def _make_dataset(seed, kind, n=240, noise=0.07):
    """Deterministic seeded synthetic classification task. Returns train/test split."""
    rng = np.random.default_rng(seed)
    if kind == "xor":
        X = rng.uniform(-2, 2, size=(n, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int); C = 2
    elif kind == "circles":
        ang = rng.uniform(0, 2 * np.pi, n); inner = rng.uniform(0, 1, n) < 0.5
        rad = np.where(inner, rng.uniform(0, 0.6, n), rng.uniform(1.0, 1.6, n))
        X = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
        y = (~inner).astype(int); C = 2
    elif kind == "spiral":
        k = n // 2; out = []; lab = []
        for c in range(2):
            t = np.linspace(0, 1, k); rr = t * 4
            th = t * 3.5 * np.pi + c * np.pi + rng.normal(0, 0.15, k)
            out.append(np.stack([rr * np.cos(th), rr * np.sin(th)], 1)); lab += [c] * k
        X = np.concatenate(out); y = np.array(lab); C = 2
    elif kind == "moons":
        k = n // 2; t = np.linspace(0, np.pi, k)
        a = np.stack([np.cos(t), np.sin(t)], 1)
        b = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], 1)
        X = np.concatenate([a, b]) + rng.normal(0, 0.12, (2 * k, 2))
        y = np.array([0] * k + [1] * k); C = 2
    elif kind == "blobs3":
        centers = np.array([[0, 1.5], [-1.3, -1], [1.3, -1]]); k = n // 3
        out = []; lab = []
        for c in range(3):
            out.append(centers[c] + rng.normal(0, 0.55, (k, 2))); lab += [c] * k
        X = np.concatenate(out); y = np.array(lab); C = 3
    elif kind == "checker":
        X = rng.uniform(-2, 2, size=(n, 2))
        y = ((np.floor(X[:, 0]) + np.floor(X[:, 1])) % 2).astype(int); C = 2
    else:
        raise ValueError("unknown kind")
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)              # standardize
    flip = rng.uniform(0, 1, len(y)) < noise            # label noise (Bayes error > 0)
    y = np.where(flip, rng.integers(0, C, len(y)), y)
    idx = rng.permutation(len(y)); ntr = len(y) // 2
    tr, te = idx[:ntr], idx[ntr:]
    return X[tr], y[tr], X[te], y[te], C

# ----------------------------- activation from table -----------------------
def _act_apply(grid_y, z):
    """Piecewise-linear activation + its derivative, evaluated elementwise on z."""
    zc = np.clip(z, X_MIN, X_MAX)
    val = np.interp(zc, GRID, grid_y)
    seg = np.clip(np.searchsorted(GRID, zc) - 1, 0, M - 2)
    slopes = (grid_y[1:] - grid_y[:-1]) / (GRID[1:] - GRID[:-1])
    d = slopes[seg]
    d = np.where((z < X_MIN) | (z > X_MAX), 0.0, d)     # saturate outside grid
    return val, d

# ----------------------------- one MLP train/eval --------------------------
def _train_eval(grid_y, seed, Xtr, ytr, Xte, yte, C, h=16, T=160, lr=0.3):
    rng = np.random.default_rng(seed); d = Xtr.shape[1]
    W1 = rng.normal(0, np.sqrt(2.0 / d), (d, h)); b1 = np.zeros(h)
    W2 = rng.normal(0, np.sqrt(1.0 / h), (h, C)); b2 = np.zeros(C)
    Y = np.eye(C)[ytr]; n = len(ytr)
    for _ in range(T):
        z1 = Xtr @ W1 + b1
        a1, da = _act_apply(grid_y, z1)
        logits = a1 @ W2 + b2
        logits = logits - logits.max(1, keepdims=True)
        ex = np.exp(logits); P = ex / ex.sum(1, keepdims=True)
        dlog = (P - Y) / n
        gW2 = a1.T @ dlog; gb2 = dlog.sum(0)
        dz1 = (dlog @ W2.T) * da
        gW1 = Xtr.T @ dz1; gb1 = dz1.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    z1 = Xte @ W1 + b1
    a1, _ = _act_apply(grid_y, z1)
    logits = a1 @ W2 + b2
    return float((logits.argmax(1) == yte).mean())

def _gmean_battery(grid_y, specs):
    accs = []
    for seed, kind in specs:
        Xtr, ytr, Xte, yte, C = _make_dataset(seed, kind)
        acc = _train_eval(grid_y, seed * 7 + 13, Xtr, ytr, Xte, yte, C)
        accs.append(max(acc, 0.01))
    return float(np.exp(np.mean(np.log(accs))))

# ----------------------------- instance family -----------------------------
def _build_instances():
    """Deterministic list of MLP batteries. Each: name + list of (seed, kind) nets.
    The LAST net in each battery is the HELD-OUT ('hidden') generalisation setting;
    only the earlier nets' kinds are announced to the candidate."""
    specs = [
        ("battery-A", [(1101, "xor"), (1102, "moons"), (1103, "blobs3"), (1104, "circles")]),
        ("battery-B", [(1201, "circles"), (1202, "checker"), (1203, "xor"), (1204, "spiral")]),
        ("battery-C", [(1301, "moons"), (1302, "blobs3"), (1303, "circles"), (1304, "xor")]),
        ("battery-D", [(1401, "spiral"), (1402, "moons"), (1403, "checker"), (1404, "blobs3")]),
        ("battery-E", [(1501, "xor"), (1502, "circles"), (1503, "moons"), (1504, "checker")]),
        ("battery-F", [(1601, "blobs3"), (1602, "spiral"), (1603, "xor"), (1604, "moons")]),
        ("battery-G", [(1701, "checker"), (1702, "xor"), (1703, "circles"), (1704, "blobs3")]),
        ("battery-H", [(1801, "moons"), (1802, "spiral"), (1803, "blobs3"), (1804, "circles")]),
        ("battery-I", [(1901, "circles"), (1902, "xor"), (1903, "checker"), (1904, "moons")]),
    ]
    out = []
    for name, nets in specs:
        out.append({"name": name, "nets": nets})
    return out

def _public_view(inst):
    nets = inst["nets"]
    announced = [k for (_s, k) in nets[:-1]]        # hide the last (held-out) kind
    return {"name": inst["name"], "x_min": X_MIN, "x_max": X_MAX,
            "n_grid": M, "grid": [float(x) for x in GRID],
            "n_nets": len(nets), "announced_kinds": announced}

# ----------------------------- answer validation ---------------------------
def _parse_table(answer):
    """Return a numpy grid_y of length M, or None if the answer is invalid."""
    if not isinstance(answer, dict):
        return None
    y = answer.get("y")
    if not isinstance(y, list) or len(y) != M:
        return None
    out = np.empty(M, dtype=float)
    for i, v in enumerate(y):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        fv = float(v)
        if not math.isfinite(fv) or abs(fv) > Y_MAG_CAP:
            return None
        out[i] = fv
    return out

# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>"); sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        specs = inst["nets"]
        g_base = _gmean_battery(GRID, specs)          # identity reference (parent-only)
        public = _public_view(inst)
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        grid_y = _parse_table(ans)
        if grid_y is None:
            vec.append(0.0); continue
        try:
            g_cand = _gmean_battery(grid_y, specs)
        except Exception:
            vec.append(0.0); continue
        denom = 1.0 - g_base
        if denom < 1e-9:
            denom = 1e-9
        r = 0.1 + 0.9 * (g_cand - g_base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0); continue
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
