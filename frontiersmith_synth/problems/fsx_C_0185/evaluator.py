#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0185 -- "Glacier Sensor Net: A Drop-in Activation Across Stations"
(family: modular-component-cross-setting; eval_form: quality-metric; MLS-Bench dl-* CPU proxy shape).

A network of autonomous sensor stations is buried across an alpine glacier.  Each
station streams a handful of physical channels (sub-surface temperature, ice
velocity, tilt, meltwater conductivity, seismic/acoustic emission, ...) and a small
on-station MLP must classify the local regime (e.g. quiescent / creep / calving-risk).
Every station has the SAME tiny model shape and the SAME fixed training recipe -- the
ONLY thing you get to design is the hidden-layer ACTIVATION FUNCTION, which is then
flashed identically to every station in the field, including stations you never see.

The candidate designs ONE drop-in activation function, expressed as its values on a
fixed input grid.  It is run as an ISOLATED subprocess (isorun): it reads ONE JSON
"public instance" from stdin (the grid + metadata -- NEVER any station's data or
labels) and writes ONE JSON answer (the activation's value at each grid point) to
stdout.  The evaluator then:

  * treats the returned samples as a piecewise-LINEAR function g (linear
    interpolation between grid knots; flat/clamped outside the grid), whose
    derivative g' is the slope of the containing segment (0 outside the grid) --
    a fully differentiable drop-in activation;
  * plugs g into a fixed 2-layer MLP and trains it with plain full-batch gradient
    descent (fixed init seed, fixed lr, fixed #epochs) on EACH station's dataset;
  * measures held-out test accuracy per station, and normalizes it against the
    evaluator's own IDENTITY (linear) baseline -- the identical training run whose
    hidden activation is g(x)=x, i.e. no nonlinearity at all.

Public instance JSON (what the candidate reads on stdin):
    {
      "grid":  [float, ...],   # K strictly-increasing knot positions x_0 < ... < x_{K-1}
      "n_grid": int,           # K
      "note":  str,            # human hint
      "seed":  int             # a per-call seed the candidate MAY use for its own RNG
    }

Answer JSON (what the candidate writes on stdout):
    [float, float, ..., float] # length K; answer[i] = g(grid[i])
    (a dict {"activation": [...]} or {"g": [...]} is also accepted)

Per-station normalization is an affine anchor against the identity-baseline accuracy:

    r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / (acc_ceiling - acc_base), 0, 1 )

so an activation that merely matches the linear baseline maps to ~0.1 and one that reaches the
per-station accuracy ceiling maps to 1.0.  The final score is a GEOMETRIC MEAN over
the per-station normalized values, so an activation that helps one station but
collapses another (dead units, exploding gradients, degeneracy) is punished hard.
This rewards a single activation that GENERALIZES across settings, not per-dataset
tuning (the candidate never sees the data).

Valid-but-weak stations are floored to a small positive value so the geometric mean
stays defined; a call that raises, returns the wrong shape/length, emits non-finite
values, or drives training to non-finite weights scores 0.0 on the affected
station(s).

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <geometric mean of per-station r, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys, json, math
import numpy as np
import isorun

# ------------------------------ constants ---------------------------------
GRID_LO, GRID_HI, GRID_K = -8.0, 8.0, 81
GRID = np.linspace(GRID_LO, GRID_HI, GRID_K)
VALID_FLOOR = 0.02        # floor for VALID stations so gmean stays defined
MIN_DENOM = 0.06          # min accuracy headroom above baseline (keeps r<1 room)
BIAS_INIT = 0.0           # hidden-bias init
CAND_TIMEOUT = 20
CAND_SEED = 20240185


# ------------------------------ station datasets --------------------------
def _standardize(Xtr, Xte):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-9
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _split(rng, X, y, n_test):
    perm = rng.permutation(X.shape[0])
    X, y = X[perm], y[perm]
    Xte, yte = X[:n_test], y[:n_test]
    Xtr, ytr = X[n_test:], y[n_test:]
    Xtr, Xte = _standardize(Xtr, Xte)
    return Xtr, ytr.astype(np.int64), Xte, yte.astype(np.int64)


def _gen_xor(seed, n, d, noise):
    """Sign-XOR of two channels -> classes need units that respond to NEGATIVE
    pre-activations (ReLU kills half of them -> dead-unit pressure)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, d))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return _split(rng, X, y, n // 4)


def _gen_rings(seed, n, d, noise):
    """Concentric-ring regimes (radius bands) -> smooth, saturating activations
    generalize better under label noise than a hard rectifier."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, d))
    r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    thr = np.quantile(r, 0.5)
    y = (r > thr).astype(np.int64)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return _split(rng, X, y, n // 4)


def _gen_bands(seed, n, d, noise):
    """Three interleaved sinusoidal bands along channel 0 collapsed to 2 classes
    -> a non-monotone / multi-response activation shape helps."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.2, size=(n, d))
    s = np.sin(1.7 * X[:, 0] + 0.9 * X[:, 1])
    y = (s > 0).astype(np.int64)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return _split(rng, X, y, n // 4)


def _gen_spiral(seed, n, d, noise):
    """Two-arm spiral (classic hard nonlinear boundary)."""
    rng = np.random.default_rng(seed)
    m = n // 2
    t = np.sqrt(rng.random(m)) * 3.2
    Xa = np.stack([t * np.cos(t * 2.2), t * np.sin(t * 2.2)], axis=1)
    Xb = np.stack([t * np.cos(t * 2.2 + np.pi), t * np.sin(t * 2.2 + np.pi)], axis=1)
    Xc = np.vstack([Xa, Xb]) + rng.normal(0.0, 0.28, size=(2 * m, 2))
    extra = rng.normal(0.0, 1.0, size=(2 * m, d - 2))
    X = np.hstack([Xc, extra])
    y = np.array([0] * m + [1] * m, dtype=np.int64)
    flip = rng.random(2 * m) < noise
    y[flip] = 1 - y[flip]
    return _split(rng, X, y, (2 * m) // 4)


def _gen_sine(seed, n, d, noise):
    """A high-frequency sinusoidal decision boundary (y above/below a wavy curve).
    A single linear boundary cannot follow the oscillation, so genuine
    nonlinearity is required across the whole input range."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.3, size=(n, d))
    y = (X[:, 1] > 1.0 * np.sin(2.5 * X[:, 0])).astype(np.int64)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return _split(rng, X, y, n // 4)


def _build_stations():
    """Each station: dataset + fixed MLP/train recipe.  The last two are
    'unseen field stations' (held-out structure / larger)."""
    specs = [
        ("xor",    dict(seed=3101, n=360, d=4, noise=0.06), dict(H=12, lr=0.6, epochs=350, iseed=71)),
        ("rings",  dict(seed=3102, n=360, d=4, noise=0.08), dict(H=14, lr=0.5, epochs=350, iseed=72)),
        ("bands",  dict(seed=3103, n=400, d=4, noise=0.06), dict(H=16, lr=0.5, epochs=380, iseed=73)),
        ("spiral", dict(seed=3104, n=420, d=4, noise=0.04), dict(H=18, lr=0.6, epochs=420, iseed=74)),
        ("sine",   dict(seed=3105, n=380, d=5, noise=0.06), dict(H=16, lr=0.5, epochs=380, iseed=75)),
        # unseen field stations (held-out):
        ("xor",    dict(seed=3106, n=480, d=5, noise=0.05), dict(H=14, lr=0.6, epochs=400, iseed=76)),
    ]
    gens = {"xor": _gen_xor, "rings": _gen_rings, "bands": _gen_bands,
            "spiral": _gen_spiral, "sine": _gen_sine}
    out = []
    for kind, dp, tp in specs:
        Xtr, ytr, Xte, yte = gens[kind](**dp)
        out.append({"name": f"{kind}{dp['seed']}", "Xtr": Xtr, "ytr": ytr,
                    "Xte": Xte, "yte": yte, "cfg": tp})
    return out


# ------------------------------ activation machinery ----------------------
def _apply(gv, z):
    """Forward: piecewise-linear interpolation of the sampled activation."""
    return np.interp(z, GRID, gv)   # clamps to endpoints outside [GRID_LO, GRID_HI]


def _deriv(gv, z):
    """Backward: slope of the containing segment (0 outside the grid, matching
    the flat clamp of np.interp)."""
    idx = np.clip(np.searchsorted(GRID, z, side="right") - 1, 0, GRID_K - 2)
    slope = (gv[idx + 1] - gv[idx]) / (GRID[idx + 1] - GRID[idx])
    slope = np.where((z < GRID[0]) | (z > GRID[-1]), 0.0, slope)
    return slope


def _softmax(logits):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def _train_eval(gv, station):
    """Train the fixed 2-layer MLP with activation gv; return test accuracy in
    [0,1], or None if training goes non-finite."""
    Xtr, ytr, Xte, yte = station["Xtr"], station["ytr"], station["Xte"], station["yte"]
    cfg = station["cfg"]
    H, lr, epochs, iseed = cfg["H"], cfg["lr"], cfg["epochs"], cfg["iseed"]
    n, d = Xtr.shape
    C = int(max(ytr.max(), yte.max())) + 1

    rng = np.random.default_rng(iseed)
    W1 = rng.normal(0.0, math.sqrt(2.0 / d), size=(d, H))
    # Deliberately NEGATIVE bias init: a hard rectifier leaves many hidden units
    # in the dead (zero-gradient) region so they never recover, which handicaps
    # ReLU-like activations and rewards ones with a live negative branch.
    b1 = np.full(H, BIAS_INIT)
    W2 = rng.normal(0.0, math.sqrt(2.0 / H), size=(H, C))
    b2 = np.zeros(C)
    Y = np.zeros((n, C)); Y[np.arange(n), ytr] = 1.0

    for _ in range(epochs):
        Z1 = Xtr @ W1 + b1
        A1 = _apply(gv, Z1)
        logits = A1 @ W2 + b2
        P = _softmax(logits)
        dlogits = (P - Y) / n
        dW2 = A1.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dA1 = dlogits @ W2.T
        dZ1 = dA1 * _deriv(gv, Z1)
        dW1 = Xtr.T @ dZ1
        db1 = dZ1.sum(axis=0)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
        if not np.isfinite(W1).all() or not np.isfinite(W2).all():
            return None

    Zt = _apply(gv, Xte @ W1 + b1) @ W2 + b2
    if not np.isfinite(Zt).all():
        return None
    pred = np.argmax(Zt, axis=1)
    return float((pred == yte).mean())


# ------------------------------ answer handling ---------------------------
def _valid_activation(ans):
    if isinstance(ans, dict):
        ans = ans.get("activation", ans.get("g", None))
    if ans is None:
        return None
    try:
        g = np.asarray(ans, dtype=np.float64)
    except Exception:
        return None
    if g.ndim != 1 or g.shape[0] != GRID_K:
        return None
    if not np.all(np.isfinite(g)):
        return None
    if np.max(np.abs(g)) > 1e6:      # keep training numerics sane; still lets bad shapes score low
        return None
    return g


# ------------------------------ main --------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    stations = _build_stations()

    public = {
        "grid": GRID.tolist(),
        "n_grid": int(GRID_K),
        "note": ("Return g(x) at each grid point. g is used as a piecewise-linear "
                 "drop-in hidden activation, trained identically on every glacier "
                 "station. One activation is scored across all stations by geometric "
                 "mean of test accuracy; you never see the station data."),
        "seed": int(CAND_SEED),
    }
    ans, st = isorun.run_candidate(cand, public, timeout=CAND_TIMEOUT)
    gv = _valid_activation(ans) if st == "OK" else None

    base_act = GRID.copy()           # IDENTITY (linear) activation -- the weak internal baseline

    vec = []
    for stn in stations:
        acc_base = _train_eval(base_act, stn)
        if acc_base is None:
            acc_base = 0.5
        ceiling = min(1.0, max(acc_base + MIN_DENOM, 0.95))

        if gv is None:
            vec.append(0.0)
            continue
        acc_cand = _train_eval(gv, stn)
        if acc_cand is None:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (acc_cand - acc_base) / (ceiling - acc_base)
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
