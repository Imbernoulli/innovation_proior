#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0395 -- "Cave Mapping Expedition: A Drop-in Activation
Function Across Survey Sites" (family: modular-component-cross-setting; eval_form:
quality-metric; MLS-Bench dl-* CPU-proxy shape).

An underground survey team drops an identical tiny classifier at every cave site.  Each
site's on-rig model reads a handful of sonar/echo features off the rock face and predicts
the local passage regime (open chamber vs. blocked seam).  Every rig ships with the SAME
tiny model shape, the SAME fixed 1-hidden-layer MLP, the SAME fixed weight init, the SAME
deliberately-HOT fixed base learning rate, and the SAME fixed number of full-batch
gradient-descent epochs.  The ONE module you get to design is the hidden-layer ACTIVATION
FUNCTION -- shipped as a piecewise-linear curve given by its values on a fixed input grid
-- which is then flashed IDENTICALLY to every rig, including sites the team never mapped.

So you are not tuning a network to one cave; you are inventing a single transferable
activation that has to work everywhere.  Every cave here is genuinely nonlinear, so the
naive drop-in -- passing the pre-activation straight through (a LINEAR / identity
activation) -- collapses the net to a linear classifier that under-fits every site.  Any
genuine nonlinearity (ReLU / leaky / ELU / tanh / swish / GELU / softsign shapes) unlocks
the hidden layer and jumps well above the linear baseline -- but under the hot base LR a
hard rectifier over-shoots the noisier caves, a too-flat/saturating curve throws away
signal, and a runaway slope drives training to non-finite weights.  No trivial choice is
best across all sites; a well-SHAPED curve (nonlinear, smoothly gated, ~linear tails) can.

The candidate designs ONE activation, expressed as its values at the fixed grid knots.  It
runs as an ISOLATED subprocess (isorun): it reads ONE JSON "public instance" from stdin
(the grid geometry only -- NEVER any site's data or labels) and writes ONE JSON answer
(the list of activation values at the grid knots) to stdout.  The evaluator builds a
piecewise-linear activation from those knots (linear interpolation inside the grid, linear
extrapolation with the end-segment slopes outside it; the segment slope is used as the
derivative for backprop) and then, for EACH site:

  * trains the fixed MLP with plain full-batch gradient descent (fixed init seed, fixed
    base LR, fixed #epochs) using that activation;
  * measures held-out test accuracy, normalized against the evaluator's own internal
    LINEAR baseline (the identical run whose activation is exactly the identity, a(x)=x).

Public instance JSON (what the candidate reads on stdin):
    {
      "n_grid":   int,     # K -- number of grid knots; your answer has length K
      "x_lo":     float,   # grid spans [x_lo, x_hi] inclusive, evenly (linspace)
      "x_hi":     float,
      "n_settings_hint": int,
      "note":     str,
      "seed":     int      # a per-call seed the candidate MAY use for its own RNG
    }

Answer JSON (what the candidate writes on stdout):
    [float, float, ..., float]   # length K; answer[k] = activation value at grid knot k
    (a dict {"values": [...]} or {"a": [...]} is also accepted)

Per-site normalization is an affine anchor against the LINEAR baseline accuracy:

    r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / (ceiling - acc_base), 0, 1 )

so an activation that merely matches the ReLU baseline maps to ~0.1 and one that reaches
the per-site accuracy ceiling maps to 1.0.  The final score is a GEOMETRIC MEAN over sites,
so an activation that helps one cave but collapses another (kills learning, over-fits noise,
or drives training non-finite) is punished hard.  This rewards a single activation that
GENERALIZES across settings, not per-dataset tuning (the candidate never sees the data).

Valid-but-weak sites are floored to a small positive value so the geometric mean stays
defined; a call that raises, returns the wrong length, emits non-finite values, or drives
training to non-finite weights scores 0.0 on the affected site(s).

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <geometric mean of per-site r, in [0,1]>
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
N_GRID = 41              # grid knots on [X_LO, X_HI]; answer length
X_LO = -8.0
X_HI = 8.0
N_EPOCHS = 200
BASE_LR = 1.5            # full-batch GD step (gradients are averaged over n)
B1_BIAS = 0.0            # hidden-layer bias init
CEILING = 0.95           # per-site accuracy ceiling for the affine anchor
MIN_DENOM = 0.12         # min headroom above baseline (keeps r < 1 room, guards div)
VALID_FLOOR = 0.02       # floor for VALID-but-weak sites so gmean stays defined
CAND_TIMEOUT = 20
CAND_SEED = 20240395

GRID = np.linspace(X_LO, X_HI, N_GRID)
DGRID = np.diff(GRID)    # length K-1, all equal (uniform grid)


# ------------------------------ site datasets -----------------------------
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


def _gen_tunnel(seed, n, d, noise):
    """Twisting two-arm tunnel (spiral): a hard nonlinear boundary that needs real
    hidden-unit capacity -- if too many units are dead the boundary is missed."""
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


def _gen_fault(seed, n, d, noise):
    """Cross-fault (sign-XOR of two echo channels)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, d))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return _split(rng, X, y, n // 4)


def _gen_chambers(seed, n, d, noise):
    """Concentric chambers (radius bands)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, d))
    r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    thr = np.quantile(r, 0.5)
    y = (r > thr).astype(np.int64)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return _split(rng, X, y, n // 4)


def _gen_strata(seed, n, d, noise):
    """Interleaved sinusoidal strata (a periodic-stripe boundary a linear rule can't
    follow)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.2, size=(n, d))
    s = np.sin(1.7 * X[:, 0] + 0.9 * X[:, 1])
    y = (s > 0).astype(np.int64)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return _split(rng, X, y, n // 4)


def _build_sites():
    """Each site: dataset + fixed MLP width + fixed init seed.  Every geometry is
    genuinely nonlinear, so a linear (identity) activation under-fits everywhere.  The
    last two are HELD-OUT (unmapped) caves -- unseen seeds / an extra echo channel."""
    specs = [
        ("tunnel",   _gen_tunnel,   dict(seed=5104, n=440, d=4, noise=0.03), dict(H=20, iseed=74)),
        ("fault",    _gen_fault,    dict(seed=5101, n=360, d=4, noise=0.05), dict(H=12, iseed=71)),
        ("chambers", _gen_chambers, dict(seed=5102, n=360, d=4, noise=0.08), dict(H=14, iseed=72)),
        ("strata",   _gen_strata,   dict(seed=5106, n=400, d=4, noise=0.06), dict(H=16, iseed=76)),
        # held-out / unmapped caves:
        ("crossbeds",_gen_fault,    dict(seed=5111, n=400, d=5, noise=0.05), dict(H=16, iseed=81)),
        ("domes",    _gen_chambers, dict(seed=5112, n=420, d=5, noise=0.06), dict(H=18, iseed=82)),
    ]
    out = []
    for kind, gen, dp, tp in specs:
        Xtr, ytr, Xte, yte = gen(**dp)
        out.append({"name": f"{kind}{dp['seed']}", "Xtr": Xtr, "ytr": ytr,
                    "Xte": Xte, "yte": yte, "cfg": tp})
    return out


# ------------------------------ activation machinery ----------------------
def _pw_eval(vals, Z):
    """Piecewise-linear activation from knot values `vals` on GRID.
    Returns (A, S): A = activation at Z, S = local segment slope (the derivative used
    for backprop).  Inside the grid -> linear interpolation; outside -> linear
    extrapolation with the nearest end-segment slope."""
    slopes = np.diff(vals) / DGRID                 # length K-1
    idx = np.searchsorted(GRID, Z) - 1
    idx = np.clip(idx, 0, N_GRID - 2)
    x0 = GRID[idx]
    v0 = vals[idx]
    s = slopes[idx]
    A = v0 + s * (Z - x0)
    return A, s


def _softmax(logits):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def _train_eval(vals, site):
    """Train the fixed MLP with the given piecewise-linear activation `vals`;
    return test accuracy in [0,1], or None if training goes non-finite."""
    Xtr, ytr, Xte, yte = site["Xtr"], site["ytr"], site["Xte"], site["yte"]
    H, iseed = site["cfg"]["H"], site["cfg"]["iseed"]
    n, d = Xtr.shape
    C = int(max(ytr.max(), yte.max())) + 1

    rng = np.random.default_rng(iseed)
    W1 = rng.normal(0.0, math.sqrt(2.0 / d), size=(d, H))
    b1 = np.full(H, B1_BIAS)                       # negatively-biased -> dead-zone init
    W2 = rng.normal(0.0, math.sqrt(2.0 / H), size=(H, C))
    b2 = np.zeros(C)
    Y = np.zeros((n, C)); Y[np.arange(n), ytr] = 1.0

    for _ in range(N_EPOCHS):
        Z1 = Xtr @ W1 + b1
        A1, S1 = _pw_eval(vals, Z1)
        logits = A1 @ W2 + b2
        P = _softmax(logits)
        dlogits = (P - Y) / n
        dW2 = A1.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dA1 = dlogits @ W2.T
        dZ1 = dA1 * S1
        dW1 = Xtr.T @ dZ1
        db1 = dZ1.sum(axis=0)
        W1 -= BASE_LR * dW1; b1 -= BASE_LR * db1
        W2 -= BASE_LR * dW2; b2 -= BASE_LR * db2
        if not (np.isfinite(W1).all() and np.isfinite(W2).all()):
            return None

    A1t, _ = _pw_eval(vals, Xte @ W1 + b1)
    Zt = A1t @ W2 + b2
    if not np.isfinite(Zt).all():
        return None
    pred = np.argmax(Zt, axis=1)
    return float((pred == yte).mean())


# ------------------------------ answer handling ---------------------------
def _valid_values(ans):
    if isinstance(ans, dict):
        ans = ans.get("values", ans.get("a", ans.get("activation", None)))
    if ans is None:
        return None
    try:
        v = np.asarray(ans, dtype=np.float64)
    except Exception:
        return None
    if v.ndim != 1 or v.shape[0] != N_GRID:
        return None
    if not np.all(np.isfinite(v)):
        return None
    return v


# ------------------------------ main --------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    sites = _build_sites()

    public = {
        "n_grid": int(N_GRID),
        "x_lo": float(X_LO),
        "x_hi": float(X_HI),
        "n_settings_hint": len(sites),
        "note": ("Return a length-n_grid list of activation VALUES at the grid knots "
                 "grid = linspace(x_lo, x_hi, n_grid). The evaluator builds a "
                 "piecewise-linear hidden-layer activation from them (linear interp "
                 "inside, linear extrapolation with the end-segment slope outside; the "
                 "segment slope is the backprop derivative) and flashes the SAME "
                 "activation to every cave site, scoring by geometric mean of held-out "
                 "accuracy. You never see any site's data. Every cave is nonlinear, so a "
                 "linear (identity) activation under-fits and maps to ~0.1; any genuine "
                 "nonlinearity beats it, but the base LR is hot so a hard rectifier over-"
                 "shoots the noisier caves while a too-flat/saturating curve loses signal."),
        "seed": int(CAND_SEED),
    }
    ans, st = isorun.run_candidate(cand, public, timeout=CAND_TIMEOUT)
    vals = _valid_values(ans) if st == "OK" else None

    base_vals = GRID.copy()   # identity (linear) activation -- the internal weak baseline

    vec = []
    for site in sites:
        acc_base = _train_eval(base_vals, site)
        if acc_base is None:
            acc_base = 0.5
        ceiling = min(1.0, max(acc_base + MIN_DENOM, CEILING))

        if vals is None:
            vec.append(0.0)
            continue
        acc_cand = _train_eval(vals, site)
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
