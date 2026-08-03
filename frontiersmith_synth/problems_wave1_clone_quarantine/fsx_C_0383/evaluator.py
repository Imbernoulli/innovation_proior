#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0383 -- "Rooftop Gardens: A Drop-in Learning-Rate Schedule
Across Plots" (family: modular-component-cross-setting; eval_form: quality-metric;
MLS-Bench dl-* CPU-proxy shape).

A city block of rooftop gardens each runs a tiny on-site classifier that reads a handful
of microclimate channels (soil moisture, canopy temperature, PAR light, wind, substrate
pH, ...) and predicts the local growth regime (thriving / stressed).  Every rooftop ships
with the SAME tiny model shape, the SAME fixed ReLU MLP, the SAME fixed weight init, the
SAME fixed base learning rate, and the SAME fixed number of full-batch gradient-descent
epochs.  The ONE module you get to design is the per-epoch LEARNING-RATE SCHEDULE -- a
list of multipliers m_0..m_{E-1} applied on top of the fixed base LR -- which is then
flashed IDENTICALLY to every rooftop, including plots you never see.

So you are not tuning an optimizer to one dataset; you are inventing a single transferable
schedule that has to work everywhere.  The block's fixed base LR is deliberately HOT: a
flat schedule (all ones) over-shoots on the noisy near-linear plots yet a naively low flat
LR under-fits the hard nonlinear plots within the epoch budget.  No single constant LR
does well on every plot; a well-SHAPED schedule (e.g. brief warm-up + anneal) can.

The candidate designs ONE schedule, expressed as its E per-epoch multipliers.  It runs as
an ISOLATED subprocess (isorun): it reads ONE JSON "public instance" from stdin (the epoch
count, base LR, and the multiplier clamp -- NEVER any plot's data or labels) and writes
ONE JSON answer (the list of multipliers) to stdout.  The evaluator then, for EACH plot:

  * clamps each multiplier to [0, mult_max] and sets lr_t = base_lr * m_t at epoch t;
  * trains the fixed ReLU MLP with plain full-batch gradient descent (fixed init seed,
    fixed base LR, fixed #epochs) using that schedule;
  * measures held-out test accuracy, normalized against the evaluator's own internal
    FLAT-SCHEDULE baseline (the identical run with every m_t = 1).

Public instance JSON (what the candidate reads on stdin):
    {
      "n_epochs":  int,     # E -- the schedule length you must return
      "base_lr":   float,   # the fixed base learning rate (multipliers scale this)
      "mult_max":  float,   # each multiplier is clamped to [0, mult_max]
      "n_settings_hint": int,
      "note":      str,
      "seed":      int      # a per-call seed the candidate MAY use for its own RNG
    }

Answer JSON (what the candidate writes on stdout):
    [float, float, ..., float]   # length E; answer[t] = m_t (the epoch-t LR multiplier)
    (a dict {"schedule": [...]} or {"lr_mult": [...]} is also accepted)

Per-plot normalization is an affine anchor against the flat-schedule baseline accuracy:

    r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / (ceiling - acc_base), 0, 1 )

so a schedule that merely matches the flat baseline maps to ~0.1 and one that reaches the
per-plot accuracy ceiling maps to 1.0.  The final score is a GEOMETRIC MEAN over plots,
so a schedule that helps one plot but collapses another (divergence on the hard plot,
over-shoot on the noisy plot, or no learning at all) is punished hard.  This rewards a
single schedule that GENERALIZES across settings, not per-dataset tuning (the candidate
never sees the data).

Valid-but-weak plots are floored to a small positive value so the geometric mean stays
defined; a call that raises, returns the wrong length, emits non-finite values, or drives
training to non-finite weights scores 0.0 on the affected plot(s).

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <geometric mean of per-plot r, in [0,1]>
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
N_EPOCHS = 120
BASE_LR = 4.0            # deliberately HOT: a flat schedule over-shoots the noisy plots
MULT_MAX = 5.0           # per-epoch multiplier clamp [0, MULT_MAX]
CEILING = 0.95           # per-plot accuracy ceiling for the affine anchor
MIN_DENOM = 0.12         # min headroom above baseline (keeps r < 1 room, guards div)
VALID_FLOOR = 0.02       # floor for VALID-but-weak plots so gmean stays defined
CAND_TIMEOUT = 20
CAND_SEED = 20240383


# ------------------------------ plot datasets -----------------------------
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


def _gen_spiral(seed, n, d, noise):
    """Two-arm spiral -- a hard nonlinear boundary that needs SUSTAINED learning
    rate to converge within the fixed epoch budget (a naively low flat LR under-fits)."""
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
    """Wavy, near-linear boundary under heavy label noise -- a HOT flat LR keeps
    over-shooting the noise; you must ANNEAL down to settle above the baseline."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.3, size=(n, d))
    y = (X[:, 1] > 1.0 * np.sin(2.5 * X[:, 0])).astype(np.int64)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return _split(rng, X, y, n // 4)


def _gen_xor(seed, n, d, noise):
    """Sign-XOR of two channels."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, d))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return _split(rng, X, y, n // 4)


def _gen_rings(seed, n, d, noise):
    """Concentric-ring regimes (radius bands)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, d))
    r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    thr = np.quantile(r, 0.5)
    y = (r > thr).astype(np.int64)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return _split(rng, X, y, n // 4)


def _gen_bands(seed, n, d, noise):
    """Interleaved sinusoidal stripes -- HELD-OUT geometry."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.2, size=(n, d))
    s = np.sin(1.7 * X[:, 0] + 0.9 * X[:, 1])
    y = (s > 0).astype(np.int64)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return _split(rng, X, y, n // 4)


def _build_plots():
    """Each plot: dataset + fixed MLP width + fixed init seed.  The last two are
    HELD-OUT rooftops (unseen geometry / different channel count)."""
    specs = [
        ("spiral", _gen_spiral, dict(seed=3104, n=440, d=4, noise=0.03), dict(H=20, iseed=74)),
        ("sine",   _gen_sine,   dict(seed=3105, n=380, d=5, noise=0.14), dict(H=16, iseed=75)),
        ("xor",    _gen_xor,    dict(seed=3101, n=360, d=4, noise=0.05), dict(H=12, iseed=71)),
        ("rings",  _gen_rings,  dict(seed=3102, n=360, d=4, noise=0.08), dict(H=14, iseed=72)),
        # held-out rooftops:
        ("bands",  _gen_bands,  dict(seed=3106, n=400, d=4, noise=0.06), dict(H=16, iseed=76)),
        ("spiral", _gen_spiral, dict(seed=3107, n=460, d=5, noise=0.04), dict(H=22, iseed=77)),
    ]
    out = []
    for kind, gen, dp, tp in specs:
        Xtr, ytr, Xte, yte = gen(**dp)
        out.append({"name": f"{kind}{dp['seed']}", "Xtr": Xtr, "ytr": ytr,
                    "Xte": Xte, "yte": yte, "cfg": tp})
    return out


# ------------------------------ training machinery ------------------------
def _softmax(logits):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def _train_eval(sched, plot):
    """Train the fixed ReLU MLP with the given per-epoch LR-multiplier schedule;
    return test accuracy in [0,1], or None if training goes non-finite."""
    Xtr, ytr, Xte, yte = plot["Xtr"], plot["ytr"], plot["Xte"], plot["yte"]
    H, iseed = plot["cfg"]["H"], plot["cfg"]["iseed"]
    n, d = Xtr.shape
    C = int(max(ytr.max(), yte.max())) + 1

    rng = np.random.default_rng(iseed)
    W1 = rng.normal(0.0, math.sqrt(2.0 / d), size=(d, H))
    b1 = np.zeros(H)
    W2 = rng.normal(0.0, math.sqrt(2.0 / H), size=(H, C))
    b2 = np.zeros(C)
    Y = np.zeros((n, C)); Y[np.arange(n), ytr] = 1.0

    for t in range(N_EPOCHS):
        lr = BASE_LR * float(sched[t])
        Z1 = Xtr @ W1 + b1
        A1 = np.maximum(0.0, Z1)
        logits = A1 @ W2 + b2
        P = _softmax(logits)
        dlogits = (P - Y) / n
        dW2 = A1.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dA1 = dlogits @ W2.T
        dZ1 = dA1 * (Z1 > 0.0)
        dW1 = Xtr.T @ dZ1
        db1 = dZ1.sum(axis=0)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
        if not np.isfinite(W1).all() or not np.isfinite(W2).all():
            return None

    Zt = np.maximum(0.0, Xte @ W1 + b1) @ W2 + b2
    if not np.isfinite(Zt).all():
        return None
    pred = np.argmax(Zt, axis=1)
    return float((pred == yte).mean())


# ------------------------------ answer handling ---------------------------
def _valid_schedule(ans):
    if isinstance(ans, dict):
        ans = ans.get("schedule", ans.get("lr_mult", ans.get("mult", None)))
    if ans is None:
        return None
    try:
        s = np.asarray(ans, dtype=np.float64)
    except Exception:
        return None
    if s.ndim != 1 or s.shape[0] != N_EPOCHS:
        return None
    if not np.all(np.isfinite(s)):
        return None
    # clamp each multiplier into the advertised range [0, MULT_MAX]
    s = np.clip(s, 0.0, MULT_MAX)
    return s


# ------------------------------ main --------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    plots = _build_plots()

    public = {
        "n_epochs": int(N_EPOCHS),
        "base_lr": float(BASE_LR),
        "mult_max": float(MULT_MAX),
        "n_settings_hint": len(plots),
        "note": ("Return a length-n_epochs list of per-epoch learning-rate MULTIPLIERS. "
                 "At epoch t the fixed ReLU MLP is trained with lr = base_lr * m_t (each "
                 "m_t clamped to [0, mult_max]). The SAME schedule is flashed to every "
                 "rooftop garden and scored by geometric mean of held-out accuracy; you "
                 "never see any plot's data. The base LR is hot: a flat schedule over-"
                 "shoots the noisy plots while a low flat LR under-fits the hard plots."),
        "seed": int(CAND_SEED),
    }
    ans, st = isorun.run_candidate(cand, public, timeout=CAND_TIMEOUT)
    sched = _valid_schedule(ans) if st == "OK" else None

    base_sched = np.ones(N_EPOCHS)   # FLAT schedule -- the internal weak baseline

    vec = []
    for plot in plots:
        acc_base = _train_eval(base_sched, plot)
        if acc_base is None:
            acc_base = 0.5
        ceiling = min(1.0, max(acc_base + MIN_DENOM, CEILING))

        if sched is None:
            vec.append(0.0)
            continue
        acc_cand = _train_eval(sched, plot)
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
