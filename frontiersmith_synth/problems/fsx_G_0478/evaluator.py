#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0478 -- "Design a regularizer for an overfit-prone net."

Task shape (Format B, isolated):
  A candidate DESIGNS a regularizer (a per-feature L2 penalty vector + optional L1 /
  input-jitter / weight-decay knobs). The TRAINER IS FIXED and lives HERE in the parent:
  full-batch gradient descent on a wide random-Fourier-feature (RFF) regression model that
  badly overfits 25 noisy training points when unregularized. The candidate only ever sees
  the PUBLIC view of an instance (training data + the feature frequencies); the held-out
  test set stays hidden in this process. Objective = held-out mean-squared error (MINIMIZE):
  a well-designed regularizer closes the generalization gap.

The candidate is run OS-sandboxed via isorun.run_candidate (untrusted model output).
Deterministic: every instance + the trainer's RNG are seeded; no wall-time / GPU in scoring.
"""
import sys, json, math
import numpy as np
import isorun

N_INSTANCES = 10
RIDGE_CAP   = 20.0     # per-feature L2 weight cap (keeps fixed-lr GD stable)
L1_CAP      = 100.0
JITTER_CAP  = 3.0
WD_CAP      = 20.0


# ---------------------------------------------------------------- instances ---
def _feats(x, omega, phase, scale):
    x = np.asarray(x, dtype=float)
    return scale * np.cos(np.outer(x, omega) + phase)


def make_instances():
    insts = []
    for idx in range(N_INSTANCES):
        rng = np.random.RandomState(4000 + idx)
        M, n_train, T, lr = 200, 25, 8000, 0.03
        sigma = 0.30 + 0.15 * (idx % 4)              # noise level: 0.30 .. 0.75
        band = 7.0 + 1.5 * ((idx // 4) % 2)          # feature bandwidth: 7.0 or 8.5
        omega = rng.randn(M) * band
        phase = rng.rand(M) * 2 * np.pi
        scale = math.sqrt(2.0 / M)
        # smooth teacher: sum of a few low-frequency sinusoids
        ncomp = 3
        tw = rng.randn(ncomp)
        tfreq = rng.uniform(0.5, 2.0, ncomp)
        tph = rng.rand(ncomp) * 2 * np.pi

        def teacher(x):
            return sum(tw[i] * np.sin(tfreq[i] * x + tph[i]) for i in range(ncomp))

        def gen(n):
            x = rng.uniform(-3, 3, n)
            y = teacher(x) + sigma * rng.randn(n)
            return x, y

        xtr, ytr = gen(n_train)
        xte, yte = gen(600)
        pub = {
            "M": M, "T": T, "lr": lr,
            "scale": scale,
            "omega": omega.tolist(),
            "phase": phase.tolist(),
            "xtr": xtr.tolist(),
            "ytr": ytr.tolist(),
            "train_seed": int(5000 + idx),
            "caps": {"ridge": RIDGE_CAP, "l1": L1_CAP,
                     "jitter": JITTER_CAP, "weight_decay": WD_CAP},
        }
        hidden = {"xte": xte.tolist(), "yte": yte.tolist()}
        insts.append({"public": pub, "hidden": hidden})
    return insts


# ---------------------------------------------------------------- trainer -----
def _train(pub, ridge_vec, l1, jitter, wd):
    """FIXED trainer: full-batch GD on RFF ridge objective. Returns weight vector w (M,1)."""
    M, T, lr = pub["M"], pub["T"], pub["lr"]
    omega = np.asarray(pub["omega"], dtype=float)
    phase = np.asarray(pub["phase"], dtype=float)
    scale = float(pub["scale"])
    xtr = np.asarray(pub["xtr"], dtype=float)
    y = np.asarray(pub["ytr"], dtype=float).reshape(-1, 1)
    n = xtr.shape[0]
    Phi = _feats(xtr, omega, phase, scale)
    ridge = np.asarray(ridge_vec, dtype=float).reshape(M, 1)
    w = np.zeros((M, 1))
    rng = np.random.RandomState(pub["train_seed"])
    for _ in range(T):
        if jitter > 0.0:
            P = _feats(xtr + jitter * rng.randn(n), omega, phase, scale)
        else:
            P = Phi
        grad = (2.0 / n) * (P.T @ (P @ w - y)) + 2.0 * ridge * w + l1 * np.sign(w)
        w = w - lr * grad
        if wd > 0.0:
            w = w * (1.0 - lr * wd)
    return w


def _test_mse(pub, hidden, w):
    omega = np.asarray(pub["omega"], dtype=float)
    phase = np.asarray(pub["phase"], dtype=float)
    scale = float(pub["scale"])
    Pte = _feats(hidden["xte"], omega, phase, scale)
    yte = np.asarray(hidden["yte"], dtype=float).reshape(-1, 1)
    return float(np.mean((Pte @ w - yte) ** 2))


def baseline(inst):
    """Trivial construction: zero regularization -> the overfit held-out MSE."""
    pub = inst["public"]
    w = _train(pub, np.zeros(pub["M"]), 0.0, 0.0, 0.0)
    return _test_mse(pub, inst["hidden"], w)


# ---------------------------------------------------------------- scoring -----
def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and v == v and v not in (float("inf"), float("-inf"))


def _parse_answer(pub, answer):
    """Validate + normalize an answer into (ridge_vec[M], l1, jitter, wd) or None."""
    if not isinstance(answer, dict):
        return None
    M = pub["M"]

    # ridge: absent -> zeros; scalar -> uniform; list -> per-feature
    r = answer.get("ridge", 0.0)
    if isinstance(r, (list, tuple)):
        if len(r) != M:
            return None
        ridge = []
        for v in r:
            if not _finite(v) or v < 0.0 or v > RIDGE_CAP:
                return None
            ridge.append(float(v))
        ridge = np.asarray(ridge, dtype=float)
    elif _finite(r):
        if r < 0.0 or r > RIDGE_CAP:
            return None
        ridge = np.full(M, float(r))
    else:
        return None

    def scalar(key, cap):
        v = answer.get(key, 0.0)
        if not _finite(v) or v < 0.0 or v > cap:
            return None, False
        return float(v), True

    l1, ok1 = scalar("l1", L1_CAP)
    jit, ok2 = scalar("jitter", JITTER_CAP)
    wd, ok3 = scalar("weight_decay", WD_CAP)
    if not (ok1 and ok2 and ok3):
        return None
    return ridge, l1, jit, wd


def score(inst, answer):
    pub = inst["public"]
    parsed = _parse_answer(pub, answer)
    if parsed is None:
        return False, None
    ridge, l1, jit, wd = parsed
    w = _train(pub, ridge, l1, jit, wd)
    if not np.all(np.isfinite(w)):
        return False, None
    obj = _test_mse(pub, inst["hidden"], w)
    if not (obj == obj and obj not in (float("inf"), float("-inf")) and obj >= 0.0):
        return False, None
    return True, obj


# ---------------------------------------------------------------- main --------
def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, None
        if not ok:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))     # minimize held-out MSE
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
