"""
fsx_C_0197 -- APIARY FORAGING RESPONSE (modular-component-cross-setting, Format B).

Theme: a beekeeper is tuning the *foraging response curve* r = f(s) that every worker
bee in a colony applies to a scalar "stimulus" s inside a tiny colony-controller network
(a 2-layer MLP whose hidden units all share the drop-in curve f). The SAME curve is stamped
into several independent apiaries (datasets/"meadows") plus one HIDDEN meadow the beekeeper
never inspects; colony fitness = held-out foraging accuracy, aggregated by GEOMETRIC MEAN so a
curve that collapses even one meadow is punished.

The candidate is an untrusted stdin->stdout program. It sees ONLY the public view of an
instance (the stimulus grid + the public meadow descriptors + the fixed training protocol) and
must output the curve as its values on the grid. The evaluator (this file, the parent process)
builds a piecewise-linear activation from those values, trains each tiny MLP itself, and scores.
The datasets, the hidden meadow, and the reference baseline live only in the parent.

Deterministic: every dataset + init is seeded; BLAS pinned to 1 thread by isorun.
"""
import sys, json, math, random
import numpy as np
import isorun

GRID = np.linspace(-6.0, 6.0, 41)          # stimulus grid the curve is specified on
TRAIN_STEPS = 400
LR = 0.5
# All meadows are deliberately NOT linearly separable (a linear/identity foraging curve collapses
# the colony MLP to a linear map and is stuck near chance), yet each is learnable by a small MLP
# given a good nonlinearity. Moderate label/coordinate noise keeps the ceiling below saturation so
# curve *shape* keeps mattering.
POOL = ["xor", "circles", "spiral", "blobxor", "sine"]


# ---------------------------------------------------------------- datasets (meadows)
def _standardize(Xtr, Xte):
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd

def make_dataset(kind, seed, noise, n_train, n_test):
    rng = np.random.RandomState(seed)
    N = n_train + n_test
    if kind == "xor":
        X = rng.randn(N, 2) * 1.1
        Y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        X = X + rng.randn(N, 2) * noise
    elif kind == "circles":
        r = np.where(rng.rand(N) < 0.5, 1.0, 2.2)
        th = rng.rand(N) * 2 * math.pi
        X = np.stack([r * np.cos(th), r * np.sin(th)], 1) + rng.randn(N, 2) * noise
        Y = (r > 1.5).astype(int)
    elif kind == "spiral":
        half = N // 2
        Y = np.array([0] * half + [1] * (N - half))
        idx = np.arange(N)
        frac = np.where(idx < half, idx / max(half, 1), (idx - half) / max(N - half, 1))
        ang = frac * 1.6 * math.pi + Y * math.pi
        rad = 0.5 + frac * 2.2
        X = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1) + rng.randn(N, 2) * noise
    elif kind == "blobxor":
        centers = np.array([[1.6, 1.6], [-1.6, -1.6], [1.6, -1.6], [-1.6, 1.6]])
        lab = np.array([0, 0, 1, 1])
        k = rng.randint(0, 4, size=N)
        X = centers[k] + rng.randn(N, 2) * 0.60
        Y = lab[k]
        X = X + rng.randn(N, 2) * noise
    elif kind == "sine":
        X = np.stack([rng.rand(N) * 8.0 - 4.0, rng.rand(N) * 5.0 - 2.5], 1)
        Y = (X[:, 1] > 2.0 * np.sin(1.3 * X[:, 0])).astype(int)
        X = X + rng.randn(N, 2) * noise
    else:
        raise ValueError(kind)
    perm = rng.permutation(N)
    X, Y = X[perm], Y[perm]
    Xtr, Xte = X[:n_train], X[n_train:]
    Ytr, Yte = Y[:n_train], Y[n_train:]
    Xtr, Xte = _standardize(Xtr, Xte)
    C = int(max(Y.max() + 1, 2))
    return Xtr, Ytr, Xte, Yte, C


# ---------------------------------------------------------------- activation from grid values
def apply_act(ys, X):
    """Piecewise-linear activation defined by ys on GRID; linear extrapolation past the ends.
    Returns (value, derivative) same shape as X."""
    xf = X.ravel()
    idx = np.searchsorted(GRID, xf, side="right") - 1
    idx = np.clip(idx, 0, len(GRID) - 2)
    x0 = GRID[idx]; x1 = GRID[idx + 1]
    y0 = ys[idx]; y1 = ys[idx + 1]
    slope = (y1 - y0) / (x1 - x0)
    val = y0 + slope * (xf - x0)
    return val.reshape(X.shape), slope.reshape(X.shape)


# ---------------------------------------------------------------- tiny MLP, full-batch GD
def train_acc(ys, Xtr, Ytr, Xte, Yte, C, H, seed):
    rng = np.random.RandomState(seed ^ 0x5bd1e995)
    W1 = rng.randn(2, H) * math.sqrt(2.0 / 2)
    b1 = np.zeros(H)
    W2 = rng.randn(H, C) * math.sqrt(1.0 / H)
    b2 = np.zeros(C)
    N = Xtr.shape[0]
    Yoh = np.zeros((N, C)); Yoh[np.arange(N), Ytr] = 1.0
    for _ in range(TRAIN_STEPS):
        pre = Xtr @ W1 + b1
        A, dA = apply_act(ys, pre)
        logits = A @ W2 + b2
        logits -= logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        P = ex / ex.sum(axis=1, keepdims=True)
        dlogits = (P - Yoh) / N
        gW2 = A.T @ dlogits; gb2 = dlogits.sum(axis=0)
        dApost = dlogits @ W2.T
        dpre = dApost * dA
        gW1 = Xtr.T @ dpre; gb1 = dpre.sum(axis=0)
        W1 -= LR * gW1; b1 -= LR * gb1; W2 -= LR * gW2; b2 -= LR * gb2
        if not np.all(np.isfinite(W1)):     # a wild curve can diverge -> that curve just fails
            return 0.0
    pre = Xte @ W1 + b1
    A, _ = apply_act(ys, pre)
    logits = A @ W2 + b2
    pred = logits.argmax(axis=1)
    return float((pred == Yte).mean())


def _gmean_over_settings(ys, settings):
    accs = []
    for st in settings:
        Xtr, Ytr, Xte, Yte, C = make_dataset(st["kind"], st["seed"], st["noise"],
                                              st["n_train"], st["n_test"])
        a = train_acc(ys, Xtr, Ytr, Xte, Yte, C, st["hidden_dim"], st["seed"])
        accs.append(max(a, 1e-3))
    return math.exp(sum(math.log(a) for a in accs) / len(accs))


# ---------------------------------------------------------------- instances
def make_instances():
    out = []
    configs = [
        # (rotation of the pool for the 3 public + 1 hidden meadows, hidden_dim, noise, seedbase)
        (0, 16, 0.22, 100),
        (1, 16, 0.26, 200),
        (2, 20, 0.20, 300),
        (3, 12, 0.30, 400),
        (4, 16, 0.24, 500),
        (2, 20, 0.28, 600),
        (1, 12, 0.24, 700),
    ]
    for rot, H, noise, base in configs:
        kinds = [POOL[(rot + k) % len(POOL)] for k in range(4)]  # 4 meadows
        settings = []
        for j, kind in enumerate(kinds):
            settings.append({"kind": kind, "seed": base + 7 * j, "noise": noise,
                             "hidden_dim": H, "n_train": 220, "n_test": 220})
        public_settings = [{k: s[k] for k in ("kind", "hidden_dim", "noise")}
                           for s in settings[:3]]
        out.append({
            "public": {"grid": [round(float(x), 6) for x in GRID],
                       "train_steps": TRAIN_STEPS, "lr": LR,
                       "public_meadows": public_settings},
            "hidden": {"settings": settings},   # all 4 (incl. the hidden 4th meadow) live here
        })
    return out


def baseline(inst):
    """Reference beekeeper curve: the identity (linear) response f(s)=s. A linear response
    makes the whole colony controller collapse to a linear map -> weak on nonlinear meadows."""
    ys = GRID.copy()
    return _gmean_over_settings(ys, inst["hidden"]["settings"])


def score(inst, ans):
    grid = inst["public"]["grid"]
    if not isinstance(ans, dict) or "ys" not in ans:
        return False, 0.0
    ys = ans["ys"]
    if not isinstance(ys, list) or len(ys) != len(grid):
        return False, 0.0
    try:
        arr = np.array([float(v) for v in ys], dtype=float)
    except (TypeError, ValueError):
        return False, 0.0
    if not np.all(np.isfinite(arr)):
        return False, 0.0
    if float(np.abs(arr).max()) > 1e4:      # reject absurd curves outright
        return False, 0.0
    obj = _gmean_over_settings(arr, inst["hidden"]["settings"])
    if not (obj == obj and obj < math.inf):
        return False, 0.0
    return True, obj


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
            ok, obj = False, 0.0
        if not ok:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * obj / max(b, 1e-12))   # maximization: trivial (==baseline) -> 0.1
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
