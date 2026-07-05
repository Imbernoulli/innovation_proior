"""
fsx_C_0397 -- "Interstellar Relay Fingerprint Sorting"
Format B (isolated heuristic evaluation). Deterministic scoring only.

Task family: classical-ml-algorithm (clustering). The candidate is an untrusted, isolated
stdin->stdout program: it receives ONLY the public view of one relay instance (the point
cloud of signal fingerprints + the known number of source stations k), and must emit a
cluster label for every fingerprint. The evaluator holds the hidden ground-truth station
assignment and recomputes the Adjusted Rand Index (ARI) itself.

Aggregation: GEOMETRIC MEAN of per-instance normalized ARI across a diverse battery of
relay geometries (well-separated blobs, varied-density, anisotropic, high-dim, two-moons,
concentric rings). The gmean punishes a method that overfits one geometry (e.g. plain
k-means, great on blobs but useless on rings) and rewards a transferable clustering design.

Everything (data gen, ARI, weak reference) is pure-numpy + seeded -> reproducible.
"""
import sys, json, math
import numpy as np
import isorun

# --------------------------------------------------------------------------- data gen
def _blobs(rng, n, k, dim, std, box):
    centers = rng.uniform(-box, box, size=(k, dim))
    y = rng.integers(0, k, size=n)
    X = centers[y] + rng.normal(0.0, std, size=(n, dim))
    return X, y

def _varied(rng, n, k, dim, box, stds):
    centers = rng.uniform(-box, box, size=(k, dim))
    y = rng.integers(0, k, size=n)
    scale = np.array([stds[int(c) % len(stds)] for c in y]).reshape(-1, 1)
    X = centers[y] + rng.normal(0.0, 1.0, size=(n, dim)) * scale
    return X, y

def _aniso(rng, n, k, dim, std, box):
    X, y = _blobs(rng, n, k, dim, std, box)
    T = np.array([[0.6, -0.63], [-0.41, 0.82]])
    X = X @ T
    return X, y

def _moons(rng, n, noise):
    n1 = n // 2
    n2 = n - n1
    t1 = np.linspace(0.0, math.pi, n1)
    out = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    t2 = np.linspace(0.0, math.pi, n2)
    inn = np.stack([1.0 - np.cos(t2), 1.0 - np.sin(t2) - 0.5], axis=1)
    X = np.vstack([out, inn]) + rng.normal(0.0, noise, size=(n, 2))
    y = np.array([0] * n1 + [1] * n2)
    return X, y

def _circles(rng, n, noise, factor):
    n1 = n // 2
    n2 = n - n1
    t1 = np.linspace(0.0, 2 * math.pi, n1, endpoint=False)
    outer = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    t2 = np.linspace(0.0, 2 * math.pi, n2, endpoint=False)
    inner = factor * np.stack([np.cos(t2), np.sin(t2)], axis=1)
    X = np.vstack([outer, inner]) + rng.normal(0.0, noise, size=(n, 2))
    y = np.array([0] * n1 + [1] * n2)
    return X, y

# fixed battery of relay geometries (deterministic)
SPECS = [
    {"type": "blobs",   "seed": 1101, "n": 220, "k": 3, "dim": 2, "std": 0.80, "box": 6.0},
    {"type": "blobs",   "seed": 1202, "n": 260, "k": 4, "dim": 2, "std": 1.15, "box": 7.0},
    {"type": "blobs",   "seed": 1303, "n": 240, "k": 3, "dim": 5, "std": 1.35, "box": 6.0},
    {"type": "varied",  "seed": 1404, "n": 260, "k": 3, "dim": 2, "box": 7.0, "stds": [0.5, 1.6, 0.95]},
    {"type": "aniso",   "seed": 1505, "n": 260, "k": 3, "dim": 2, "std": 0.90, "box": 6.0},
    {"type": "moons",   "seed": 1606, "n": 240, "k": 2, "noise": 0.09},
    {"type": "circles", "seed": 1707, "n": 240, "k": 2, "noise": 0.06, "factor": 0.5},
    {"type": "blobs",   "seed": 1808, "n": 300, "k": 5, "dim": 2, "std": 1.45, "box": 7.0},
]

def _generate(spec):
    rng = np.random.default_rng(spec["seed"])
    t = spec["type"]
    if t == "blobs":
        X, y = _blobs(rng, spec["n"], spec["k"], spec["dim"], spec["std"], spec["box"])
    elif t == "varied":
        X, y = _varied(rng, spec["n"], spec["k"], spec["dim"], spec["box"], spec["stds"])
    elif t == "aniso":
        X, y = _aniso(rng, spec["n"], spec["k"], spec["dim"], spec["std"], spec["box"])
    elif t == "moons":
        X, y = _moons(rng, spec["n"], spec["noise"])
    elif t == "circles":
        X, y = _circles(rng, spec["n"], spec["noise"], spec["factor"])
    else:
        raise ValueError(t)
    return np.round(X, 6), y.astype(int)

def make_instances():
    out = []
    for spec in SPECS:
        X, y = _generate(spec)
        pub = {"points": X.tolist(), "k": int(spec["k"]),
               "n": int(X.shape[0]), "dim": int(X.shape[1])}
        out.append({"public": pub, "hidden": {"labels": y.tolist()}})
    return out

# --------------------------------------------------------------------------- metric
def _ari(a, b):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    n = a.shape[0]
    if n < 2:
        return 0.0
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    na = int(ai.max()) + 1
    nb = int(bi.max()) + 1
    cont = np.zeros((na, nb), dtype=np.int64)
    np.add.at(cont, (ai, bi), 1)
    comb2 = lambda x: x * (x - 1) // 2
    sum_c = int(comb2(cont).sum())
    sa = int(comb2(cont.sum(axis=1)).sum())
    sb = int(comb2(cont.sum(axis=0)).sum())
    tot = comb2(np.int64(n))
    if tot == 0:
        return 0.0
    expected = sa * sb / tot
    max_index = 0.5 * (sa + sb)
    if abs(max_index - expected) < 1e-12:
        return 1.0
    return float((sum_c - expected) / (max_index - expected))

def baseline(inst):
    """Weak deterministic reference: median split on the max-variance coordinate."""
    X = np.asarray(inst["public"]["points"], dtype=float)
    y = np.asarray(inst["hidden"]["labels"], dtype=int)
    j = int(np.argmax(X.var(axis=0)))
    lab = (X[:, j] > np.median(X[:, j])).astype(int)
    return _ari(lab, y)

def score(inst, ans):
    """Validate the candidate answer strictly; return (ok, raw_ARI)."""
    pub = inst["public"]
    n = pub["n"]
    if not isinstance(ans, dict) or "labels" not in ans:
        return False, 0.0
    lab = ans["labels"]
    if not isinstance(lab, list) or len(lab) != n:
        return False, 0.0
    clean = []
    for v in lab:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False, 0.0
        if isinstance(v, float):
            if v != v or v in (float("inf"), float("-inf")) or v != int(v):
                return False, 0.0
            v = int(v)
        clean.append(int(v))
    y = np.asarray(inst["hidden"]["labels"], dtype=int)
    raw = _ari(np.asarray(clean, dtype=int), y)
    if raw != raw or raw in (float("inf"), float("-inf")):
        return False, 0.0
    return True, raw

def _normalize(raw, weak):
    """Anchor: weak reference -> ~0.1, perfect (ARI=1) -> ~0.95, headroom for strong."""
    base = min(max(weak, 0.0), 0.45)
    denom = max(1.0 - base, 0.25)
    t = (raw - base) / denom
    v = 0.1 + 0.85 * t
    if v != v:
        return 0.0
    return float(min(1.0, max(0.0, v)))

# --------------------------------------------------------------------------- main
def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, raw = score(inst, ans)
        except Exception:
            ok, raw = False, 0.0
        if not ok:
            vec.append(0.0)
            continue
        weak = baseline(inst)
        vec.append(_normalize(raw, weak))
    logs = [math.log(min(1.0, max(1e-12, x))) for x in vec]
    ratio = math.exp(sum(logs) / len(logs))     # geometric mean
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))

if __name__ == "__main__":
    main()
