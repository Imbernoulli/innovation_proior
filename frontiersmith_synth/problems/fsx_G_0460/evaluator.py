#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0460 -- "Tiny Sprite Classifier: Design the Augmentation Policy"
(family: ml-data-augmentation; format B, quality-metric).

THEME.  A hobbyist trains a TINY image classifier on a CPU.  The dataset is a set of
12x12 grayscale "sprites" belonging to K=4 shape classes (vertical bar, horizontal bar,
diagonal, anti-diagonal).  Every sprite is a class prototype that has been jittered by an
unknown-to-the-solver amount of integer TRANSLATION, additive Gaussian NOISE, and a global
BRIGHTNESS offset.  Only a handful of labelled sprites per class are available for training;
a much larger, differently-jittered test set is held out.

The classifier itself is FIXED (a 1-nearest-neighbour matcher over the training gallery).
The one knob the solver controls is the DATA-AUGMENTATION POLICY: a small list of
label-preserving transforms (with magnitudes and copy counts) that the trainer applies to
each training sprite to enlarge the gallery before matching.  A good policy manufactures
gallery variants that resemble the test-time jitter (small shifts / noise / brightness),
so a shifted test sprite finds a matching shifted gallery sprite -> higher held-out accuracy.
A careless policy that injects transforms the test distribution never contains (mirror flips,
heavy contrast, cutout) plants confusers that DROP accuracy below no-augmentation.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
      {"name": str, "H": 12, "W": 12, "K": 4,
       "n_train": N,
       "train_images": [[float x 144], ...],   # row-major flattened 12x12 sprites
       "train_labels": [int in 0..K-1, ...],
       "vocab": { <op-type>: <param schema>, ... },   # allowed transforms + ranges
       "limits": {"max_ops": 6, "max_copies_per_op": 8, "total_copies": 10}}
  stdout: ONE JSON object:
      {"ops": [ {"type": "shift",    "mag": int 0..4,          "copies": int 1..8},
                {"type": "noise",    "std": float 0..0.5,      "copies": int 1..8},
                {"type": "bright",   "delta": float 0..0.5,    "copies": int 1..8},
                {"type": "contrast", "factor": float 0..1.0,   "copies": int 1..8},
                {"type": "flip",     "axis": "h"|"v",          "copies": int 1..8},
                {"type": "cutout",   "size": int 1..6,         "copies": int 1..8} ] }
      An EMPTY ops list ({"ops": []}) is legal and means "no augmentation".

  A policy is VALID iff: it is an object with an "ops" list of at most max_ops entries; each
  entry is an object with a known "type", the required finite in-range parameter, and an
  integer "copies" in [1, max_copies_per_op]; and the SUM of all copies is <= total_copies.
  Any violation (bad shape, unknown type, missing/out-of-range/non-finite parameter, copies
  out of range, over-budget), a crash, a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance the evaluator computes three held-out
1-NN accuracies on the SAME hidden test set:
    a_base   = accuracy with NO augmentation (weak reference)
    a_oracle = accuracy augmenting with the TRUE jitter parameters at a copy budget FAR above
               the solver's cap (generally-unreachable strong reference)
    a_cand   = accuracy augmenting with the candidate's policy
  and normalizes (weak baseline -> 0.1, oracle -> 1.0):
    r = clamp( 0.1 + 0.9 * (a_cand - a_base) / max(1e-9, a_oracle - a_base), 0, 1 )
  An empty policy scores exactly 0.1; a policy worse than no-augmentation scores < 0.1; the
  oracle's huge copy budget (never granted to the solver) keeps even excellent policies below
  1.0 -> headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH OS-SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The hidden test set, the true
jitter parameters, and all references (a_base, a_oracle) live only in THIS parent process, so
a frame-walking / introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import numpy as np
import isorun

H = 12
W = 12
K = 4
N_TRAIN_PER = 4          # labelled sprites per class (small on purpose)
N_TEST_PER = 30          # held-out sprites per class
MAX_OPS = 6
MAX_COPIES_PER_OP = 8
TOTAL_COPIES = 10        # solver copy budget (oracle is NOT bound by this)

VOCAB = {
    "shift":    {"mag":    "int in [0,4]"},
    "noise":    {"std":    "float in [0.0,0.5]"},
    "bright":   {"delta":  "float in [0.0,0.5]"},
    "contrast": {"factor": "float in [0.0,1.0]"},
    "flip":     {"axis":   "'h' or 'v'"},
    "cutout":   {"size":   "int in [1,6]"},
}


# --------------------------- deterministic data ----------------------------
def _prototypes():
    P = np.zeros((K, H, W), dtype=np.float64)
    lo, hi = 3, 9
    P[0][lo:hi, 5:7] = 1.0                       # vertical bar
    P[1][5:7, lo:hi] = 1.0                       # horizontal bar
    for i in range(lo, hi):                      # diagonal
        P[2][i, i] = 1.0
        P[2][i, i - 1] = 1.0
    for i in range(lo, hi):                      # anti-diagonal
        j = (lo + hi - 1) - i
        P[3][i, j] = 1.0
        P[3][i, max(0, j - 1)] = 1.0
    return P


def _shift(img, dx, dy):
    out = np.zeros_like(img)
    r0 = max(0, dy); r1 = min(H, H + dy)
    c0 = max(0, dx); c1 = min(W, W + dx)
    sr0 = max(0, -dy); sc0 = max(0, -dx)
    out[r0:r1, c0:c1] = img[sr0:sr0 + (r1 - r0), sc0:sc0 + (c1 - c0)]
    return out


def _gen_sample(rng, proto, S, sigma, beta):
    dx = int(rng.integers(-S, S + 1))
    dy = int(rng.integers(-S, S + 1))
    img = _shift(proto, dx, dy)
    img = img + rng.normal(0.0, sigma, img.shape)
    img = img + rng.uniform(-beta, beta)
    return img


def _make_dataset(seed, S, sigma, beta):
    P = _prototypes()
    rng = np.random.default_rng(seed)
    Xtr = []; ytr = []; Xte = []; yte = []
    for k in range(K):
        for _ in range(N_TRAIN_PER):
            Xtr.append(_gen_sample(rng, P[k], S, sigma, beta).ravel()); ytr.append(k)
    for k in range(K):
        for _ in range(N_TEST_PER):
            Xte.append(_gen_sample(rng, P[k], S, sigma, beta).ravel()); yte.append(k)
    return (np.array(Xtr), np.array(ytr, dtype=int),
            np.array(Xte), np.array(yte, dtype=int))


# --------------------------- fixed classifier ------------------------------
def _knn_acc(Xg, yg, Xte, yte):
    """1-nearest-neighbour (euclidean); ties -> lowest gallery index (argmin)."""
    d = (Xte ** 2).sum(1)[:, None] + (Xg ** 2).sum(1)[None, :] - 2.0 * Xte @ Xg.T
    idx = d.argmin(1)
    return float((yg[idx] == yte).mean())


# --------------------------- augmentation engine ---------------------------
def _augment(Xtr, ytr, ops, seed):
    """Apply a validated op list to each training sprite; return enlarged gallery."""
    Xa = [row for row in Xtr]
    ya = [int(v) for v in ytr]
    for oi, op in enumerate(ops):
        t = op["type"]; copies = int(op["copies"])
        for i in range(len(Xtr)):
            img0 = Xtr[i].reshape(H, W)
            for c in range(copies):
                rng = np.random.default_rng((seed * 1000003 + oi * 97 + i * 7 + c * 13) & 0xFFFFFFFF)
                img = img0.copy()
                if t == "shift":
                    m = int(op["mag"])
                    dx = int(rng.integers(-m, m + 1)); dy = int(rng.integers(-m, m + 1))
                    img = _shift(img, dx, dy)
                elif t == "noise":
                    img = img + rng.normal(0.0, float(op["std"]), img.shape)
                elif t == "bright":
                    d = float(op["delta"]); img = img + rng.uniform(-d, d)
                elif t == "contrast":
                    f = float(op["factor"]); mn = img.mean(); img = (img - mn) * (1.0 + rng.uniform(-f, f)) + mn
                elif t == "flip":
                    img = img[:, ::-1].copy() if op["axis"] == "h" else img[::-1, :].copy()
                elif t == "cutout":
                    k = int(op["size"])
                    r0 = int(rng.integers(0, H - k + 1)); c0 = int(rng.integers(0, W - k + 1))
                    img[r0:r0 + k, c0:c0 + k] = 0.0
                Xa.append(img.ravel()); ya.append(int(ytr[i]))
    return np.array(Xa), np.array(ya, dtype=int)


# --------------------------- policy validation -----------------------------
def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _validate(answer):
    """Return a normalized ops list, or None if the policy is invalid."""
    if not isinstance(answer, dict):
        return None
    ops = answer.get("ops")
    if not isinstance(ops, list):
        return None
    if len(ops) > MAX_OPS:
        return None
    total = 0
    clean = []
    for op in ops:
        if not isinstance(op, dict):
            return None
        t = op.get("type")
        if t not in VOCAB:
            return None
        copies = op.get("copies")
        if not _is_int(copies) or copies < 1 or copies > MAX_COPIES_PER_OP:
            return None
        total += copies
        c = {"type": t, "copies": int(copies)}
        if t == "shift":
            v = op.get("mag")
            if not _is_int(v) or v < 0 or v > 4:
                return None
            c["mag"] = int(v)
        elif t == "noise":
            v = op.get("std")
            if not _is_num(v) or v < 0.0 or v > 0.5:
                return None
            c["std"] = float(v)
        elif t == "bright":
            v = op.get("delta")
            if not _is_num(v) or v < 0.0 or v > 0.5:
                return None
            c["delta"] = float(v)
        elif t == "contrast":
            v = op.get("factor")
            if not _is_num(v) or v < 0.0 or v > 1.0:
                return None
            c["factor"] = float(v)
        elif t == "flip":
            v = op.get("axis")
            if v not in ("h", "v"):
                return None
            c["axis"] = v
        elif t == "cutout":
            v = op.get("size")
            if not _is_int(v) or v < 1 or v > 6:
                return None
            c["size"] = int(v)
        clean.append(c)
    if total > TOTAL_COPIES:
        return None
    return clean


# --------------------------- instance family -------------------------------
def _build_instances():
    """Deterministic instance family: (seed, S, sigma, beta)."""
    specs = [
        (101, 2, 0.15, 0.10),
        (102, 2, 0.20, 0.10),
        (103, 3, 0.15, 0.15),
        (104, 2, 0.25, 0.10),
        (105, 3, 0.20, 0.10),
        (106, 2, 0.15, 0.20),
        # harder / larger-jitter held-out instances
        (311, 3, 0.20, 0.15),
        (312, 3, 0.25, 0.10),
        (401, 3, 0.25, 0.15),
        (402, 2, 0.20, 0.20),
    ]
    out = []
    for seed, S, sigma, beta in specs:
        Xtr, ytr, Xte, yte = _make_dataset(seed, S, sigma, beta)
        out.append({"name": f"sprite{seed}", "seed": seed, "S": S,
                    "sigma": sigma, "beta": beta,
                    "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte})
    return out


def _public(inst):
    return {"name": inst["name"], "H": H, "W": W, "K": K,
            "n_train": int(inst["Xtr"].shape[0]),
            "train_images": [[round(float(v), 6) for v in row] for row in inst["Xtr"]],
            "train_labels": [int(v) for v in inst["ytr"]],
            "vocab": VOCAB,
            "limits": {"max_ops": MAX_OPS, "max_copies_per_op": MAX_COPIES_PER_OP,
                       "total_copies": TOTAL_COPIES}}


# --------------------------- scoring driver --------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        Xtr, ytr, Xte, yte = inst["Xtr"], inst["ytr"], inst["Xte"], inst["yte"]
        seed, S, sigma, beta = inst["seed"], inst["S"], inst["sigma"], inst["beta"]

        a_base = _knn_acc(Xtr, ytr, Xte, yte)
        oracle_ops = [{"type": "shift", "mag": S, "copies": 12},
                      {"type": "noise", "std": sigma, "copies": 4},
                      {"type": "bright", "delta": beta, "copies": 3}]
        Xo, yo = _augment(Xtr, ytr, oracle_ops, seed)
        a_oracle = _knn_acc(Xo, yo, Xte, yte)
        denom = a_oracle - a_base
        if denom < 1e-9:
            denom = 1e-9

        ans, st = isorun.run_candidate(cand, _public(inst), timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ops = _validate(ans)
        except Exception:
            ops = None
        if ops is None:
            vec.append(0.0); continue
        try:
            Xa, ya = _augment(Xtr, ytr, ops, seed)
            a_cand = _knn_acc(Xa, ya, Xte, yte)
        except Exception:
            vec.append(0.0); continue

        r = 0.1 + 0.9 * (a_cand - a_base) / denom
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
