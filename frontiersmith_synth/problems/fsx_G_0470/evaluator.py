import sys, json, math, isorun
import numpy as np

# ------------------------------------------------------------------ #
# fsx_G_0470  --  Face-verification metric learning (Format B, isolated)
#
# The candidate receives a PUBLIC instance: a small labelled training gallery of
# raw face "features" (feature vectors + identity labels) plus an UNLABELLED test
# gallery.  It must return a single linear transform W (an embedding matrix) that
# maps a raw d-dim feature to a k-dim embedding.  The evaluator applies W to the
# test features, forms cosine similarities over a HELD-OUT set of face pairs whose
# same/different-identity labels are HIDDEN, and reports verification ROC-AUC.
#
# The identity signal lives in a low-dimensional subspace that a random rotation
# smears across every coordinate and buries under high-variance nuisance ("pose /
# lighting"): per-coordinate scaling is useless, so real metric learning (a
# full-covariance / Fisher transform learned from the training labels) is required.
# An inherent per-sample identity jitter caps the achievable AUC well below 1.
# ------------------------------------------------------------------ #

FDIM = 40          # raw feature dimension
LO, HI = 0.52, 0.90   # AUC -> score affine calibration (identity transform ~ 0.1)


def _rot(rng, d):
    Q, _ = np.linalg.qr(rng.randn(d, d))
    return Q


def _make_group(rng, Q, n_id, per_id, start, cfg):
    d = FDIM; r = cfg["r_sig"]
    feats = []; labs = []
    for k in range(n_id):
        id_sig = rng.randn(r) * cfg["sig"]
        for _ in range(per_id):
            z = np.zeros(d)
            z[:r] = id_sig + rng.randn(r) * cfg["idjit"]      # identity + per-shot jitter
            z[r:] = rng.randn(d - r) * cfg["nuis"]            # nuisance (pose/lighting)
            z = z + rng.randn(d) * cfg["noise"]
            feats.append(Q @ z); labs.append(start + k)
    return np.array(feats), np.array(labs)


def _pairs(yte, seed, n=400):
    import random
    rng = random.Random(seed)
    idx = {}
    for i, l in enumerate(yte):
        idx.setdefault(int(l), []).append(i)
    ids = list(idx)
    pos = []; neg = []
    while len(pos) < n:
        l = rng.choice(ids)
        if len(idx[l]) < 2:
            continue
        i, j = rng.sample(idx[l], 2); pos.append([i, j])
    while len(neg) < n:
        a, b = rng.sample(ids, 2)
        neg.append([rng.choice(idx[a]), rng.choice(idx[b])])
    return pos, neg


def make_instances():
    base = dict(r_sig=6, sig=1.0, idjit=0.65, nuis=2.2, noise=0.5,
                ntr=18, ptr=6, nte=24, pte=6)
    insts = []
    for s in range(10):
        cfg = dict(base)
        if s >= 8:                      # harder held-out instances (generalization)
            cfg["nuis"] = 2.7; cfg["noise"] = 0.7; cfg["idjit"] = 0.8
        rng = np.random.RandomState(1000 + s)
        Q = _rot(rng, FDIM)
        Xtr, ytr = _make_group(rng, Q, cfg["ntr"], cfg["ptr"], 0, cfg)
        Xte, yte = _make_group(rng, Q, cfg["nte"], cfg["pte"], 1000, cfg)
        pos, neg = _pairs(yte, 5000 + s)
        insts.append({
            "public": {
                "d": FDIM,
                "max_out_dim": FDIM,
                "X_train": Xtr.round(6).tolist(),
                "y_train": [int(v) for v in ytr],
                "X_test": Xte.round(6).tolist(),
            },
            "hidden": {"pos": pos, "neg": neg},
        })
    return insts


def _auc(sim_pos, sim_neg):
    sp = np.asarray(sim_pos, float); sn = np.asarray(sim_neg, float)
    allv = np.concatenate([sp, sn])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    r_pos = ranks[:len(sp)].sum()
    return (r_pos - len(sp) * (len(sp) + 1) / 2.0) / (len(sp) * len(sn))


def baseline(inst):
    """AUC of the raw features (identity transform) -- the trivial construction."""
    Xte = np.asarray(inst["public"]["X_test"], float)
    return _auc_from_Z(Xte, inst["hidden"])


def _auc_from_Z(Z, hidden):
    Z = np.asarray(Z, float)
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    Zn = Z / (nrm + 1e-12)
    pos = hidden["pos"]; neg = hidden["neg"]
    sp = [float(Zn[i] @ Zn[j]) for i, j in pos]
    sn = [float(Zn[i] @ Zn[j]) for i, j in neg]
    return _auc(sp, sn)


def score(inst, ans):
    pub = inst["public"]; d = pub["d"]; kmax = pub["max_out_dim"]
    if not isinstance(ans, dict) or "W" not in ans:
        return False, 0.0
    W = ans["W"]
    if not isinstance(W, list) or not (1 <= len(W) <= kmax):
        return False, 0.0
    for row in W:
        if not isinstance(row, list) or len(row) != d:
            return False, 0.0
    try:
        Wm = np.asarray(W, float)
    except Exception:
        return False, 0.0
    if Wm.shape != (len(W), d) or not np.all(np.isfinite(Wm)):
        return False, 0.0
    Xte = np.asarray(pub["X_test"], float)
    Z = Xte @ Wm.T
    if not np.all(np.isfinite(Z)):
        return False, 0.0
    auc = _auc_from_Z(Z, inst["hidden"])
    if not (auc == auc and math.isfinite(auc)):
        return False, 0.0
    return True, float(auc)


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, auc = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0); continue
        r = (auc - LO) / (HI - LO)
        r = 0.0 if r < 0 else (1.0 if r > 1 else r)
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
