#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0456 -- "Gene-Expression Cell-Type Atlas:
Unsupervised Linear Embedding for Downstream kNN"
(family: ml-dim-reduction-projection; eval_form: quality-metric; MLS-Bench ml-* shape).

A single-cell RNA-seq atlas measures the expression of D genes across many cells.
Each cell belongs to one of C cell types.  The candidate is given ONLY a large
UNLABELLED training matrix (a fresh copy of expression vectors -- no cell-type
labels).  From these data statistics alone it must design an AFFINE (linear)
projection to a small target dimension d:

    z = W @ (x - b)          W: d x D,   b: length D

that is then used, UNCHANGED, in a downstream nearest-neighbour classifier that
this evaluator runs on HELD-OUT data:  a labelled reference set and a held-out
test set (both hidden from the candidate) are projected with the candidate's
(W, b); each test cell is labelled by majority vote over its k nearest reference
cells in the d-dimensional embedding.  Quality = classification accuracy on the
hidden test labels, recomputed deterministically here (no sklearn, no wall-time).

The atlas family is intentionally DIVERSE and mimics real scRNA-seq pathology:
  * clean       -- cell-type signal is the dominant variance (raw PCA already fine);
  * noisy       -- many high-variance TECHNICAL-noise genes drown the signal, so
                   variance-chasing (raw PCA / random projection) is distracted and
                   only standardization + co-expression structure recovers the types;
  * scaled      -- genes live on wildly different expression scales (housekeeping vs
                   markers); raw distance/PCA is dominated by the loud genes;
  * correlated  -- strong co-expression modules with redundancy;
  * heldout     -- more cell types / more genes (generalization stress).
No single off-the-shelf reducer wins them all.  The final score is a GEOMETRIC
MEAN over per-instance normalized scores, so a projection that overfits one atlas
regime and collapses on another is heavily penalized -- rewarding a method that
GENERALIZES.

Per-instance normalization is an affine anchor between a weak internal baseline
(a fixed random projection, seeded) and a strong SUPERVISED oracle (Fisher LDA
fit on the hidden reference labels), so headroom always remains for an
unsupervised candidate:

    r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / max(acc_oracle - acc_base, MIN), 0, 1 )

A candidate that merely reproduces the random baseline maps to ~0.1; the
supervised oracle is (essentially) unreachable without labels, so the best
unsupervised projections stay below 1.0.  A candidate that raises, returns the
wrong shape, or emits a non-finite / non-real projection scores exactly 0.0 on
that instance (dragging the geometric mean to 0).

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <geometric mean of per-instance r, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import os
# Cap BLAS/OMP threads so numpy imports cleanly inside isorun's address-space-capped
# child (OpenBLAS otherwise reserves per-thread buffers that blow the limit).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys, json, math
import numpy as np
import isorun

VALID_FLOOR = 0.02      # floor for VALID instances so the geometric mean stays defined
MIN_DENOM = 0.15        # cap on the (oracle - base) gap so an easy atlas still leaves headroom
KNN_K = 7               # neighbours in the downstream classifier
CAND_TIMEOUT = 20


# ============================ atlas family (instances) =====================
def _gen_atlas(seed, C, D, n_signal, d_target, noise_scale, scale_spread,
               class_sep, within, sig_noise, n_train, n_ref, n_test,
               n_nuis=18, k_nuis=3, nuis_std=1.35):
    """Latent-factor scRNA-seq generator.

    THREE kinds of gene are laid out in blocks:
      * `n_signal` MARKER genes carry C class-linked co-expression modules (one
        factor per cell type); a cell type up-regulates its own factor.
      * `n_nuis` NUISANCE genes carry `k_nuis` correlated co-expression modules
        that vary per-cell but are INDEPENDENT of cell type (biological programs
        like cell-cycle / stress).  They have high variance, so variance-chasing
        reducers (raw/standardized PCA) burn embedding dimensions on them, while a
        supervised discriminant (LDA) ignores them -- this is what leaves headroom.
      * the remaining genes are independent TECHNICAL-noise genes with std
        `noise_scale`.
    Per-gene expression scales span `scale_spread` (log-uniform).  Returns train
    (UNLABELLED, public) plus labelled reference and held-out test splits (hidden)."""
    rng = np.random.default_rng(seed)
    n_total = n_train + n_ref + n_test

    # ---- class-linked signal loadings on the marker-gene block ----
    Ls = np.zeros((D, C))
    for g in range(n_signal):
        Ls[g, g % C] = rng.uniform(0.8, 1.6)
    Fmeans = np.zeros((C, C))
    for c in range(C):
        Fmeans[c, c] = class_sep
        Fmeans[c] += rng.normal(0.0, 0.20, size=C)   # mild cross-talk

    # ---- class-IRRELEVANT nuisance loadings on the nuisance-gene block ----
    Ln = np.zeros((D, k_nuis))
    for j in range(n_nuis):
        g = n_signal + j
        if g < D:
            Ln[g, j % k_nuis] = rng.uniform(0.9, 1.7)

    # per-gene expression scale + per-gene technical-noise std
    scales = np.exp(rng.uniform(np.log(1.0 / scale_spread), np.log(scale_spread), size=D))
    gene_noise = np.full(D, sig_noise)
    gene_noise[n_signal + n_nuis:] = noise_scale     # pure technical-noise genes

    # balanced labels, then shuffle
    y = np.arange(n_total) % C
    rng.shuffle(y)

    fs = Fmeans[y] + rng.normal(0.0, within, size=(n_total, C))      # signal factors
    fn = rng.normal(0.0, nuis_std, size=(n_total, k_nuis))           # nuisance factors
    X = fs @ Ls.T + fn @ Ln.T
    X = X + rng.normal(0.0, 1.0, size=(n_total, D)) * gene_noise[None, :]
    X = X * scales[None, :]                          # apply per-gene scale
    X = X.astype(np.float64)

    perm = rng.permutation(n_total)
    X, y = X[perm], y[perm]
    tr = slice(0, n_train)
    rf = slice(n_train, n_train + n_ref)
    te = slice(n_train + n_ref, n_total)
    return {
        "X_train": X[tr], "X_ref": X[rf], "y_ref": y[rf].astype(np.int64),
        "X_test": X[te], "y_test": y[te].astype(np.int64),
        "d_target": int(d_target), "seed": int(seed),
    }


def _build_instances():
    base = dict(class_sep=2.2, within=0.95, sig_noise=0.35,
                n_train=300, n_ref=260, n_test=200, d_target=6,
                n_nuis=24, k_nuis=4, nuis_std=1.7)
    specs = [
        # clean: low technical noise -> raw PCA competitive (but still chases nuisance)
        ("clean",      dict(seed=4560101, C=5, D=85,  n_signal=25, noise_scale=0.60, scale_spread=1.0)),
        ("clean",      dict(seed=4560102, C=5, D=90,  n_signal=25, noise_scale=0.80, scale_spread=1.0)),
        # noisy: loud independent technical-noise genes -> raw PCA / random distracted
        ("noisy",      dict(seed=4560103, C=5, D=100, n_signal=25, noise_scale=2.4,  scale_spread=1.0)),
        ("noisy",      dict(seed=4560104, C=5, D=110, n_signal=25, noise_scale=2.8,  scale_spread=1.0)),
        # scaled: wildly different gene scales -> standardization essential
        ("scaled",     dict(seed=4560105, C=5, D=95,  n_signal=25, noise_scale=1.1,  scale_spread=12.0)),
        ("scaled",     dict(seed=4560106, C=5, D=95,  n_signal=25, noise_scale=1.3,  scale_spread=16.0)),
        # correlated: bigger marker modules + more nuisance + moderate noise
        ("correlated", dict(seed=4560107, C=5, D=100, n_signal=30, noise_scale=1.6,  scale_spread=1.5, n_nuis=30, k_nuis=5, nuis_std=1.75)),
        ("correlated", dict(seed=4560108, C=5, D=105, n_signal=30, noise_scale=1.9,  scale_spread=2.0, n_nuis=30, k_nuis=5, nuis_std=1.8)),
        # heldout: more cell types + more genes (generalization stress)
        ("heldout",    dict(seed=4560109, C=6, D=115, n_signal=30, noise_scale=2.0,  scale_spread=3.0, n_nuis=30, k_nuis=5)),
        ("heldout",    dict(seed=4560110, C=6, D=125, n_signal=30, noise_scale=2.3,  scale_spread=4.0, n_nuis=30, k_nuis=5)),
    ]
    out = []
    for kind, p in specs:
        q = dict(base); q.update(p)
        inst = _gen_atlas(**q)
        inst["name"] = f"{kind}{p['seed']}"
        out.append(inst)
    return out


# ============================ downstream kNN (deterministic) ===============
def _embed(X, W, b):
    """Affine projection z = (x - b) @ W.T ."""
    return (X - b[None, :]) @ W.T


def _knn_acc(emb_ref, y_ref, emb_test, y_test, k):
    """Majority-vote kNN accuracy.  Deterministic: mergesort neighbour order,
    ties among classes broken by earliest appearance in that sorted order."""
    ref_sq = np.sum(emb_ref ** 2, axis=1)
    correct = 0
    kk = min(k, emb_ref.shape[0])
    for i in range(emb_test.shape[0]):
        d2 = ref_sq - 2.0 * (emb_ref @ emb_test[i]) + float(np.dot(emb_test[i], emb_test[i]))
        order = np.argsort(d2, kind="mergesort")[:kk]
        labs = y_ref[order]
        counts = {}
        best_lab = None
        best_ct = -1
        for lab in labs:                    # first-seen order = nearest-first
            c = counts.get(lab, 0) + 1
            counts[lab] = c
            if c > best_ct:
                best_ct = c
                best_lab = lab
        if best_lab == y_test[i]:
            correct += 1
    return correct / emb_test.shape[0]


# ============================ internal baseline + oracle ===================
def _baseline_proj(inst):
    """Weak reference reducer: a fixed random Gaussian projection (JL style),
    centred on the training mean -- the SAME affine pipeline the candidate uses."""
    D = inst["X_train"].shape[1]
    d = inst["d_target"]
    rng = np.random.default_rng(90000 + inst["seed"])
    W = rng.normal(0.0, 1.0 / math.sqrt(D), size=(d, D))
    b = inst["X_train"].mean(axis=0)
    return W, b


def _oracle_proj(inst):
    """Supervised Fisher LDA fit on the HIDDEN reference labels -> strong upper
    anchor (unreachable without labels).  Symmetric whitening form for
    determinism.  Falls back to standardized-PCA directions if LDA is degenerate."""
    X = inst["X_ref"]
    y = inst["y_ref"]
    D = X.shape[1]
    d = inst["d_target"]
    mu = X.mean(axis=0)
    classes = np.unique(y)
    Sw = np.zeros((D, D))
    Sb = np.zeros((D, D))
    for c in classes:
        Xc = X[y == c]
        mc = Xc.mean(axis=0)
        dc = Xc - mc[None, :]
        Sw += dc.T @ dc
        diff = (mc - mu)[:, None]
        Sb += Xc.shape[0] * (diff @ diff.T)
    reg = 1e-3 * (np.trace(Sw) / D + 1e-9)
    Sw += reg * np.eye(D)
    try:
        L = np.linalg.cholesky(Sw)
        Linv = np.linalg.inv(L)
        A = Linv @ Sb @ Linv.T
        A = 0.5 * (A + A.T)
        w, U = np.linalg.eigh(A)          # ascending
        idx = np.argsort(w)[::-1][:d]
        V = (Linv.T @ U[:, idx]).T        # (d, D)
    except np.linalg.LinAlgError:
        std = X.std(axis=0) + 1e-9
        Z = (X - mu[None, :]) / std[None, :]
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        V = (Vt[:d] / std[None, :])
    return V.astype(np.float64), mu


# ============================ candidate answer handling ====================
def _valid_proj(ans, D, d):
    """Return (W (d,D), b (D,)) or None if the answer is malformed / non-finite."""
    if not isinstance(ans, dict):
        return None
    if "W" not in ans or "b" not in ans:
        return None
    try:
        W = np.asarray(ans["W"], dtype=np.float64)
        b = np.asarray(ans["b"], dtype=np.float64)
    except Exception:
        return None
    if W.ndim != 2 or W.shape != (d, D):
        return None
    if b.ndim != 1 or b.shape[0] != D:
        return None
    if not (np.all(np.isfinite(W)) and np.all(np.isfinite(b))):
        return None
    return W, b


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        D = inst["X_train"].shape[1]
        d = inst["d_target"]

        Wb, bb = _baseline_proj(inst)
        acc_base = _knn_acc(_embed(inst["X_ref"], Wb, bb), inst["y_ref"],
                            _embed(inst["X_test"], Wb, bb), inst["y_test"], KNN_K)
        Wo, bo = _oracle_proj(inst)
        acc_orac = _knn_acc(_embed(inst["X_ref"], Wo, bo), inst["y_ref"],
                            _embed(inst["X_test"], Wo, bo), inst["y_test"], KNN_K)
        denom = max(acc_orac - acc_base, MIN_DENOM)

        public = {
            "X_train": inst["X_train"].tolist(),
            "n_train": int(inst["X_train"].shape[0]),
            "n_genes": int(D),
            "d_target": int(d),
            "seed": int(inst["seed"]),
        }
        ans, st = isorun.run_candidate(cand, public, timeout=CAND_TIMEOUT)
        if st != "OK":
            vec.append(0.0)
            continue

        proj = _valid_proj(ans, D, d)
        if proj is None:
            vec.append(0.0)
            continue
        W, b = proj
        try:
            acc_cand = _knn_acc(_embed(inst["X_ref"], W, b), inst["y_ref"],
                                _embed(inst["X_test"], W, b), inst["y_test"], KNN_K)
        except Exception:
            vec.append(0.0)
            continue
        if not math.isfinite(acc_cand):
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (acc_cand - acc_base) / denom
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
