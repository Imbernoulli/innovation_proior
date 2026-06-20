**Problem.** Close the three gaps MAGIC left: a single sqrt transform for all genes, no refinement
in the log-normalized scoring space, and no global low-rank structure. The endpoint is an adaptation
of the TTT-Discover denoiser (test-time-training/discover, arXiv:2601.16175) that tops the
OpenProblems leaderboard (0.71 PBMC / 0.73 Tabula vs MAGIC ~0.64).

**Key idea.** Keep MAGIC's adaptive-bandwidth diffusion graph as the backbone, built once on the
Anscombe embedding with a self-loop. Run the full pipeline under **three variance-stabilizing
transforms** — Anscombe (low-dropout genes), Freeman-Tukey (high-dropout genes), square root
(bridge) — and **ensemble gene-wise by dropout fraction**. Inside each pass: impute exact zeros with
a diffusion-weighted neighbor average, apply gene-wise weighted **multi-scale diffusion**
(`X + b·PX + b²·P²X + …` with per-gene decay and a guaranteed baseline), then **adaptively blend**
diffused and raw signal per gene by dropout / variance-reduction / correlation. On top of the
ensemble: a **light truncated-SVD low-rank refinement** (small weight — the rate is not exactly
low-rank), then **two diffusion passes in log-normalized space** (rescale → log1p → diffuse → expm1),
the exact space the MSE metric scores.

**Why these choices.** Different transforms stabilize different dropout regimes, so the gene-wise
ensemble fixes the per-gene mis-stabilization. Zero-imputation distinguishes dropouts from true
zeros; multi-scale diffusion gives each gene the smoothing its dropout warrants; the adaptive blend
is the safety valve against over-smoothing. Low-rank captures global structure local diffusion
misses, but the weight is kept small because the over-dispersed rate punishes aggressive low-rank.
The log-space diffusion is the most direct attack on the MSE term — smoothing in the very space the
metric is computed in. None of it can exceed the true-rate ceiling (irreducible noise), so this is
the top of the ladder.

**Hyperparameters / contract.** `knn=12`, `t=7`, `n_pca=50`, `decay(α)=2.0`, `diff_decay=0.9`,
transforms `(anscombe, ft, sqrt)`, `lowrank_components=30`, `lowrank_weight=0.10`, `log_smooth_t=4`
`log_smooth_weight=0.6`, `final_smooth_t=3` `final_smooth_weight=0.30`, `impute_steps=2`,
`max_alpha=0.7`. Shape preserved, output non-negative, deterministic given the PCA/SVD random state.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.sparse import csr_matrix, eye as speye


def _pca_embed(Xn, n_pca, seed):
    k = min(n_pca, min(Xn.shape) - 1)
    return PCA(n_components=k, random_state=seed).fit_transform(Xn)


def _adaptive_affinity(emb, knn, decay):
    """MAGIC-style alpha-decay kernel: adaptive-bandwidth (knn-th distance) Gaussian."""
    n = emb.shape[0]
    k = min(knn * 3 + 1, n)
    nn = NearestNeighbors(n_neighbors=k).fit(emb)
    dist, idx = nn.kneighbors(emb)
    sigma = np.maximum(dist[:, knn], 1e-12)
    rows = np.repeat(np.arange(n), k); cols = idx.ravel()
    aff = np.exp(-((dist.ravel() / sigma[rows]) ** decay))
    A = csr_matrix((aff, (rows, cols)), shape=(n, n)); A = A + A.T
    rs = np.asarray(A.sum(axis=1)).ravel(); rs[rs == 0] = 1.0
    return A.multiply(1.0 / rs[:, None]).tocsr()


def _apply_vst(c, tr):
    if tr == "anscombe": return 2.0 * np.sqrt(c + 3.0 / 8.0)
    if tr == "ft":       return np.sqrt(c) + np.sqrt(c + 1.0)
    if tr == "sqrt":     return np.sqrt(c)
    raise ValueError(tr)


def _inv_vst(y, tr):
    if tr == "anscombe":
        x = (y / 2.0) ** 2 - 3.0 / 8.0
        for _ in range(8):
            s = np.sqrt(np.maximum(x + 3.0 / 8.0, 0.0)); x = x - (2.0 * s - y) * s
        return np.maximum(x, 0.0)
    if tr == "ft":
        out = np.zeros_like(y); m = y > 0; yy = y[m]
        s = np.maximum((yy * yy - 1.0) / (2.0 * yy), 0.0); x = s * s
        for _ in range(8):
            a = np.sqrt(np.maximum(x, 0.0)); b = np.sqrt(np.maximum(x + 1.0, 0.0))
            f = a + b - yy; fp = 0.5 / np.maximum(a, 1e-12) + 0.5 / np.maximum(b, 1e-12)
            x = np.maximum(x - f / fp, 0.0)
        out[m] = x; return out
    if tr == "sqrt":     return y ** 2
    raise ValueError(tr)


def _multiscale_diffuse(P, Y, t, dropout, decay=0.9):
    """Gene-wise weighted multi-scale diffusion: sum of P^i with per-gene decay + baseline."""
    cur = Y.copy(); wsum = Y.copy(); wtot = np.ones(Y.shape[1])
    base = decay * (0.2 + 0.8 * dropout)
    for i in range(1, t + 1):
        cur = P @ cur; wi = np.power(base, i)
        wsum += cur * wi[None, :]; wtot += wi
    return wsum / np.maximum(wtot[None, :], 1e-12)


def denoise(X, knn=12, t=7, n_pca=50, decay=2.0, diff_decay=0.9,
            transforms=("anscombe", "ft", "sqrt"),
            lowrank_components=30, lowrank_weight=0.10,
            log_smooth_t=4, log_smooth_weight=0.6,
            final_smooth_t=3, final_smooth_weight=0.30,
            impute_steps=2, max_alpha=0.7, seed=0):
    """TTT-Discover endpoint: multi-VST ensemble + low-rank SVD + log-space diffusion."""
    X = X.astype(np.float64)
    n_cells, n_genes = X.shape
    libsize_raw = X.sum(axis=1); libsize_raw[libsize_raw == 0] = 1.0
    dropout = np.mean(X == 0, axis=0)

    # graph built once on the primary (anscombe) VST embedding, with a self-loop
    Xp = _apply_vst(X, transforms[0])
    lib_p = Xp.sum(axis=1, keepdims=True); lib_p[lib_p == 0] = 1.0
    emb = _pca_embed(Xp / lib_p, n_pca, seed)
    P = _adaptive_affinity(emb, knn, decay)
    P = (1 - 0.5) * speye(n_cells, format="csr") + 0.5 * P
    rs = np.asarray(P.sum(axis=1)).ravel(); rs[rs == 0] = 1.0
    P = P.multiply(1.0 / rs[:, None]).tocsr()

    outs = []
    for tr in transforms:
        Xv = _apply_vst(X, tr)
        lib = Xv.sum(axis=1, keepdims=True); lib[lib == 0] = 1.0
        Xn = Xv / lib
        # impute exact zeros with a diffusion-weighted neighbour average
        nbr = Xn.copy()
        for _ in range(impute_steps):
            nbr = P @ nbr
        Yimp = Xn.copy(); Yimp[Xn == 0] = nbr[Xn == 0]
        Yimp = Yimp / np.maximum(Yimp.sum(axis=1, keepdims=True), 1e-12)
        diff = _multiscale_diffuse(P, Yimp, t, dropout, diff_decay)
        # adaptive gene-wise blend of raw-normalised and diffused
        var_o = Xn.var(axis=0); var_d = diff.var(axis=0)
        vred = np.clip((var_o - var_d) / (var_o + 1e-12), 0.0, 1.0)
        mu = Xn.mean(axis=0); mun = (mu - mu.min()) / (mu.max() - mu.min() + 1e-12)
        w = np.clip(dropout * vred * (1.0 - mun), 0.0, max_alpha)
        blended = Xn * (1.0 - w) + diff * w
        blended = blended / np.maximum(blended.sum(axis=1, keepdims=True), 1e-12)
        # restore VST-space library size BEFORE inverting
        outs.append(np.maximum(_inv_vst(blended * lib, tr), 0.0))

    # gene-wise VST ensemble: anscombe (low dropout), ft (high dropout), sqrt (bridge)
    if len(outs) == 3:
        wm = np.stack([(1 - dropout) ** 2, dropout ** 2, 2 * dropout * (1 - dropout)])
        wm /= np.maximum(wm.sum(0, keepdims=True), 1e-12)
        denoised = sum(outs[i] * wm[i][None, :] for i in range(3))
    else:
        denoised = np.mean(outs, axis=0)
    denoised = np.maximum(denoised, 0.0)

    # light low-rank TruncatedSVD refinement
    if lowrank_weight > 0:
        nc = min(lowrank_components, min(n_cells, n_genes) - 1)
        svd = TruncatedSVD(n_components=nc, random_state=seed, algorithm="randomized")
        low = svd.fit_transform(denoised) @ svd.components_
        denoised = np.maximum((1 - lowrank_weight) * denoised + lowrank_weight * low, 0.0)

    # extra diffusion in log-normalised space (the space the MSE is computed in)
    def log_diffuse(M, steps, weight):
        cs = M.sum(axis=1, keepdims=True); scale = 1e4 / np.maximum(cs, 1e-12)
        lg = np.log1p(M * scale)
        for _ in range(steps):
            lg = P @ lg
        return (1 - weight) * M + weight * (np.expm1(lg) * (cs / 1e4))
    if log_smooth_weight > 0 and log_smooth_t > 0:
        denoised = log_diffuse(denoised, log_smooth_t, log_smooth_weight)
    if final_smooth_weight > 0 and final_smooth_t > 0:
        denoised = log_diffuse(denoised, final_smooth_t, final_smooth_weight)

    return np.maximum(denoised, 0.0)
```
