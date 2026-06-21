MAGIC reached a held-out combined score of $0.8488$, clearing kNN, and the feedback named exactly where it stops short. Two things, both of which I had built in deliberately. First, I forced every gene through the same square-root transform — but a gene expressed in almost every cell and a gene that drops out in nine cells out of ten are not stabilized by the same nonlinearity, and treating them identically leaves variance on the table. Second, plain diffusion smooths in count space and never looks at the log-normalized space the MSE is actually computed in, nor at the global low-rank structure sitting underneath the local manifold geometry. I want to attack all three: gene-adaptive transforms, a low-rank refinement, and a final pass that smooths in the exact space the metric scores. The MAGIC diffusion graph stays as the backbone; everything else is built around it.

I propose the **TTT-Discover endpoint**, an adaptation of the denoiser that test-time-training/discover evolved to the top of the OpenProblems leaderboard (0.71 PBMC / 0.73 Tabula versus MAGIC's ~0.64). Its defining structural change is a **gene-wise multi-VST ensemble**. The square root is one variance-stabilizing transform for Poisson data, but not the only one and not uniformly best. The Anscombe transform $2\sqrt{x + 3/8}$ stabilizes Poisson variance more accurately at low counts; the Freeman–Tukey transform $\sqrt{x} + \sqrt{x+1}$ behaves better still in the deep-dropout regime where counts are mostly zeros and ones. Each suits a different part of the dropout spectrum — Anscombe for well-expressed genes, Freeman–Tukey for high-dropout genes, the plain square root as a bridge — so instead of picking one, I run the whole MAGIC-style pipeline three times, once under each transform, and ensemble the three outputs *gene-wise* by dropout fraction. With $p_g$ the fraction of cells in which gene $g$ is zero, the ensemble weights are $w^{\text{ans}}_g \propto (1-p_g)^2$, $w^{\text{ft}}_g \propto p_g^2$, $w^{\text{sqrt}}_g \propto 2p_g(1-p_g)$, normalized to sum to one — a smooth interpolation that pushes rarely-dropped genes toward Anscombe, constantly-dropped genes toward Freeman–Tukey, and the bridge in between. This directly cures the per-gene mis-stabilization MAGIC's single transform left in place. The graph $P$ itself is built once, on the Anscombe embedding, and reused across all three passes; I also mix in a self-loop, $P \leftarrow \tfrac{1}{2}I + \tfrac{1}{2}P$ re-normalized, so each cell retains half its own mass per step for stability.

Inside each transform's pass I am smarter than plain $P^t X$ in three ways. First, the exact zeros in a high-dropout gene are not biology, they are dropouts, so before diffusing I impute them: take a few diffusion steps to get a diffusion-weighted neighbor average, and fill only the zero entries with what the manifold says should be there, then renormalize. Second, instead of a single power $P^t$ I use a gene-wise weighted **multi-scale diffusion** — accumulate $X, PX, P^2X, \dots, P^tX$ with per-gene decaying weights and a guaranteed baseline,

$$\hat{X} \;=\; \frac{\sum_{i=0}^{t} b_g^{\,i}\, P^{i} X}{\sum_{i=0}^{t} b_g^{\,i}}, \qquad b_g = 0.9\,(0.2 + 0.8\,p_g),$$

so each gene gets the amount of smoothing its dropout warrants — high-dropout genes carry weight further out into the powers — while the additive baseline guarantees every gene is smoothed at least a little. Third, after diffusing I **adaptively blend** the diffused signal back toward the raw normalized signal with a per-gene weight $w_g = \text{clip}\big(p_g \cdot \text{vred}_g \cdot (1 - \tilde\mu_g),\, 0,\, 0.7\big)$, where $\text{vred}_g$ is the fraction of variance diffusion removed for that gene and $\tilde\mu_g$ is its min-max-scaled mean expression. Genes that diffusion clearly helps (high dropout, large variance reduction, low mean) get more of the diffused version; genes where diffusion mostly destroyed signal keep more of their raw values. This blend is the safety valve that stops the ensemble from over-smoothing the genes that did not need it.

On top of the ensemble sit the two global refinements that close the remaining gaps. The first is a **light truncated-SVD low-rank refinement**: the cells live on a low-dimensional manifold, so the denoised matrix should be approximately low-rank in the gene directions too, and a truncated SVD captures that global structure local diffusion misses. I reconstruct from the top components and blend it in — but only with a small weight (0.10). The true rate in this data is deliberately *not* exactly low-rank: there is multiplicative biological over-dispersion on top of the low-rank signal, so an aggressive low-rank weight would throw away real per-gene variation and hurt the MSE. A light touch recovers the dominant global structure and tightens the Poisson likelihood without flattening the over-dispersion. The second refinement closes the last gap the feedback named: I smooth **in the log-normalized space the MSE is actually computed in**. Everything so far diffused in count or VST space, but the metric rescales each cell to a fixed total, takes $\log1p$, and measures squared error there — so I add diffusion passes performed in exactly that space: rescale to the target total, $\log1p$, diffuse with the same $P$ for a few steps, $\text{expm1}$, rescale back, and blend. Smoothing directly in the scoring space is the most direct possible attack on the MSE term, and I do it twice, once with a stronger weight (steps 4, weight 0.6) and once as a final lighter polish (steps 3, weight 0.30).

So the full endpoint builds the diffusion graph once on the Anscombe embedding with a self-loop; runs the impute-then-multiscale-diffuse-then-adaptive-blend pipeline under Anscombe, Freeman–Tukey, and square root; ensembles the three gene-wise by dropout; applies the light truncated-SVD refinement; and finishes with the two log-normalized-space diffusion passes. Every piece exists to close a gap MAGIC left open, and none is free — each adds a knob and the knobs interact, so I tune them jointly on the tune set and accept what the held-out sets say. I expect the gain to land most on the term the gene-adaptive transforms, zero-imputation, and low-rank refinement target — the absolute per-gene rate the Poisson NLL reads — with the log-space passes holding the MSE term steady against the cost of the low-rank step. What it cannot do is exceed the true-rate ceiling: the multiplicative biological over-dispersion baked into $\Lambda$ is irreducible noise no denoiser can recover, so this is the top of the ladder, not a claim of perfection.

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
