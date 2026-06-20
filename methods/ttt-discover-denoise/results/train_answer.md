I want to push single-cell RNA-seq denoising past what MAGIC's graph diffusion achieves, and the
analysis points at three specific places where MAGIC stops short. First, it forces every gene through
the same square-root transform — but a gene expressed in almost every cell and a gene that drops out
nine times in ten are not stabilized by the same nonlinearity, and treating them identically leaves
variance on the table. Second, plain diffusion smooths in the count space and never looks at the
log-normalized space the MSE is actually computed in, nor at the global low-rank structure that sits
beneath the local manifold geometry. I want to attack all three at once — gene-adaptive transforms, a
low-rank refinement, and a final pass that smooths in the exact space the metric scores — keeping
MAGIC's adaptive-bandwidth diffusion graph as the backbone and building everything else around it.

The method I propose is the TTT-Discover denoiser. The cleanest win is the transforms. The square
root is one variance-stabilizing transform for Poisson data, but the Anscombe transform $2\sqrt{x +
3/8}$ stabilizes Poisson variance more accurately at low counts, and the Freeman-Tukey transform
$\sqrt{x} + \sqrt{x+1}$ behaves better still in the deep-dropout regime where counts are mostly zeros
and ones. Each suits a different part of the dropout spectrum — Anscombe for well-expressed genes,
Freeman-Tukey for high-dropout genes, the bare square root as a bridge — so rather than pick one I run
the whole MAGIC-style pipeline three times, once under each transform, and ensemble the three denoised
outputs *gene-wise*, weighting each gene toward the transform that suits its dropout. With dropout
fraction $p_g$ per gene I weight Anscombe by $(1-p_g)^2$, Freeman-Tukey by $p_g^2$, and square root by
$2p_g(1-p_g)$, then renormalize — so the weights interpolate smoothly with dropout and a gene is sent
to whichever transform stabilizes it best. This gene-adaptive multi-VST ensemble is the single biggest
structural change from MAGIC and directly targets the per-gene mis-stabilization.

Inside each transform's pass I am smarter than plain $P^t X$ in three ways. The hard zeros in a
high-dropout gene are dropouts, not biology, so before diffusing I impute the exact zeros with a
diffusion-weighted neighbor average — fill the holes with what the manifold says should be there.
Instead of a single power I use a gene-wise weighted multi-scale diffusion, accumulating $X, PX, P^2X,
\dots, P^tX$ with weights that decay per gene as $b_g^i$ where $b_g = \text{decay}\cdot(0.2 + 0.8\,
p_g)$, so each gene gets the smoothing its dropout warrants with a guaranteed baseline that every gene
is smoothed at least a little. And after diffusing I blend the diffused signal back toward the raw
normalized signal with a per-gene weight $w_g$ that rises with dropout, with how much variance
diffusion removed, and with low mean expression — genes diffusion clearly helps get more of it, genes
where it mostly destroyed signal keep more of their raw values. This adaptive blend is the safety
valve against over-smoothing the genes that did not need it.

On top of the ensemble sit the two global refinements that close the last two gaps. The cells live on
a low-dimensional manifold, so the denoised matrix should be approximately low-rank in the gene
directions; a truncated SVD captures that global structure local diffusion misses, and I reconstruct
from the top components and blend it back with a *modest* weight. I have to be careful here, because
the true rate in this data is deliberately not exactly low-rank — there is multiplicative biological
over-dispersion on top of the low-rank signal — so an aggressive low-rank weight would throw away real
per-gene variation and hurt the MSE; I keep it light, $\text{lowrank\_weight}=0.10$, enough to
recover dominant structure and tighten the Poisson likelihood without flattening the over-dispersion.
The second refinement is the most direct attack on the MSE term: I smooth in the log-normalized space
the metric is computed in. Everything so far diffused in count or VST space, but the metric normalizes
each cell to a fixed total, takes $\log 1p$, and measures squared error there — so I add a diffusion
pass performed in exactly that space (rescale to the target total, $\log 1p$, diffuse with the same
$P$, $\text{expm1}$, rescale back) and blend it in, doing it twice: once mid-pipeline and once as a
final polish. The full endpoint, then, builds the diffusion graph once on the Anscombe embedding with
a self-loop mixed in for stability, runs the impute-diffuse-blend pipeline under all three transforms,
ensembles them gene-wise by dropout, applies the light truncated-SVD refinement, and finishes with the
two log-normalized-space diffusion passes. Each piece closes a gap MAGIC left open, and none can
exceed the true-rate ceiling — the over-dispersion is irreducible noise — so this is the top of the
ladder, not a claim of perfection.

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
