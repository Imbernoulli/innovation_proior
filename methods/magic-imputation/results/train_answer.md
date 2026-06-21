Single-cell RNA sequencing measures a cells $\times$ genes matrix of captured mRNA counts, but the assay only sees a small random slice of each cell — on the order of 5-15% of the transcriptome makes it through capture and amplification. A gene that is genuinely on, but expressed at low or moderate level, therefore comes back as a flat zero in many cells. The matrix is mostly zeros, and the nonzero entries are small and noisy. This "dropout" is not cosmetic. Take two genes that rise and fall together along some biological process, $y$ roughly proportional to $x$ along the trajectory; thin both coordinates by independent random sampling, and in a large fraction of cells $x$ survives while $y$ drops to zero, or vice versa. Instead of a tidy diagonal trend the gene-gene scatter becomes a cloud pinned to the axes, $(x,0)$ and $(0,y)$ everywhere; the correlation is dragged toward zero, a smooth trajectory looks like a discontinuous blob, and rare intermediate states are buried. The signal is not gone — it is shredded by per-cell missingness, and that is the enemy a recovery method must beat.

The obvious fixes each fail in a specific way. Analyzing raw counts inherits the attenuation directly. Local averaging without a manifold — smoothing each cell against a fixed Euclidean neighborhood, or replacing each cell by its cluster mean — does denoise, but clustering imposes discrete states on what is usually a continuum, collapsing within-cluster variation and erasing exactly the transitions and rare states one wants to keep; and a fixed Euclidean radius ignores the manifold's curvature and the wildly uneven sample density, over-connecting dense regions and orphaning sparse ones, all while using raw distances that are themselves dominated by dropout noise. Model-based imputation posits a parametric count likelihood (zero-inflated or negative-binomial) and fits it gene-by-gene or with low-rank structure, but it commits to a noise model and a notion of which zeros are "true," and crucially it never exploits the full nonlinear cell-cell geometry. Diffusion maps and Laplacian eigenmaps capture that geometry beautifully but were built to produce an embedding for visualization; they return coordinates in a new space, not recovered values in the original cells $\times$ genes space, so they leave the count matrix itself unrecovered.

The asymmetry to exploit is that dropout is per-cell and roughly independent across cells: where one cell lost gene $y$, some other cell probably kept it, so the information needed to fill cell $i$'s value for $y$ is physically present elsewhere in the matrix. That reframes imputation as an averaging problem, and the whole difficulty becomes defining the right neighborhood to average over. The structural fact that makes this tractable is the manifold assumption: regulated, coupled gene expression confines real cells to a low-dimensional, curved surface inside the 20,000-gene space, and cells in a continuous process slide along it. "Biologically the same cell" then means "nearby on the manifold," a far more robust notion than raw Euclidean distance, and the practical handle on the manifold is a nearest-neighbor graph whose local pieces constrain each other into a coherent global geometry. I propose MAGIC — Markov Affinity-based Graph Imputation of Cells — which builds that graph, turns it into a random-walk operator, and diffuses the expression values over it.

Everything downstream rides on the graph being right, so the first job is to make even the local distances honest. I library-size normalize each cell — divide by its total count and multiply back by the median total — to remove the pure technical confound of one cell being sequenced more deeply than another. Counts are roughly Poisson-like, with variance growing with the mean, so a highly expressed gene dominates Euclidean distance through scale alone and its noise is heteroscedastic; a square-root transform stabilizes the variance of count data, flattening that out so genes weigh more evenly (a log would do similarly but mishandles the zeros, whereas $\sqrt{\cdot}$ is gentle and defined at zero). Even then, the per-gene dropout noise sits in all 20,000 coordinates and Euclidean distance sums it over every one — death by a thousand noisy dimensions — so I project onto the top principal components (a few dozen to a hundred, default $n_{\text{pca}}=100$): the leading PCs hold the coordinated, high-variance biology while the tail PCs are mostly the independent per-gene noise I want to drop, and the distances become both cleaner and cheaper to compute. Neighbor distances $d(i,j)$ are then taken in this PCA space.

Distances become a weighted graph through an affinity kernel that is high for near cells and decays smoothly to essentially zero for far cells, so a cell never talks to the whole dataset. The defining choice is the bandwidth. A single global $\sigma$ is wrong everywhere precisely because single-cell density is non-uniform: tune $\sigma$ for dense regions and sparse cells fall off the graph with no neighbors inside one $\sigma$, getting no imputation; tune it for sparse regions and dense cells each connect to hundreds, over-smoothing. The right scale is local, so I make the bandwidth adaptive per cell, setting $\sigma_i$ to cell $i$'s own distance to its $\text{knn}$-th nearest neighbor. By construction every cell then has roughly the same number of meaningfully-weighted neighbors regardless of local density — a dense cell gets a tight kernel, a sparse cell reaches far enough to stay connected. The affinity is

$$A(i,j) = \exp\!\left(-\left(\frac{d(i,j)}{\sigma_i}\right)^{\alpha}\right),$$

where the exponent $\alpha$ (the decay) is a knob on the tails: $\alpha=2$ is the Gaussian, whose tails fall off very fast and can leave borderline cells under-connected across density gaps, while a smaller $\alpha$ — default $\text{decay}=1$, an exponential-style decay — keeps a softer tail that connects cells a bit further out and makes the graph more robustly connected. The neighbor index $\text{knn}$ that sets $\sigma_i$ should be as small as possible while the graph stays connected, keeping the geometry faithful to the manifold's curvature without fragmenting into disconnected pieces that imputation cannot cross. Since the kernel is already negligible past a few $\sigma$, I truncate to at most $\text{knn}_{\max}=3\cdot\text{knn}$ nonzero neighbors per cell — about $3\sigma$, catching essentially all non-negligible weight — keeping $A$ sparse and fast.

This $A$ is not symmetric, because $A(i,j)$ uses $\sigma_i$ while $A(j,i)$ uses $\sigma_j$, and the clean spectral theory I am about to lean on wants a symmetric affinity. I symmetrize additively, $A \leftarrow (A + A^{\mathsf T})/2$, which also rescues outliers: a cell that picks neighbors but is picked by none still receives edges from the transpose, so it is not orphaned. Then I make the averaging honest by turning $A$ into a row-stochastic operator, dividing each row by its total affinity,

$$M(i,j) = \frac{A(i,j)}{\sum_k A(i,k)},$$

so every row is a probability distribution over neighbors. $M$ is exactly a Markov transition matrix: $M(i,j)$ is the probability that a one-step random walk from cell $i$ lands on cell $j$, strongly favoring close neighbors. One step of the walk applied to the data, $(MX)(i,:) = \sum_j M(i,j)\,X(j,:)$, is precisely the transition-weighted mean of neighbors' expression, and dropped zeros get filled by neighbors that kept the transcript.

One step, though, trusts $M$'s edges literally, and some of those edges are spurious shortcuts created by a lucky alignment of dropouts between two cells that are not really close on the manifold. The fix is to run the walk longer so that an edge is trusted in proportion to how robustly the two cells are connected through many independent paths. Powering does this: $M^2(i,j)=\sum_k M(i,k)M(k,j)$ sums probability over all length-2 paths, so a truly close pair linked by many short paths accumulates probability while a single thin shortcut is diluted as probability leaks to true neighbors at each hop. The imputation is therefore data diffusion,

$$X_{\text{imputed}} = M^{t}\,X,$$

each cell relaxing toward its $t$-step neighborhood consensus while moving only along graph edges. The spectral picture says exactly why this denoises. The symmetric conjugate $D^{-1/2} A D^{-1/2}$ is similar to $M$, so $M$ has real eigenvalues, and for a stochastic matrix they lie in $[0,1]$; the top eigenvalue is $1$ with the constant vector as its mode (the stationary distribution / global mean), the modes with $\lambda$ near $1$ are smooth, slowly-varying functions along the manifold — the real biological gradients — and the modes with $\lambda$ near $0$ are high-frequency oscillations where the noise lives. Writing $X$ in this eigenbasis, $M^t$ multiplies the component with eigenvalue $\lambda$ by $\lambda^t$, shrinking every $\lambda<1$ mode and shrinking it harder the smaller $\lambda$ is: $0.1^5\approx 10^{-5}$ is gone while $0.95^5\approx 0.77$ is mostly kept. $M^t$ is a graph low-pass filter, preserving slow gradients and annihilating fast noise with $t$ setting the cutoff — the discrete face of the heat kernel $e^{-t\Delta}$ on the manifold.

That same spectrum warns about $t$. Keep powering and every $\lambda<1$ eventually hits zero, leaving only the constant mode, so $M^t X$ collapses every cell toward the single global mean and melts distinct states into mush. Thus $t$ is the crucial dial: too small and dropout noise remains, too large and the biology is washed out, with a sweet spot between. Without ground truth I find it by watching how fast the data still changes per step — early steps remove a lot of noise so $X_{\text{imputed}}$ changes sharply, then once the noise is gone further steps only nibble at the slow modes and the change plateaus, and the elbow between these regimes is where to stop. I quantify the per-step change as the Procrustes disparity between successive iterates $X_t$ and $X_{t-1}$ (equivalently an $R^2 = 1 - \text{SSE}/\text{SST}$ between them dropping below a few percent change, taking the second $t$ past threshold to be safe against a noisy reading) and stop when it falls under a small threshold; for data that arrives fairly clean a small fixed default of $t=3$ works without the search. This locality is also why MAGIC does not collapse genuinely distinct populations: if two cell types are separated on the manifold there are few edges between them, $M^t$ keeps probability within each population for moderate $t$, and bleeding across thin bridges only happens at large $t$ — exactly the over-smoothing regime the $t$-selection stops short of.

A final detail concerns magnitudes. Because $M^t$ is an averaging operator with rows summing to one, $M^t X$ stays in the convex hull of the data and cannot blow up, but averaging compresses peaks: a gene high in a few cells gets pulled toward its neighbors, so the imputed per-gene dynamic range shrinks. If the original expression scale matters, an optional rescale restores it by scaling each imputed gene so its max equals the 99th percentile of the original gene's values — the 99th rather than the literal max so a single outlier count is ignored — recovering per-gene magnitude while keeping the now-smooth relative structure across cells.

The affinity-to-Markov construction (the adaptive bandwidth, the kernel, the additive symmetrization, and the row-normalization to $M = D^{-1}A$) is handled by `graphtools.Graph`, which exposes `diff_op` as the transition matrix $M$; the diffusion $M^t X$, the $t$-selection, and the rescale are the imputation itself.

```python
import numpy as np
from scipy import spatial
from sklearn.preprocessing import normalize   # l1 row-normalize -> stochastic
import graphtools                              # affinity kernel + diffusion operator


def preprocess(counts):
    """Library-size normalize + sqrt; filter empties first."""
    counts = counts[counts.sum(1) > 0]
    counts = counts[:, np.asarray(counts.sum(0)).ravel() > 0]
    libsize = counts.sum(1, keepdims=True)
    counts = counts / libsize * np.median(libsize)
    return np.sqrt(counts)


class MAGIC:
    """Markov Affinity-based Graph Imputation of Cells.

    knn      : neighbor index whose distance sets the per-cell bandwidth (ka)
    knn_max  : cap on nonzero neighbors (default 3 * knn ~ 3 sigma of the kernel)
    decay    : kernel exponent alpha in exp(-(d/sigma)^alpha); Gaussian is alpha=2
    t        : diffusion steps (power of M); 'auto' finds the elbow
    n_pca    : PCA dims for the neighbor geometry
    """

    def __init__(self, knn=5, knn_max=None, decay=1, t=3, n_pca=100):
        self.knn, self.decay, self.t, self.n_pca = knn, decay, t, n_pca
        self.knn_max = knn_max if knn_max is not None else 3 * knn

    def fit(self, X):
        # PCA -> kNN distances -> adaptive affinity A(i,j)=exp(-(d/sigma_i)^decay)
        # -> additive symmetrization -> M = D^{-1} A (row-stochastic Markov op).
        self.graph = graphtools.Graph(
            X, n_pca=self.n_pca, knn=self.knn, knn_max=self.knn_max,
            decay=self.decay, thresh=1e-4)
        self.diff_op = self.graph.diff_op          # the Markov transition matrix M
        return self

    def _select_t(self, X, t_max=20, thresh=1e-3):
        # diffuse one step at a time; stop when successive iterates stop moving
        Xt = np.asarray(X, dtype=float)
        prev = Xt.copy()
        for _ in range(t_max):
            Xt = self.diff_op.dot(Xt)
            _, _, disparity = spatial.procrustes(prev, Xt)
            prev = Xt.copy()
            if disparity < thresh:
                break
        return Xt

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.t == "auto":
            return self._select_t(X)
        Mt = np.linalg.matrix_power(self.diff_op.toarray(), self.t)
        return Mt.dot(X)                            # X_imputed = M^t X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def rescale(X_imputed, X_original, pct=99):
    """Optional: restore per-gene magnitude lost to averaging."""
    p = np.percentile(X_original, pct, axis=0)
    m = X_imputed.max(axis=0)
    scale = np.divide(p, m, out=np.zeros_like(p), where=m > 0)
    return X_imputed * scale


# usage
# X = preprocess(raw_counts)
# X_magic = MAGIC(knn=5, decay=1, t='auto').fit_transform(X)
```
