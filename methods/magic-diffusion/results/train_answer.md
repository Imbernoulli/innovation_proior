I want to denoise a single-cell RNA-seq UMI count matrix by pooling information across biologically
similar cells, and I am reacting to two specific failures of naive kNN-smoothing. The first is that
kNN chooses neighbors on the raw, noisy profiles, so in a high-dropout regime some of the pooling is
simply wrong — impostor cells dragged in by dropout, true neighbors pushed away. The second is that
the pool is a hard uniform average with a single global $k$: every neighbor counts equally, the
boundary at the $k$-th neighbor is a cliff, and one bandwidth has to serve both dense and sparse
regions of the data manifold. Both complaints point the same way — I need a smoother, weighted,
geometry-aware notion of "similar cell," and I need to stop trusting a single noisy distance to
define a hard set.

The method is MAGIC, and it cures both failures with a diffusion on a cell-cell affinity graph.
First the space: instead of measuring distances on the full noisy gene vectors, I square-root-
transform, library-size-normalize, and project the cells onto their top principal components. Most
genes are noise in any given cell; the leading PCs capture the shared low-dimensional structure — the
trajectories and branches the cells actually live on — and discard the per-gene Poisson jitter, so
distances in that embedding are far more reliable than distances on raw profiles. That directly fixes
the first failure. Then the weighting, which fixes the second: rather than a hard top-$k$ set I build
a soft affinity that falls off smoothly with distance and whose bandwidth adapts to local density. I
set each cell's kernel width from its own distance to its $k$-th neighbor, so the kernel is
automatically narrow where cells are packed and wide where they are spread out. This is the alpha-
decay kernel,
$$A_{ij} = \exp\!\left(-\left(\tfrac{d_{ij}}{\sigma_i}\right)^{\alpha}\right),$$
with $\sigma_i$ the adaptive per-cell bandwidth (the distance to the $k$-th neighbor) and $\alpha$
controlling how sharply the affinity decays. I symmetrize $A$ so the relation is mutual and
row-normalize it into a Markov transition matrix $P$, where $P_{ij}$ is the probability of stepping
from cell $i$ to cell $j$ in one diffusion step — large for similar cells, smoothly small for
dissimilar ones.

The part that genuinely separates this from kNN is that I do not stop at immediate neighbors. I
diffuse:
$$\hat X = P^{t} X.$$
One step of $P$ averages each cell with its affinity-weighted neighbors, like a soft kNN. But $P^t$
for $t>1$ lets information flow *transitively* — cell $i$ borrows from its neighbors' neighbors,
weighted by the probability of reaching them in $t$ steps along the manifold. On a smooth trajectory
this is exactly right: cells a few hops apart along the continuum should share information, and the
powered transition matrix reaches them through the chain of intermediate cells rather than requiring
them to be direct nearest neighbors. The hard cliff of kNN is gone, replaced by a smooth, distance-
weighted, transitively-pooled average whose bandwidth adapts to the data. I run the diffusion in the
same square-root, library-normalized space I built the graph in, then invert — square the diffused
values and restore each cell's library size — so the output lands on the count scale the metric
expects.

The free parameter that matters most is $t$, the number of diffusion steps, and it is the same
bias-variance lever $k$ was but better-behaved: small $t$ under-smooths, large $t$ over-diffuses and
eventually washes every cell toward the global mean, erasing the very biological variation I am trying
to preserve. I find a small value, $t=2$, is right, because the affinity weighting already does most
of the work in a single step and the powers are there to reach transitively, not to grind everything
flat. The kernel decay I fix to a Gaussian-like $\alpha=2$ and the bandwidth neighbor count to
$k=10$, both standard. Two gaps remain that this method does not address — it forces every gene
through the same square-root transform regardless of its dropout rate, and plain diffusion never
directly targets the log-normalized space the MSE is computed in nor the global low-rank structure a
factorization could capture — but those are the targets of a later, richer method.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix

def denoise(X, knn=10, t=2, n_pca=50, decay=2.0, seed=0):
    Xs = np.sqrt(X.astype(np.float64))                      # sqrt VST
    lib = Xs.sum(axis=1, keepdims=True); lib[lib == 0] = 1.0
    Xn = Xs / lib                                           # library-size normalise
    k_pca = min(n_pca, min(Xn.shape) - 1)
    emb = PCA(n_components=k_pca, random_state=seed).fit_transform(Xn)   # denoised embedding

    n = emb.shape[0]
    k = min(knn * 3 + 1, n)
    nn = NearestNeighbors(n_neighbors=k).fit(emb)
    dist, idx = nn.kneighbors(emb)
    sigma = np.maximum(dist[:, knn], 1e-12)                 # adaptive bandwidth = knn-th distance
    rows = np.repeat(np.arange(n), k); cols = idx.ravel()
    aff = np.exp(-((dist.ravel() / sigma[rows]) ** decay))  # alpha-decay kernel
    A = csr_matrix((aff, (rows, cols)), shape=(n, n)); A = A + A.T   # symmetrise
    rs = np.asarray(A.sum(axis=1)).ravel(); rs[rs == 0] = 1.0
    P = A.multiply(1.0 / rs[:, None]).tocsr()               # row-stochastic transition matrix

    Y = Xn.copy()
    for _ in range(t):                                      # diffuse: X_hat = P^t X
        Y = P @ Y
    return np.square(Y * lib)                               # restore depth, undo sqrt
```
