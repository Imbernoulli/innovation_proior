kNN-smoothing closed about 80% of the gap to the rate, reaching a held-out combined score of $0.7977$, but its two weaknesses showed up cleanly in the split between terms: the Poisson term recovered well ($0.867$) while the log-normalized MSE term lagged at $0.728$. Both failures trace to the same two decisions I made crudely on purpose. I chose neighbors on the raw, noisy profiles, so some of my pooling was simply wrong — impostors dragged in by dropout, true neighbors pushed away. And the pool was a hard uniform average with a single global $k$: every neighbor counted equally, the boundary at the $k$-th neighbor was a cliff, and one bandwidth had to serve both dense and sparse regions of the manifold. Curing both at once is what I propose here.

The method is **MAGIC** (van Dijk et al., 2018), a graph-diffusion denoiser, and it attacks the two flaws in two distinct moves. First, the space. Instead of measuring distances in the full noisy gene space, I square-root-transform and library-normalize as before, then project the cells onto their top principal components — a few dozen of them. Most genes are noise in any given cell; the leading PCs capture the shared low-dimensional structure the cells actually live on, the trajectories and branches, and discard the per-gene Poisson jitter. Distances in that denoised embedding are far more reliable than distances on raw profiles, which directly answers the first complaint. Second, the weighting. Rather than a hard top-$k$ set, I build a soft affinity that falls off smoothly with distance and, crucially, whose bandwidth adapts to local density. A cell in a dense region should have a narrow kernel because its genuine neighbors are close; a cell in a sparse region should have a wide one because its neighbors are farther but still real. I get this by setting each cell's kernel width $\sigma_i$ from its own distance to its $k$-th neighbor, so the kernel is automatically narrow where cells are packed and wide where they are spread. This is MAGIC's alpha-decay kernel,

$$A_{ij} = \exp\!\Big(-\big(d_{ij}/\sigma_i\big)^{\alpha}\Big),$$

with $\sigma_i$ the adaptive per-cell bandwidth and $\alpha$ controlling how sharply the affinity decays. I symmetrize $A$ so the relation is mutual, then row-normalize it into a Markov transition matrix $P$, where $P_{ij}$ is the probability of stepping from cell $i$ to cell $j$ in one diffusion step — large for similar cells, smoothly small for dissimilar ones.

The part that genuinely separates this from kNN is what I do with $P$. I do not stop at immediate neighbors; I diffuse:

$$\hat{X} = P^{t} X.$$

One step of $P$ averages each cell with its affinity-weighted neighbors, exactly like a soft kNN. But $P^t$ for $t > 1$ lets information flow *transitively* — cell $i$ borrows from its neighbors' neighbors, weighted by the probability of reaching them in $t$ steps along the manifold. On a smooth trajectory this is precisely right: cells a few hops apart along the continuum should share information, and the powered transition matrix reaches them through the chain of intermediate cells rather than requiring them to be direct nearest neighbors. The hard cliff of kNN is gone, replaced by a smooth, distance-weighted, transitively-pooled average whose bandwidth adapts to the data. The diffusion runs in the same square-root, library-normalized space the graph was built in, then I invert — square the diffused values and restore each cell's library size — so the output lands back on the count scale.

The parameter that matters most is $t$, and it is the same bias-variance lever $k$ was but better-behaved: too small under-smooths, too large over-diffuses and eventually washes every cell toward the global mean, erasing the very biological variation I am trying to preserve. I tune it small — $t = 2$ — because the affinity weighting already does most of the work in a single step; the powers are there to reach transitively, not to grind everything flat. The kernel decay $\alpha = 2$ (Gaussian-like) and the neighbor count $k = 10$ that sets the bandwidth are fixed to standard values, with the PCA taken to $50$ components. I expect this to clear kNN specifically on the MSE term, because the reliable embedding and adaptive weighting recover the local geometry that hard uniform pooling smeared. What it will not fix — and what becomes the endpoint's target — is two things: I am still committing to a *single* variance-stabilizing transform for every gene regardless of its dropout rate, and plain count-space diffusion never directly targets the log-normalized space the MSE is actually computed in, nor the global low-rank structure a factorization could capture on top of the local geometry.

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
