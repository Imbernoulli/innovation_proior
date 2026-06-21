The raw baseline measured exactly $0.0000$ on both normalized terms — it pools nothing, so the entire gap to the true rate is still open. The single idea that lifts a denoiser off that floor is that cells in the same biological state are independent noisy measurements of one underlying rate, so if I can find, for each cell, the other cells that share its state and average them, the Poisson noise averages down while the shared signal survives. I want to start with that idea in its crudest honest form before reaching for anything graph-theoretic, both because it is the natural next rung and because I want to see how much of the gap to the rate a bare neighbor average can close.

I propose **kNN-smoothing** in the style of Wagner, Yan & Yanai (2018): for each cell, find its $k$ nearest neighbors and replace its profile with the uniform average over itself and those $k$ cells. Two decisions inside that one sentence carry the whole method. The first is *what space distance is measured in*. Raw counts are a terrible space — a cell sequenced more deeply looks far from an identical shallow cell purely because its counts are larger — so I have to remove the depth before comparing, which means library-size-normalizing each cell to a common total. And UMI counts are heavily right-skewed and heteroscedastic, so without a stabilizing transform a few high-expression genes would dominate any Euclidean distance. The square root is the canonical variance-stabilizing transform for Poisson data: for $x \sim \text{Poisson}(\lambda)$, $\text{Var}(\sqrt{x})$ is approximately constant in $\lambda$, so the transformed distance is no longer hijacked by the loudest genes. So I square-root the counts, normalize each cell's $\sqrt{x}$ vector to a common library size, and measure neighbors in *that* space.

The second decision is the one I am deliberately crude about: *where* I look for neighbors. The classical Wagner-style baseline picks neighbors directly on the observed, noisy normalized profiles — it does not first denoise the space it searches in. That is the defining character of this rung and also its known weakness. In a high-dropout regime the observed profiles are sparse and noisy, so the nearest-neighbor relation is itself corrupted: a cell's true neighbors can be pushed away and impostors pulled in, purely by which genes happened to drop out in each cell. I accept that weakness on purpose, because curing it is exactly what the next rung's graph construction is for. Then the averaging is a *hard, uniform* pool: I take the cell plus its $k$ neighbors and average their normalized profiles with equal weight, so the boundary at the $k$-th neighbor is a cliff. After averaging in the stabilized normalized space I invert the transform — square the result and multiply each cell back by its own original library size — so the output lands on the count scale the Poisson metric expects and each cell keeps its own depth rather than being flattened to the pool's.

The one knob, $k$, is a genuine bias-variance lever, and its imperfection is structural rather than incidental. Small $k$ averages few cells, removing little noise but distorting little signal; large $k$ removes more noise but, because the pool is a hard uniform average, it inevitably reaches across the boundary of the local state and blends in cells that are not quite the same, smearing real biological variation. Worse, no single global $k$ can be right everywhere: a cell in a dense region has many genuine neighbors and could tolerate a large pool, while a cell on the edge of a trajectory has few and a large $k$ drags in strangers. That a single global $k$ with a hard uniform pool cannot serve both the dense and the sparse regions of the manifold is the structural flaw of the whole approach. I tune one value on the tune set — $k = 10$ is the bias-variance compromise it settles on — and report it honestly, knowing the two named weaknesses (neighbors chosen on noisy profiles, and a non-adaptive hard pool) are precisely what the affinity-graph diffusion rung exists to cure.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def denoise(X, knn=10):
    """Wagner-style kNN-smoothing: neighbours on NOISY profiles, hard uniform average."""
    Xs = np.sqrt(X.astype(np.float64))                 # variance-stabilising transform
    lib = Xs.sum(axis=1, keepdims=True); lib[lib == 0] = 1.0
    Xn = Xs / lib                                       # library-size normalise
    n = X.shape[0]
    k = min(knn + 1, n)
    nn = NearestNeighbors(n_neighbors=k).fit(Xn)        # neighbours in noisy normalised space
    _, idx = nn.kneighbors(Xn)
    Y = Xn[idx].mean(axis=1)                            # uniform hard average over cell + k neighbours
    Y = Y * lib                                         # restore each cell's library size
    return np.square(Y)                                 # undo sqrt → count scale
```
