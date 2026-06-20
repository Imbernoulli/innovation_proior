I want to denoise a single-cell RNA-seq UMI count matrix, where droplet capture undersamples so
severely that most expressed genes read as hard zeros. The raw counts, run through the molecular
cross-validation harness, score exactly zero because the identity map pools nothing across cells. The
single idea that lifts a denoiser off that floor is the premise that makes denoising possible at all:
cells in the same biological state are independent noisy measurements of one underlying rate, so if I
can find, for each cell, the other cells that share its state and average them together, the Poisson
noise averages down while the shared signal survives. Before reaching for anything graph-theoretic I
want this idea in its crudest honest form, both because it is the natural next step and because I want
to see how much of the gap to the true rate a bare neighbor average can close.

The method is kNN-smoothing: for each cell, find its $k$ nearest neighbors and replace its profile
with the uniform average over itself and those neighbors. Two decisions inside that sentence carry the
weight. The first is the space I measure distance in. Raw counts are a terrible space — a cell that
happened to be sequenced deeper looks far from an identical shallow cell purely because its counts are
larger — so I must remove the depth before comparing, by library-size normalizing each cell to a
common total. And the counts are heavily right-skewed and heteroscedastic, so a few high-expression
genes would dominate a Euclidean distance; I therefore apply a variance-stabilizing transform first.
The square root is the canonical choice for Poisson data, because it makes the variance of a count
roughly constant regardless of its mean, so the distance is not hijacked by the loudest genes. I
square-root the counts, normalize to a common library size, and only then measure neighbors. The
second decision is one I make deliberately crude: where to look for neighbors. The classical
Wagner-style kNN-smoothing picks neighbors directly on the observed, noisy normalized profiles — it
does not first denoise the space it searches in. That is the character of the method and also its
known weakness, because in a high-dropout regime the observed profiles are sparse and noisy, so the
nearest-neighbor relation is itself corrupted: a cell's true neighbors may be pushed away and
impostors pulled in, purely by which genes happened to drop out. I accept that weakness on purpose,
because curing it is exactly what a later graph construction is for.

The averaging is a *hard, uniform* pool: I take the cell plus its $k$ neighbors and average their
normalized profiles with equal weight, so the boundary at the $k$-th neighbor is a cliff. After
averaging in the stabilized normalized space I undo the transform — square the result and multiply
each cell back by its own original library size — so the output lands on the count scale the Poisson
metric expects and each cell keeps its own depth rather than being flattened to the average. Choosing
$k$ is a genuine bias-variance lever: small $k$ averages few cells, removing little noise but
distorting little signal; large $k$ removes more noise but, because the pool is a hard uniform
average, inevitably reaches across the boundary of the local state and blends in cells that are not
quite the same, smearing real biological variation. There is no $k$ that is right everywhere — a cell
in a dense region has many genuine neighbors while a cell on the edge of a trajectory has few — and
that a single global $k$ cannot serve both is the structural flaw of the whole approach. I set $k
\approx 10$ as a reasonable compromise. The two named weaknesses — neighbors chosen on noisy profiles,
and a hard uniform pool with one global $k$ — are precisely what a properly weighted, geometry-aware
affinity graph with an adaptive bandwidth would cure, but that is a later method; here I want the
honest floor that neighbor averaging alone reaches.

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
