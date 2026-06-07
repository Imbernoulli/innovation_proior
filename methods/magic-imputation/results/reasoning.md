OK, let me start from what's actually broken, not from any operator I'd like to write down. I have a single-cell count matrix: rows are cells, columns are genes, entries are captured mRNA counts. The thing that hurts is that the assay only sees a small random slice of each cell — something like a tenth of the transcriptome makes it through capture and sequencing. So a gene that is genuinely on, but not screaming, frequently comes back as a flat zero in a given cell. The matrix is mostly zeros, and the nonzeros are small and noisy. Dropout, people call it.

Let me make myself feel why this is fatal and not just cosmetic. Take two genes that, in clean biology, rise and fall together as cells move through some process — pick a continuous trend, y roughly proportional to x along the trajectory. Now thin both coordinates by independent random sampling. In a lot of cells, gene x survives but gene y drops to zero, or vice versa. So instead of a tidy diagonal trend I get a scatter where huge numbers of points are pinned to one axis: (x, 0) and (0, y) everywhere. The correlation I'd compute is dragged toward zero. A smooth trajectory looks like a discontinuous blob. The signal isn't gone — it's shredded by per-cell missingness. That's the enemy.

Now, the key asymmetry I should exploit: the dropout is per cell, and roughly independent from cell to cell. Where this cell lost gene y, some other cell probably kept it. So the information I need to fill cell i's value for gene y is *physically present elsewhere in the matrix* — I just need to know which other cells are "the same cell" as i, biologically, so I can borrow from them. That reframes the whole thing: imputation is an averaging problem, and the entire difficulty is defining the right neighborhood to average over.

So which cells do I average cell i with? The dumb answer is "cells with similar expression vectors, by Euclidean distance." But that's circular and broken: the distances are computed in the same dropout-corrupted space. Two cells that are truly identical can look far apart because they dropped out different genes. I can't trust raw distances directly.

Here's the structural fact that saves me. Cells don't live freely in 20,000-gene space. Expression is regulated; genes are coupled; so the set of states a real cell can occupy is a low-dimensional, curved surface inside that huge space — a manifold. Cells in a process slide along it continuously. If I can get a handle on this manifold, then "biologically the same cell" means "nearby on the manifold," and that's a much more robust notion than raw distance, because the manifold is a global object — its local pieces constrain each other. The standard handle is a graph: a node per cell, edges to each cell's most-similar neighbors. Local distances are noisy, sure, but the graph stitches local pieces into a coherent geometry, and walks on the graph will let me average along the manifold rather than straight across the ambient space.

Let me first clean up the inputs so that even local distances are as honest as I can make them, because everything downstream rides on the graph being right. Two cells can differ just because one was sequenced more deeply — more total counts — which is a pure technical confound, nothing biological. Divide each cell by its total count (and multiply back by the median total, to keep numbers in a sane range): library-size normalization. Next, the counts are roughly Poisson-ish — variance grows with the mean — so a highly expressed gene dominates Euclidean distance purely through its scale, and the noise is heteroscedastic. A square-root transform stabilizes the variance of count data, flattening that out, so distances weigh genes more evenly. (A log would do something similar but mishandles the zeros; sqrt is gentle and defined at zero.) Now distances in the transformed space mean more.

One more preconditioning move. Even after normalizing and sqrt-ing, the per-gene dropout noise is still sitting in all 20,000 coordinates, and Euclidean distance sums it over every one of them — death by a thousand noisy dimensions. But the manifold is low-dimensional, so most of those coordinates are redundant. Project onto the top principal components — keep enough to retain most of the variance, a few dozen to a hundred dims. The leading PCs capture the coordinated, high-variance structure (the biology); the tail PCs are mostly the independent per-gene noise I want to drop. Computing neighbor distances in PC space gives the diffusion a cleaner geometry to start from, and it's much cheaper. I'll compute cell-cell Euclidean distances here, in PCA space.

Now turn distances into a graph with weights. I want similarity, not distance: nearby cells strongly connected, far cells essentially disconnected, with a smooth falloff. The classic choice is a Gaussian kernel, A(i,j) = exp(-(d(i,j)/sigma)^2). Smooth, positive, decays fast — beyond a few sigma it's numerically zero, which is what I want (I don't want a cell talking to the whole dataset).

But what is sigma? If I pick one global sigma, I'm in trouble, and I can see exactly why by thinking about density. Single-cell data is wildly non-uniform on the manifold: common cell types are sampled densely, rare states sparsely. With a single sigma tuned for the dense regions, the sparse cells have no neighbors inside one sigma — they fall off the graph, disconnected, and get no imputation. Tune sigma for the sparse regions and the dense cells each connect to hundreds of others, over-smoothing them. There's no global sigma that's right everywhere because the right scale *is local*. So make sigma adaptive, per cell: set cell i's bandwidth to its own distance to its k-a-th nearest neighbor. sigma_i = d(i, neighbor_{ka}(i)). Now, by construction, every cell has roughly the same number of meaningfully-weighted neighbors regardless of how dense its region is — the kernel auto-scales to local density. Dense cell: small sigma_i, tight kernel. Sparse cell: large sigma_i, reaches out far enough to stay connected. So A(i,j) = exp(-(d(i,j)/sigma_i)^2).

How small should ka be, and do I need every cell connected to every other? ka controls locality: a small ka keeps the kernel tight and the geometry faithful to the manifold's curvature — but if I make it too small the graph can fragment into disconnected pieces and imputation can't cross the gaps. So the rule is: ka as small as I can get away with while the graph stays connected. And I certainly don't want a dense N x N matrix — at a few sigma the Gaussian is already negligible, so I truncate to at most k nonzero neighbors per cell. Since sigma is set at the ka-th neighbor, going out to about k = 3*ka catches essentially all the non-negligible weight (3 sigma) and zeros the rest, keeping A sparse and the whole thing fast.

Let me pause on the exponent. I wrote (d/sigma)^2 — the Gaussian. But the "2" is really a knob: a general kernel is exp(-(d/sigma)^alpha). The exponent controls how heavy the tails are. With alpha=2 (Gaussian) the tails fall off very fast — beyond a couple sigma, dead. That's fine when sampling is even, but with a heavy-tailed neighbor-distance distribution the Gaussian can still leave borderline cells under-connected; a smaller alpha (say 1, an exponential-style decay) keeps a softer tail, connecting cells a bit further out and making the graph more robustly connected across density gaps. So I'll keep alpha as a decay parameter and let it default low; the Gaussian is just the alpha=2 special case. A(i,j) = exp(-(d(i,j)/sigma_i)^alpha).

Now A as I've built it is not symmetric. sigma_i is i's bandwidth, sigma_j is j's, and they differ, so A(i,j) (using i's bandwidth) is not A(j,i) (using j's). That's a problem: I want a well-behaved averaging operator, and the clean spectral theory I'm about to lean on wants a symmetric affinity. Symmetrize additively: A <- A + A^T (equivalently average the two; up to a global factor of 2 it's the same thing, and that factor washes out at the next step). Additively combining the two directions also does something nice for outliers — a cell that nobody picks as a neighbor but which picks others still gets edges from the transpose, so it isn't orphaned; the symmetrization pulls outliers back in and smooths the weights.

So I have a symmetric affinity A. Now I want to average. The first instinct is just: replace each cell by the A-weighted mean of its neighbors. To make that an honest average — a convex combination that stays inside the data and doesn't rescale magnitudes by accident — I should make the rows sum to one. Divide each row by its total affinity: M(i,j) = A(i,j) / sum_k A(i,k). Now M is row-stochastic, every row a probability distribution over neighbors. And I recognize this object — it's a Markov transition matrix. M(i,j) is the probability that a one-step random walk from cell i steps to cell j: more likely to land on a strong (close) neighbor, essentially never on a far cell. The imputation "average cell i over its neighbors" is exactly one step of this walk applied to the data: (M X)(i, :) = sum_j M(i,j) X(j, :), the transition-weighted mean of neighbors' expression. Each gene in each cell becomes a weighted average over its graph neighborhood, and dropped zeros get filled by neighbors that kept the transcript.

Is one step enough? Let me stress-test it. One step trusts M's edges literally. But M still has noisy edges — the graph was built from noisy distances, so some "neighbor" links are spurious shortcuts created by a lucky alignment of dropouts between two cells that aren't really close on the manifold. A single hop averages across those bad edges just as much as the good ones. I want a way to trust an edge in proportion to how *robustly* connected the two cells are — connected through many independent paths along the manifold, not through one fragile shortcut.

That's what running the walk longer does. Take two steps: M^2(i,j) = sum_k M(i,k) M(k,j) sums probability over all length-2 paths from i to j. A pair that's truly close on the manifold is linked by many short paths through their shared neighbors, so their multi-step transition probability accumulates and grows; a spurious one-edge shortcut between two genuinely distant cells contributes a single thin path and gets diluted as probability leaks out to the true neighbors at each hop. So powering M reinforces manifold-consistent connections and washes out noise-induced ones. Run the walk for t steps and average the data with it: X_imputed = M^t X. This is data diffusion — let the expression values diffuse over the cell graph for t steps, each cell relaxing toward its neighborhood's consensus, denoising while respecting the manifold because the walk only ever moves along graph edges.

I should make sure I understand *why* powering is the right kind of smoothing, not just hand-wave "many paths." Look at the spectrum. M is row-stochastic; its symmetric conjugate D^{-1/2} A D^{-1/2} is a symmetric matrix similar to M, so M has real eigenvalues, and for a stochastic matrix they live in [0,1]. The top eigenvalue is 1, with the constant vector as its eigenmode — that's the stationary distribution, the global average, the thing the walk converges to if you run it forever. The other eigenvectors are functions on the graph ordered by eigenvalue: the ones with lambda near 1 are smooth, slowly-varying functions along the manifold (the real biological gradients), and the ones with lambda near 0 are high-frequency, rapidly oscillating functions — that's where the noise lives, it has no smooth manifold structure. Now write X in this eigenbasis. M^t acts on the component with eigenvalue lambda by multiplying it by lambda^t. So every mode with lambda<1 gets shrunk, and shrunk *harder the smaller lambda is*: lambda=0.1 after t=5 is 1e-5, gone; lambda=0.95 after t=5 is still 0.77, mostly kept. M^t is a low-pass filter on the graph — it preserves the slow biological gradients and annihilates the fast noise modes, with t setting the cutoff. That's the spectral reason diffusion denoises. (And it's the discrete face of heat diffusion on the manifold, the heat kernel e^{-t Delta} — powering the transition matrix is running the heat equation, which is the canonical smoother.)

That same spectral picture immediately warns me about t. If I keep powering, every lambda<1 eventually hits zero and only the lambda=1 constant mode survives — meaning M^t X collapses every cell toward the *single global mean* of the dataset. Over-diffusion erases all the biology, melts distinct cell states into mush. So t is the crucial dial, and it's a real tension: too small and I haven't removed the noise (under-smoothed, dropout still showing); too large and I've washed out the signal too (over-smoothed, everything converges to the stationary blob). There's a sweet spot in between, and I shouldn't just hard-code it because the right t depends on how noisy and how curved each dataset is.

How do I find it without ground truth? Watch how fast the data is still changing as I add diffusion steps. Early on, each extra step removes a lot of noise — X_imputed changes a lot from t to t+1, the high-frequency modes are dying fast. That's the imputation regime, where each step is doing real denoising work. Once the noise is gone, further steps only nibble at the slow modes, gently sliding everything toward the global mean — X_imputed barely changes per step, and what change there is, is the start of over-smoothing. So the rate of change drops sharply and then plateaus, and the elbow between the two regimes is where I want to stop: noise removed, signal not yet bled out.

So I need to quantify "how much did the data change from step t-1 to step t" and stop when it's small. One concrete measure: the coefficient of determination R^2 between the imputed matrix at t and at t-1, R^2 = 1 - SSE/SST — it's normalized to [0,1], 1 meaning "identical to the previous step." When the per-step change has fallen below a few percent (R^2 says <5% change from the previous t), the decay has essentially converged; to be safe against a single noisy reading I take the second t past that threshold as my stopping point. (Equivalently I can align successive iterates and measure their disparity — a Procrustes distance between data_t and data_{t-1} — and stop when that drops under a small threshold; same idea, "stop when the iterate stops moving.") If the data already comes in fairly clean, a small fixed t — around 3 — is a fine default without the auto-search.

There's a last, easy-to-miss point about magnitudes. The diffusion operator is an *averaging* operator, rows summing to one, so M^t X stays in the convex hull of the data — it won't blow up, good — but averaging shrinks peaks: a gene that's high in a few cells gets pulled down toward its neighbors, so the imputed dynamic range per gene is compressed relative to the original. If I care about restoring the original expression *scale* (not just the relative pattern across cells), I can rescale each gene back: set the max of the imputed gene to, say, the 99th percentile of the original gene's values (99th rather than the literal max, to ignore a single noisy outlier count), scaling the whole imputed column by that ratio. That restores per-gene magnitude while keeping the now-smooth relative structure across cells. It's an optional cosmetic step on top — the denoising is done by the diffusion.

Let me also sanity-check the failure mode I was worried about — collapsing distinct populations. The graph is local: if two cell types are genuinely separated on the manifold, there are few or no edges between them, M^t keeps probability essentially within each population for moderate t, and diffusion smooths *within* a population without bleeding across the gap. The bleeding only happens at large t when the walk has had time to leak across whatever thin bridges exist — which is exactly the over-smoothing regime my t-selection is built to stop short of. Good, the pieces are consistent.

So the whole method falls out as a chain. Sparse, dropout-ridden counts, but cells live on a manifold, so a cell's true expression is redundantly present in its graph neighbors. Normalize and sqrt and PCA so distances are honest; build a kNN graph with an adaptive Gaussian-style affinity so every cell — dense or rare — gets a sensible, density-matched neighborhood; symmetrize; row-normalize to a Markov transition matrix M; then diffuse, X_imputed = M^t X, which spectrally is a low-pass filter that keeps the slow manifold gradients and kills the fast noise modes; pick t at the elbow where per-step change plateaus, before over-smoothing collapses everything to the global mean; optionally rescale per-gene magnitude back. Let me write it. The kernel/Markov machinery and the powering are the load-bearing parts, so I'll lean on a graph builder for the affinity-to-operator construction and implement the diffusion explicitly.

```python
import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize   # l1 row-normalize -> stochastic
import graphtools                              # builds the affinity + diffusion operator

# ---- preprocessing: make distances honest (context scaffold's primitives) ----
def preprocess(counts):
    counts = counts[counts.sum(1) > 0]                  # drop empty cells
    counts = counts[:, np.asarray(counts.sum(0)).ravel() > 0]  # drop dead genes
    libsize = counts.sum(1, keepdims=True)
    counts = counts / libsize * np.median(libsize)      # library-size normalize: kill depth confound
    return np.sqrt(counts)                               # variance-stabilize count noise

class MAGIC:
    # knn: neighbor index whose distance sets the per-cell bandwidth (the "ka")
    # knn_max = 3*knn: hard cap on nonzero neighbors (~3 sigma of the kernel)
    # decay: kernel exponent alpha in exp(-(d/sigma)^alpha); Gaussian is alpha=2
    # t: diffusion steps; 'auto' finds the elbow
    def __init__(self, knn=5, knn_max=None, decay=1, t=3, n_pca=100):
        self.knn, self.decay, self.t, self.n_pca = knn, decay, t, n_pca
        self.knn_max = knn_max if knn_max is not None else 3 * knn

    def fit(self, X):
        # SLOT 1: PCA -> kNN distances -> adaptive affinity -> symmetrize ->
        #         row-stochastic Markov operator M = D^{-1} A.
        # graphtools does exactly this: per-cell bandwidth = distance to the
        # knn-th neighbor; A(i,j) = exp(-(d/sigma_i)^decay); additive
        # symmetrization A<-(A+A^T)/2; diff_op = normalize(A, 'l1', axis=1).
        self.graph = graphtools.Graph(
            X, n_pca=self.n_pca, knn=self.knn, knn_max=self.knn_max,
            decay=self.decay, thresh=1e-4)
        self.diff_op = self.graph.diff_op            # the Markov transition matrix M
        return self

    def _select_t(self, X, t_max=20, thresh=1e-3):
        # walk one step at a time; stop when the iterate stops moving
        # (per-step change has plateaued -> end of the noise-removal regime,
        #  before over-smoothing drags everything to the global mean)
        Xt = np.asarray(X, dtype=float)
        prev = Xt.copy()
        for i in range(1, t_max + 1):
            Xt = self.diff_op.dot(Xt)
            _, _, disparity = spatial.procrustes(prev, Xt)  # change from t-1 to t
            prev = Xt.copy()
            if disparity < thresh:
                return Xt
        return Xt

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.t == "auto":
            X_imputed = self._select_t(X)               # elbow search
        else:
            Mt = np.linalg.matrix_power(self.diff_op.toarray(), self.t)
            X_imputed = Mt.dot(X)                        # X_imputed = M^t X : data diffusion
        return X_imputed

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# optional: restore per-gene magnitude lost to averaging
def rescale(X_imputed, X_original, pct=99):
    p = np.percentile(X_original, pct, axis=0)
    m = X_imputed.max(axis=0)
    scale = np.divide(p, m, out=np.zeros_like(p), where=m > 0)
    return X_imputed * scale
```

Tracing the causal chain once more: counts are sparse because the assay subsamples each cell, so genes drop out — but cells lie on a low-dimensional manifold, so a cell's true expression is shared, redundantly, with its graph neighbors; build that graph with a density-adaptive affinity kernel and row-normalize it to a Markov operator M; diffusing the data, M^t X, averages each cell over its t-step neighborhood, which spectrally is a low-pass filter that keeps the slow biological gradients and erases the fast dropout noise; stop t at the elbow before the walk collapses everything to the global mean; and the zeros get filled by neighbors that kept the transcript.
