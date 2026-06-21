# Context: denoising and recovering missing transcripts in single-cell RNA-seq count matrices

## Research question

Single-cell RNA sequencing measures, for each of tens of thousands of cells, the number of captured mRNA molecules per gene — a cells x genes count matrix. The measurement is brutally undersampled: per cell, only a small random fraction of the transcriptome is actually captured and sequenced (on the order of 5-15%). For a gene expressed at low or moderate level, the captured count is frequently zero even though the gene is on — a missing value known as "dropout." The matrix is therefore extremely sparse, with a large majority of entries zero, and the nonzero entries are themselves noisy small counts.

Pairwise gene-gene relationships, which in clean data trace out smooth regulatory trends as cells move through a biological process, are compressed into a cloud of points hugging the axes when one of two co-varying genes drops out in a cell. Correlations are attenuated toward zero, continuous trajectories look discontinuous, and rare intermediate states are buried in noise. The question is how to recover, for each cell and gene, an estimate of the underlying expression from the observed sparse count matrix.

## Background

The load-bearing empirical fact is the sampling regime itself. Capture-and-amplification chemistry recovers only a small random sample of each cell's transcriptome, so observed counts are a thinned, noisy version of the truth; lowly expressed genes are recorded as zero far more often than they are truly absent. This is the measured behavior of the assay.

The second load-bearing idea is the manifold assumption. Although a cell is described by a vector in a space of ~20,000 genes, cells are not free to occupy that space arbitrarily. Gene expression is regulated, and regulatory programs couple genes together, so the set of biologically realizable cell states forms a low-dimensional, often curved, surface — a manifold — sitting inside the high-dimensional measurement space. Cells undergoing a continuous process (differentiation, a transition, a response) trace out connected paths on this manifold. The practical handle on the manifold is a nearest-neighbor graph: put a node at every cell, connect each cell to the cells most similar to it in expression, and the graph approximates the manifold's local geometry. Two cells that are neighbors on this graph share, up to biological variation, the same underlying expression profile — so a cell's true value for a gene is information that is also present, redundantly, in its neighbors. Dropout is per-cell and roughly independent across cells, so where one cell lost a transcript its neighbors likely retained it.

Several mathematical tools from manifold learning sit ready to be used. A similarity (affinity) between two cells is obtained from their distance with a kernel — classically a Gaussian, exp(-d^2/sigma^2), which decays smoothly so that only nearby cells interact. A single global bandwidth sigma behaves differently when sample density varies across the manifold, which it does in single-cell data (common cell types are densely sampled, rare states sparsely), motivating an adaptive, per-point bandwidth tied to the local neighbor distances. Row-normalizing an affinity matrix turns it into a Markov transition matrix M, where M(i,j) is the probability that a one-step random walk starting at cell i lands at cell j; M is row-stochastic. This is exactly the object studied in diffusion-based geometry.

Coifman and Lafon's diffusion-map framework (2006) is the directly relevant prior theory. They build M from a kernel on the data, observe that its (symmetric conjugate) eigenvalues lie in [0,1] with a trivial top eigenvalue 1 attached to the constant/stationary mode, and study powers M^t. Running the walk for t steps composes transition probabilities; in the eigenbasis, M^t has eigenvalues lambda^t. The slow modes (lambda near 1) are smooth, low-frequency functions along the manifold; the fast modes (lambda near 0) are high-frequency, oscillatory. The family of diffusion distances ||M^t e_i - M^t e_j|| measures how connected two points are through all paths of length t — a multiscale geometry parameterized by t. Powering a stochastic matrix is, in the continuous limit, applying the heat kernel e^{-t Delta} on the manifold.

## Baselines

The naive baseline is to do nothing and analyze raw counts.

A second baseline is local averaging without a manifold: smooth each cell against a fixed Euclidean neighborhood, or against cells in the same cluster. Clustering-then-averaging assigns cells to discrete groups and replaces each cell by its group mean. A fixed-radius Euclidean average uses a common radius for all cells.

A third family is model-based: posit a generative model of counts (e.g. a zero-inflated or negative-binomial likelihood per gene) and impute by fitting it. These commit to a parametric noise model and to a notion of which zeros are "true" versus "dropped," and they typically operate gene-by-gene or with low-rank factor structure.

Diffusion maps and related spectral embeddings (Laplacian eigenmaps) are established tools for *representing* the manifold — they produce low-dimensional coordinates from the data for dimensionality reduction and visualization.

## Evaluation settings

The natural data are scRNA-seq count matrices from droplet and plate-based platforms: continuous biological processes where smooth gene-gene structure is expected — an epithelial-to-mesenchymal transition (markers such as the epithelial CDH1 and mesenchymal VIM, ZEB1), hematopoietic differentiation in bone marrow, and developmental time courses. Inputs are large (thousands to tens of thousands of cells, ~20,000 genes), sparse, integer-valued. Standard preprocessing primitives that already exist: filtering out empty cells and unexpressed genes, library-size normalization (dividing each cell by its total count to remove sequencing-depth confounding), a variance-stabilizing transform such as square root or log on the counts, and PCA to a few-dozen-to-hundred dimensions. The yardsticks are qualitative recovery of known gene-gene relationships and continuous trajectories, the restoration of correlation structure attenuated by dropout, and — where a ground truth exists, such as bulk measurements or held-out/artificially-masked entries — agreement of recovered values with the held-out truth.

## Code framework

Pre-existing primitives: a sparse count-matrix loader, the standard preprocessing functions, a PCA, a nearest-neighbor search, and a sparse row-normalizer. The recovery method itself is a single empty operator with a fit (build whatever structure the data needs) and a transform (produce denoised values) interface.

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize   # row-normalize a nonneg matrix to stochastic

# --- already available: preprocessing primitives ---
def filter_empty(counts):
    counts = counts[counts.sum(axis=1) > 0]        # drop empty cells
    counts = counts[:, np.asarray(counts.sum(axis=0)).ravel() > 0]  # drop unexpressed genes
    return counts

def library_size_normalize(counts):
    libsize = counts.sum(axis=1, keepdims=True)
    return counts / libsize * np.median(libsize)   # remove depth confound

def sqrt_transform(x):
    return np.sqrt(x)                               # variance-stabilize counts

# --- the recovery operator: empty slots the method will fill ---
class Recover:
    def __init__(self, knn=5, t=3, n_pca=100):
        self.knn, self.t, self.n_pca = knn, t, n_pca

    def fit(self, X):
        # SLOT 1: reduce dimension, find neighbors, turn distances into
        #         cell-cell affinities, and build the averaging operator
        #         that the manifold geometry implies.
        # TODO
        return self

    def transform(self, X):
        # SLOT 2: use the operator from fit() to produce denoised /
        #         recovered expression values for the requested genes.
        # TODO
        pass

    def fit_transform(self, X):
        return self.fit(X).transform(X)
```
