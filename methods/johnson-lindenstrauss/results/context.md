# Context: low-distortion dimensionality reduction for finite point sets

## Research question

Given $n$ points $v_1,\dots,v_n$ in a Euclidean space $\mathbb{R}^d$ with $d$ very large, we want a map $f:\mathbb{R}^d\to\mathbb{R}^k$ into a much smaller space that preserves **all** pairwise Euclidean distances up to a small multiplicative error: for a target accuracy $0<\varepsilon<1$,

$$(1-\varepsilon)\,\|v_i-v_j\|^2 \;\le\; \|f(v_i)-f(v_j)\|^2 \;\le\; (1+\varepsilon)\,\|v_i-v_j\|^2 \qquad \text{for every pair } i,j.$$

The prize is a target dimension $k$ that is small and — crucially — **independent of the ambient dimension $d$**, depending only on the number of points $n$ and the tolerance $\varepsilon$. If $k$ could be made to grow like $\log n / \varepsilon^2$, then a great many geometric algorithms whose cost scales badly with dimension could be run in the compressed space and have their answers transfer back with a controlled distortion.

The pain point is the **simultaneity over all $\binom{n}{2}$ pairs**. Preserving the length of one fixed vector after projection is easy on average; the difficulty is that a single map must keep *every* one of the $\binom{n}{2}$ inter-point distances inside the $(1\pm\varepsilon)$ window at once, and these events are coupled through the shared map. A satisfactory answer must (i) exhibit a concrete, efficiently computable $f$, (ii) prove the all-pairs guarantee holds, and (iii) keep $k$ free of $d$ — so the compression does not silently smuggle the ambient dimension back in.

## Background

The setting is finite-metric / low-distortion embeddings, whose algorithmic uses were crystallized by Linial, London and Rabinovich (*Combinatorica* 1995): embedding a point set into a low-dimensional normed space with distances approximately preserved lets one run dimension-sensitive algorithms in the small space and pull the answer back. The natural targets are problems whose running time depends exponentially (or at least polynomially) on the working dimension — approximate nearest neighbor (Indyk–Motwani, STOC 1998), clustering to minimize intra-cluster squared distances (Schulman), low-rank approximation (Papadimitriou–Raghavan–Tamaki–Vempala), learning mixtures of Gaussians (Dasgupta 1999), and data-stream computation (Indyk 2000). In all of these, the *only* thing one needs from the embedding is that pairwise distances survive; the geometry beyond pairwise distances is irrelevant.

The load-bearing probabilistic facts are classical. For a standard Gaussian $X\sim N(0,1)$ the moment generating function of $X^2$ is $\mathbb{E}[e^{sX^2}] = (1-2s)^{-1/2}$ for $s<\tfrac12$; this is the engine behind every tail bound on sums of squared Gaussians. A sum of $k$ independent squared standard Gaussians is a $\chi^2_k$ variable, sharply concentrated about its mean $k$. The Gaussian is **2-stable**: for fixed reals $\alpha_1,\dots,\alpha_d$ and independent $Z_i\sim N(0,1)$, the linear combination $\sum_i \alpha_i Z_i$ is again $N(0,\|\alpha\|^2)$ — so an inner product of a fixed vector with a Gaussian vector is a one-dimensional Gaussian whose variance is the squared length of the fixed vector. A vector of $d$ i.i.d. $N(0,1)$ entries, normalized, is uniform on the sphere $S^{d-1}$, which lets "project a fixed vector onto a random subspace" be exchanged for "project a random unit vector onto a fixed subspace."

Two diagnostic facts about *existing* approaches set up the problem. First, the obvious data-dependent reduction — keep the top-$k$ principal directions (SVD/PCA) — minimizes a *global* error: the best rank-$k$ approximation $A_k$ satisfies $\|A-A_k\|\le\|A-D\|$ for every rank-$k$ matrix $D$ in any unitarily invariant (e.g. Frobenius) norm, so it minimizes the *sum* of squared displacements. But a global optimum offers **no local floor**: the distance between a particular pair of points can be made arbitrarily small relative to its original value if that helps the total, so SVD gives no per-pair $(1\pm\varepsilon)$ guarantee. Second, any data-dependent top-$k$ subspace must, in the worst case, let $k$ grow with the intrinsic spread of the data and hence cannot be promised independent of $d$ a priori. So both stall on the same two failures: the optimization is global rather than per-pair, and the dimension it needs is tied to the data.

## Baselines

**Singular value decomposition / PCA projection.** Embed the $n\times d$ data matrix $A$ into $\mathbb{R}^k$ by projecting onto the span of the top-$k$ singular vectors, producing the rank-$k$ matrix $A_k$. Core guarantee: $\|A-A_k\|\le\|A-D\|$ for all rank-$k$ $D$ under any unitarily invariant norm, so under Frobenius the total squared displacement $\sum_i \|(\text{row }i)-(\text{projected row }i)\|^2$ is minimized. Gap: this is a global criterion with no local control — an individual pairwise distance can collapse arbitrarily; the method gives no all-pairs $(1\pm\varepsilon)$ guarantee, and the required $k$ is dictated by the data's spectrum, not by $n,\varepsilon$ alone.

**Random projection onto a uniformly random $k$-dimensional subspace (sharpened by Frankl–Maehara, *J. Combin. Theory Ser. B* 1988).** Pick a uniformly random $k$-dimensional subspace and project. By rotational symmetry the squared length of a fixed unit vector's projection is sharply concentrated about $k/d$; preserving distances for all pairs then follows by a union bound over the $\binom{n}{2}$ events. Frankl–Maehara gave the explicit sufficiency $k \ge 9(\varepsilon^2 - 2\varepsilon^3/3)^{-1}\ln n + 1$. Gap: constructing and applying a uniformly random orthonormal $k$-frame requires orthogonalizing $k$ vectors in $\mathbb{R}^d$ — conceptually clean but heavier than necessary, and awkward to implement, especially in systems (e.g. databases) restricted to simple aggregate operations.

**Random Gaussian matrix without orthonormalization (Indyk–Motwani, STOC 1998).** Instead of a true random subspace, draw $k$ independent vectors $U_1,\dots,U_k$ with i.i.d. $N(0,1)$ coordinates and set the $i$-th coordinate of $f(x)$ to $\tfrac{1}{\sqrt k}\langle U_i,x\rangle$; equivalently, fill a $k\times d$ matrix with i.i.d. $N(0,1/k)$ entries and multiply by it. For a fixed $x$, each unscaled inner product is exactly $N(0,\|x\|^2)$, so the projected squared norm is a scaled $\chi^2_k$ variable without any orthogonalization. Gap: the entries are real Gaussians requiring floating-point sampling and dense multiplication, and the construction still leaves open whether simpler integer or sparse entries can give the same tail behavior.

**Search/construction-only embeddings.** One can also hunt directly for a good projection matrix or a good low-dimensional placement by optimizing a distortion objective over many random or perturbed candidates. Such a search can return a matrix that happens to work on the given inputs, but it produces no distribution-level guarantee, no proof that an arbitrary point set is handled, no $d$-independent bound on $k$, and no certificate — it optimizes one instance rather than establishing the property for all inputs. It is the instance-tuned counterpart to the principled constructions above and closes none of their gaps.

## Evaluation settings

The natural yardsticks are (i) the **target dimension** $k$ as a function of $n$ and $\varepsilon$ — the smaller and the more clearly $d$-independent, the better; (ii) the **distortion** actually achieved on all $\binom{n}{2}$ pairs, measured by the empirical ratio $\|f(v_i)-f(v_j)\|^2 / \|v_i-v_j\|^2$ and whether it lands in $[1-\varepsilon,1+\varepsilon]$; (iii) the **success probability** of a single random draw and how it degrades or boosts under repetition; and (iv) the **construction/application cost** of $f$ — sampling the matrix, the arithmetic of applying it (dense floating-point multiply vs. integer/sparse aggregation), and the number of random bits consumed. The standard regime of interest is $0<\varepsilon<1$ with $n$ large and $d\gg k$; high-dimensional point sets (sparse text/term–document vectors, dense feature matrices) are the typical inputs, and the metric is always the Euclidean ($\ell_2$) distance.

## Code framework

Existing primitives: a numerical array/linear-algebra library (`numpy`), sparse matrices (`scipy.sparse`), Gaussian and Bernoulli samplers, random subset sampling without replacement, and dense/sparse matrix products. The base objects are the source dimension $d$ (`n_features`), the number of points $n$ (`n_samples`), the tolerance $\varepsilon$ (`eps`), and a target dimension $k$ (`n_components`). The harness samples a $k\times d$ projection matrix and applies it to the data $X\in\mathbb{R}^{n\times d}$ as $X R^\top$.

```python
import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.random import sample_without_replacement

def min_embedding_dim(n_samples, *, eps=0.1):
    # TODO: choose k from n_samples and eps.
    pass

def _check_density(density, n_features):
    # TODO: decide how sparse the projection matrix may be.
    pass

def _check_input_size(n_components, n_features):
    pass

def _dense_random_matrix(n_components, n_features, random_state=None):
    # TODO: choose and scale a dense entry distribution.
    pass

def _sparse_random_matrix(n_components, n_features, density="auto", random_state=None):
    # TODO: choose and scale a sparse entry distribution.
    pass

class RandomProjection:
    def __init__(self, n_components="auto", eps=0.1, kind="gaussian",
                 density="auto", dense_output=False, random_state=None):
        self.n_components = n_components
        self.eps = eps
        self.kind = kind
        self.density = density
        self.dense_output = dense_output
        self.random_state = random_state

    def fit(self, X):
        n_samples, n_features = X.shape
        random_state = check_random_state(self.random_state)
        if self.n_components == "auto":
            k = min_embedding_dim(n_samples=n_samples, eps=self.eps)
        else:
            k = self.n_components
        self.n_components_ = int(k)
        if self.kind == "gaussian":
            self.components_ = _dense_random_matrix(
                self.n_components_, n_features, random_state=random_state
            )
        elif self.kind == "sparse":
            self.density_ = _check_density(self.density, n_features)
            self.components_ = _sparse_random_matrix(
                self.n_components_, n_features, density=self.density_,
                random_state=random_state
            )
        else:
            pass
        return self

    def transform(self, X):
        return safe_sparse_dot(X, self.components_.T, dense_output=self.dense_output)
```
