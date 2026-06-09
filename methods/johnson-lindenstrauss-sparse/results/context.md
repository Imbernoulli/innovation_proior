# Context

## Research question

A unit vector $x \in \mathbb{R}^d$ lives in a space whose dimension $d$ is the bottleneck of almost every downstream algorithm: nearest-neighbor search, closest pair, clustering, linear regression, low-rank approximation. The Johnson–Lindenstrauss lemma says we can collapse $d$ down to $k = \Theta(\varepsilon^{-2}\log(1/\delta))$ while preserving every Euclidean norm (hence every pairwise distance) to within $1\pm\varepsilon$ with probability $1-\delta$: there is a distribution over $k\times d$ matrices $S$ with
$$\Pr_S\!\left[(1-\varepsilon)\|x\|_2 \le \|Sx\|_2 \le (1+\varepsilon)\|x\|_2\right] > 1-\delta,$$
and a union bound over $\binom{n}{2}$ difference vectors turns this into the flattening theorem: any $n$ points embed into $O(\varepsilon^{-2}\log n)$ dimensions. The value $k = \Theta(\varepsilon^{-2}\log(1/\delta))$ is known to be optimal.

The question is not *whether* the embedding exists — it does — but *how fast it can be applied*. The standard proof draws $S$ dense (i.i.d. Gaussian or $\pm 1$ entries), so computing $Sx$ costs $\Theta(k\cdot\|x\|_0)$, i.e. $\Theta(kd)$ for a dense vector. In the pipelines where JL is used, this matrix-vector product is the dominant cost — the dimension reduction "overwhelms the running time" of the search algorithm it is supposed to accelerate. Two regimes sharpen the pain:
1. **Dense $x$.** Can we apply the embedding in $o(kd)$, ideally $O(d\log d)$, time, beating the dense product?
2. **Sparse $x$ and streaming.** When $\|x\|_0 \ll d$ (bag-of-words documents, IP-pair traffic counts, sparse rating matrices), or when $x$ receives coordinate updates $x \leftarrow x + v\,e_i$ in a stream, can the embedding cost scale with $\|x\|_0$ rather than $d$? A dense $S$ already does $O(k\cdot\|x\|_0)$, so any "fast" transform that costs $\Omega(d\log d)$ even for $\|e_i\|_0=1$ is a regression for this case.

A solution must keep the optimal $k=\Theta(\varepsilon^{-2}\log(1/\delta))$ rows, keep an $\ell_2\to\ell_2$ (or $\ell_2\to\ell_p$) embedding — not a median-based sketch — and cut the per-vector cost well below $kd$.

## Background

**The dense projection and its proofs.** The original construction projects onto a uniformly random $k$-dimensional subspace; equivalently, after the usual global rescaling, one can take $S$ with i.i.d. $N(0,1/k)$ entries, dropping the orthogonality of rows and keeping normality/sphericity only in expectation. The proof reduces to showing that for a unit $x$, $\|Sx\|_2^2$ concentrates around its mean: each coordinate $(S_i\cdot x)$ is Gaussian, $\|Sx\|_2^2$ is a scaled $\chi^2$ variable with $k$ degrees of freedom, and a Chernoff/MGF bound gives the $1-\delta$ tail at $k=\Theta(\varepsilon^{-2}\log(1/\delta))$. Achlioptas observed the only property the proof needs is that each projected coordinate concentrates with the right variance, which holds for rescaled Rademacher entries too — flipping coins instead of sampling Gaussians.

**Two strands toward speed.** Reducing the embedding time split into two intertwined lines:
- *Sparsity.* Make $S$ have few nonzeros per column; if each column has $s$ nonzeros, $Sx$ costs $\Theta(s\cdot\|x\|_0)$.
- *Structure / Fourier.* Replace the unstructured dense product with a transform that admits an FFT, computable in $O(d\log d)$ regardless of structure.

**The obstacle to sparsity: sparse matrices distort sparse vectors.** A sparse $S$ cannot preserve the norm of a spiky input. The extreme case is $x=e_i$: the estimator $\|Sx\|_p$ only sees the entries of column $i$, and if that column is mostly zero the estimate has variance too high to concentrate. This is why a constant-factor sparsification is the most one can do *directly*. Formally (Matoušek): for a sparse Rademacher construction to preserve the norm of a unit vector $x$, one needs $\|x\|_\infty$ small — the vector must be "spread out." The relevant niceness parameter is the ratio $\nu = \|x\|_\infty/\|x\|_2 \in [d^{-1/2}, 1]$; spiky vectors ($\nu$ near $1$) are the hard instances.

**The uncertainty principle.** A signal and its Fourier spectrum cannot both be concentrated (the Heisenberg principle of harmonic analysis). The Walsh–Hadamard transform $H$ — the DFT over the additive group $\mathrm{GF}(2)^d$, with entries $H_{ij}=d^{-1/2}(-1)^{\langle i-1,\,j-1\rangle}$ — is real-valued, orthogonal, and admits a recursive FFT in $O(d\log d)$ time. As an orthogonal map it preserves $\ell_2$ norms exactly.

**Hashing-based sparse constructions.** Encoding $S$ via a hash function lets each source coordinate be sent to one (or a few) target coordinate(s) with a random sign, giving a fixed number of nonzeros per column without storing a dense matrix. The contribution of each target coordinate is a sum of signed source values; the error comes from *collisions* — two source coordinates landing in the same target with the same sign.

**Concentration tools.** Two are load-bearing. (i) The **Hanson–Wright inequality** (1971): for a Rademacher vector $z$ and symmetric $B$, $\mathbb{E}\,|z^TBz - \mathrm{tr}\,B|^\ell \le C^\ell \max\{\sqrt\ell\,\|B\|_F,\ \ell\,\|B\|_2\}^\ell$ — a moment bound for a quadratic form controlled by the Frobenius and operator norms of $B$. (ii) The **moment method via monomial-to-graph combinatorics**: expanding a high power of a quadratic form into monomials, associating each monomial with a labeled multigraph, grouping by graph, and counting — the technique used by Wigner (1955) and Füredi–Komlós (1981) to analyze the eigenvalue spectrum of random matrices.

## Baselines

- **Dense Gaussian / subspace projection (Johnson–Lindenstrauss 1984; Frankl–Maehara; Dasgupta–Gupta).** In norm-preserving scaling, $S$ has i.i.d. $N(0,1/k)$ entries (or projects onto a random subspace and rescales). Achieves optimal $k=\Theta(\varepsilon^{-2}\log(1/\delta))$. **Gap:** $\Theta(kd)$ time per vector; nothing about it is fast or sparse.

- **Indyk–Motwani / Achlioptas Rademacher (Achlioptas 2003, "Database-friendly random projections").** Achlioptas's sparse distribution sets each entry to $+\sqrt{3/k}$ w.p. $1/6$, $-\sqrt{3/k}$ w.p. $1/6$, and $0$ w.p. $2/3$ in the norm-preserving convention — equivalently $\pm\sqrt{3/d}$ in the unnormalized projection convention — so two thirds of $S$ is zero in expectation, a 3× speedup, while keeping the same $k$. The proof needs only the same one-dimensional concentration as the dense random projection. **Gap:** the density can only be cut by a *constant* factor; pushing further fails because a sparse matrix distorts a sparse vector (the $x=e_i$ obstacle above). Column sparsity stays $\Theta(k)$.

- **CountSketch / Charikar–Chen–Farach-Colton; Thorup–Zhang.** One nonzero per column ($s=1$), $\pm1$ sign, target chosen by a hash. Extremely sparse, $O(\log(1/\delta))$ nonzeros per column across repetitions. **Gap:** the norm is recovered by a **median** of independent estimates, which is **nonlinear** — it is not an embedding into $\ell_2$ (or any normed space). This breaks applications that require an actual $\ell_2$ image: nearest-neighbor search in the reduced space, approximate regression, and learning classifiers by stochastic gradient descent (which needs a differentiable estimator).

- **DKS sparse JL (Dasgupta–Kumar–Sarlós 2010).** The hashed sparse line gives the first $o(k)$ column sparsity that still embeds into $\ell_2$; the original DKS analysis gives polylogarithmic sparsity above $\varepsilon^{-1}$, and the sharpened analysis of the same with-replacement scheme reaches $s=\tilde O(\varepsilon^{-1}\log^2(1/\delta))$ nonzeros per column. Each source coordinate is replicated $s$ times (preserving the $\ell_2$ norm), then hashed to $s$ target coordinates **with replacement**, each with a random sign; the estimator is the linear $\|Sx\|_2$. To control the variance it needs the input to be nice, $\|x\|_\infty = O(\sqrt\varepsilon)$, achieved by a (block-)Hadamard preconditioner; the proof bounds the noise from each hash bucket and the cross-bucket correlations via an FKG / moment-generating-function argument. **Gap:** the with-replacement scheme itself has a near-matching lower bound $s=\Omega(\varepsilon^{-1}\lceil\log^2(1/\delta)/\log^2(1/\varepsilon)\rceil)$, i.e. $\tilde\Omega(\varepsilon^{-1}\log^2(1/\delta))$ — a $\log(1/\delta)$ above the $\varepsilon^{-1}\log(1/\delta)$ one might hope for; and it requires $O(ds\log k)$ random bits to sample $S$ (prohibitive for streaming). It is open what the best achievable $s$ is and whether a low-randomness construction exists.

- **Ailon–Liberty and other FFT-based fast transforms.** These embeddings reach $O(d\log d)$ time. **Gap (shared by FFT-based fast transforms):** they cost $\Omega(d\log d)$ *even when $\|x\|_0 = 1$*, so for sparse inputs and streaming updates they are slower than the naïve $O(k\cdot\|x\|_0)$ dense product.

## Evaluation settings

The natural yardsticks are the high-dimensional geometric and numerical primitives that JL accelerates, with the metrics being (a) the target dimension $k$ as a function of $\varepsilon,\delta$ (or $n$), (b) the column sparsity / number of nonzeros $s$, (c) the per-vector embedding time as a function of $d$, $k$, $\|x\|_0$, and (d) the random-seed length needed to sample the matrix (the relevant cost in the streaming / turnstile model, where an update $x\leftarrow x+v\,e_i$ must be applied to the sketch).

- **Distortion guarantee.** Preserve $\|x\|_2$ to $1\pm\varepsilon$ with probability $1-\delta$ for any fixed $x$ (distributional JL), and via union bound preserve all $\binom{n}{2}$ pairwise distances of $n$ points (metric / flattening JL) into $O(\varepsilon^{-2}\log n)$ dimensions.
- **Embeddings into $\ell_p$.** $p=2$ is the main target; $p=1$ is relevant for approximate nearest neighbor (the embedding bottleneck of $\varepsilon$-ANN search) and point-location reductions.
- **Streaming / turnstile model.** A vector receives coordinate-wise updates; the figure of merit is per-update time and total seed length.
- **Numerical linear algebra sketches.** Approximate matrix product, linear regression, and best rank-$k$ approximation in the streaming model, where a JL distribution is applied as an oblivious subspace/sketch.
- **Input regimes.** Both dense vectors (where $d\log d$ vs. $kd$ matters) and sparse/spiky vectors (where the $\ell_\infty/\ell_2$ ratio $\nu$ governs whether a sparse matrix can be used directly).

## Code framework

The scaffold is a bare JL-sketch harness: a generic linear embedding $x \mapsto Sx$ with the optimal row count baked in, an unbiased-estimator skeleton, and one empty slot for *how* the $k\times d$ map is built and applied. The unresolved choice is what kind of structure or sparsity makes that map fast.

```python
import numpy as np

def jl_rows(eps, delta):
    # Optimal target dimension for distributional JL.
    return int(np.ceil(C_ROWS * eps**-2 * np.log(1.0 / delta)))

C_ROWS = 4.0  # absolute constant from the concentration bound

class Embedding:
    """A linear map R^d -> R^k preserving ||x||_2 up to (1 +/- eps) w.p. 1 - delta."""

    def __init__(self, d, eps, delta, seed=0):
        self.d = d
        self.k = jl_rows(eps, delta)
        self.eps = eps
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self._build()

    def _build(self):
        # TODO: sample the embedding map.
        # The map is linear, k = Theta(eps^-2 log(1/delta)), and
        # ||map(x)||_2^2 must concentrate around 1 for every unit x.
        # Questions: dense, structured-and-fast, or sparse; how many nonzeros
        # per column; whether a preconditioner is needed for spiky inputs.
        pass

    def apply(self, x):
        # TODO: compute Sx. Cost should beat the dense O(k * nnz(x)) product;
        # for streaming we also want apply(e_i) to be cheap.
        raise NotImplementedError

def estimate_sq_norm(embedding, x):
    # Linear estimator of ||x||_2^2 (NOT a median -- we need an l2 image).
    y = embedding.apply(x)
    return float(np.dot(y, y))
```
