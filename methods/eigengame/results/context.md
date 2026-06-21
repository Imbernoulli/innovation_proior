# Context: decentralized recovery of the top-k principal components at scale

## Research question

Given a data matrix `X ∈ R^{n×d}` whose rows are (centered) samples, the principal
components are the eigenvectors of the `d×d` Gram matrix `M = XᵀX` — equivalently the
right singular vectors of `X`. We want the **top-k eigenvectors themselves** (not merely the
subspace they span), ordered by eigenvalue, when both `n` and `d` are enormous: think
`n ≈ 10⁶` samples and `d ≈ 2·10⁷` features (the flattened activations of a deep ResNet over
ImageNet, tens of terabytes of data). A full singular value decomposition costs
`O(min{nd², n²d})` time and `O(nd)` space — utterly infeasible here. Even forming `M`
explicitly (`d²` entries) is out of the question.

The catch that makes this harder than it looks: there is an important difference between
the top-k **subspace** and the top-k **components**. Many fast methods return *some*
orthonormal basis of the subspace of maximal variance and leave the rotation-within-subspace
to a downstream step. But the components are wanted as interpretable directions (the
first principal axis, the second, …) — each must align with a specific direction of
variance, which depends on the covariance, not just span the right subspace. A method that
only nails the subspace has solved a different, easier problem.

The operational pain point is **hardware**. Modern compute is a farm of accelerators (dozens
to thousands of GPUs/TPUs) connected by fast interconnects, ideal for data- and
model-parallel workloads. We would like an eigensolver that maps onto that farm: each
accelerator owns one component, sees its own shard of data, and communicates as little as
possible. A solution would have to (a) recover the actual components, ordered; (b) never
materialize `M` or run a global SVD; (c) require no centralized step that forces all `k`
vectors to synchronize every iteration; (d) come with a convergence guarantee that does not
depend on a lucky initialization. The cultural mood of the moment is captured by a remark
made at a major conference: *"It seems easier to train a bi-directional LSTM with attention
than to compute the SVD of a large matrix."*

## Background

**The eigenvalue problem and the matrix `R`.** For symmetric `M`, the eigendecomposition
finds orthonormal `V` with `MV = VΛ`, `Λ` diagonal. Left-multiplying by `Vᵀ` and using
`VᵀV = I` gives `VᵀMV = Λ`. For an *estimate* `V̂` of the eigenvectors define
`R(V̂) := V̂ᵀMV̂`. Its diagonal entries `R_ii = ⟨v̂_i, M v̂_i⟩` are **Rayleigh quotients**
(variance captured along `v̂_i`, since `‖v̂_i‖=1`); its off-diagonals
`R_ij = ⟨v̂_i, M v̂_j⟩` measure alignment of `v̂_i` with `v̂_j` under the *generalized*
inner product `⟨·,·⟩_M`. Two facts about `R` set up everything:

- Maximizing the trace: `max_{V̂ᵀV̂=I} Tr(V̂ᵀMV̂) = Tr(V̂V̂ᵀM)`. When `k=d` this equals
  `Tr(M)` and is **independent of `V̂`** — the trace objective alone cannot distinguish one
  orthonormal basis of the top-k subspace from another. It captures variance/subspace but
  says nothing about *which* directions are the components.
- The eigenvalue equation `VᵀMV = Λ` is, by definition, a statement that `R` is **diagonal**
  at the true eigenvectors: the eigenvectors are exactly the orthonormal `V̂` for which `R(V̂)`
  has no off-diagonal entries.

**Hebb's rule and Oja's rule.** In a connectionist reading, the top eigenvector of `M`
can be found by the additive update `v ← v + η M v` (Hebb's rule), which finds the leading
direction but lets `‖v‖` grow without bound. Oja's rule adds a self-normalizing correction,
`v ← v + η(I − vvᵀ)Mv`; the subtractive `−vvᵀMv` term implicitly keeps `v` near the unit
sphere. In ML, "Oja's algorithm" usually means Hebb's rule followed by an explicit
renormalization `v ← v/‖v‖`; as `η→∞` this becomes the classical power method. Adding a
renormalization to Oja's rule recovers Krasulina's algorithm. In the language of Riemannian
geometry (Absil, Mahony & Sepulchre 2009), `v/‖v‖` is a **retraction** onto the sphere and
`(I − vvᵀ)` **projects** the ambient gradient `Mv` onto the sphere's tangent space.

**Going from top-1 to top-k.** All of the above are top-1 rules. The standard way to extend
them to `k` components is to bolt on an orthonormalization after each update — typically a
`QR` factorization of the `d×k` iterate plus some sign bookkeeping (this is the usual
top-k "Oja's algorithm"). Matrix Krasulina (Tang 2019) projects onto the Stiefel manifold of
orthonormal matrices and re-orthonormalizes with `QR`. An alternative to `QR` is
**deflation**: solve for the `i`-th component with a top-1 rule on data with the already-found
subspace removed, `X_{(i)} ← X(I − Σ_{j<i} v̂_j v̂_jᵀ)`. The **Generalized Hebbian Algorithm**
(GHA, Sanger 1989) interleaves deflation with learning in a single update,

  `Δv̂_i = 2[ M v̂_i − (v̂_iᵀM v̂_i) v̂_i − Σ_{j<i} (v̂_iᵀM v̂_j) v̂_j ]`,

and converges asymptotically to the components (not just the subspace), and can be run in a
distributed fashion. The Jacobi eigenvalue algorithm instead represents the eigenvector
matrix as a product of Givens rotations and rotates `M` toward diagonal form.

**Riemannian optimization rates.** For nonconvex objectives on a Riemannian manifold,
generic gradient descent with a constant step reaches a point with small Riemannian gradient,
`‖∇^R f(x)‖ ≤ ρ`, in `⌈(f(x₀) − f*)/ξ · 1/ρ²⌉` iterations, provided the objective is bounded
below and satisfies a sufficient-decrease (descent-Lipschitz) condition (Boumal, Absil &
Cartis 2019, Thm 2.5). This `1/ρ²` rate is the available engine for turning an iterative
update on the sphere into a finite-sample guarantee.

**Diagnostic facts about the prior art, knowable in advance.** Trace-maximization is
provably blind to the rotation inside the top-k subspace (the `Tr(M)` identity above).
Among the streaming/online methods, those that recover the actual components rely on a
re-orthonormalization step — `QR`, Stiefel projection, or matrix inversion — that couples
all `k` vectors together each iteration. GHA does recover components and is distributable,
yet (as one can check by computing the Jacobian of its update) the GHA update is **not the
gradient of any scalar function**, so it carries no notion of an objective each vector is
optimizing. Methods that only target the subspace (Matrix Krasulina, Frequent Directions,
sketching) often come with guarantees that *assume the initialization already captures the
top-`(k−1)` subspace* — an assumption that is unlikely to hold when `d ≫ k`.

**A cultural precedent.** Around this time, generative modeling had been recast as a
two-player zero-sum game (Goodfellow et al. 2014), with learning driven by competing
objectives rather than a single loss. Two-player zero-sum games are well understood;
many-player general-sum games are far less so, and their equilibrium concepts (Nash, and the
complexity of computing one) are active territory in the field.

## Baselines

A new method for ordered top-k component recovery would be measured against these.

- **Power method / Oja's algorithm (top-1, then top-k via `QR`).** Repeatedly apply
  `v ← v + η M v` (`v ← v/‖v‖`); for `k` components, stack into `V̂ ∈ R^{d×k}` and run
  `Q,R ← QR(V̂); V̂ ← Q·S` with a sign correction `S` each step. Converges to the actual
  components and is the most heavily studied online PCA method. *Gap:* the per-iteration `QR`
  on a `d×k` matrix is a global, synchronizing operation across all `k` vectors — a serial
  bottleneck that does not distribute across accelerators, and it grows with `d`.

- **Krasulina / Matrix Krasulina (Tang 2019).** Oja's rule with renormalization; the matrix
  version projects onto the Stiefel manifold and re-orthonormalizes with `QR`. *Gap:*
  targets the top-k **subspace**, not the ordered components, so it answers a different
  question; its exponential-convergence guarantee assumes the initial guess already captures
  the top-`(k−1)` subspace, implausible when `d ≫ k`.

- **Generalized Hebbian Algorithm (Sanger 1989).** Interleaves Hebbian learning with
  deflation in a single update (formula above); recovers the components and can be
  decentralized. *Gap:* its update is not the gradient of any objective (the Jacobian of the
  update is not symmetric, so it cannot be a Hessian), so each vector has no payoff it is
  maximizing — there is no objective-level handle on the method, and it does not obviously
  extend beyond the linear/Hebbian setting.

- **Frequent Directions (Ghashami et al. 2016) and randomized/sketching SVD
  (Halko et al. 2011).** Maintain a small sketch capturing the high-variance subspace; FD
  runs an SVD on the sketch as an inner step. *Gap:* subspace-oriented; FD relies on an SVD
  inner step that becomes infeasible when `d` is in the tens of millions (cannot run SVD on a
  `k×d` sketch at `d ≈ 2·10⁷`), and recovering the components requires a final SVD/rotation.

- **Exact SVD (LAPACK/`numpy.linalg`).** The ground-truth reference. *Gap:* `O(min{nd²,n²d})`
  time and `O(nd)` space — the very thing that is infeasible at the target scale; used only
  to validate on small problems.

## Evaluation settings

- **Synthetic.** `M ∈ R^{50×50}` diagonal (without loss of generality, since the iterates are
  initialized randomly on the sphere). Two spectra: **linear**, eigenvalues equally spaced
  from `1` to `1000`; **exponential**, from `10⁰` to `10³` with equally spaced exponents.
  This isolates the effect of spectral decay / eigenvalue gaps on convergence.
- **MNIST.** Top-`k` (e.g. `k=16`) components of the image data, with minibatch sizes such as
  `1024, 512, 256` to probe sensitivity to batch size.
- **Large-scale.** Flattened activations of a deep ResNet on ImageNet (`n ≈ 10⁶`,
  `d ≈ 2·10⁷`), recovering the top-`32` components — a regime where exact SVD and
  sketch-then-SVD methods do not run at all.
- **Metrics.** (i) **Longest streak**: the number of consecutive components (from the top)
  whose angular error `θ_i = sin⁻¹(√(1 − ⟨v_i, v̂_i⟩²))` is below a threshold (e.g. `π/8`),
  which rewards getting the *ordered* components right. (ii) **Subspace distance**
  `‖V̂ᵀV_{¬k}‖_F²` against the bottom `d−k` true eigenvectors. (iii) Recovered Rayleigh
  quotients (a scree plot) compared to true eigenvalues. (iv) Wall-clock runtime. Learning
  rates are swept over `{10⁻³,…,10⁻⁶}` on held-out runs; results are averaged over several
  trials with standard-error shading. Ground truth on problems with no analytic solution is
  obtained by running a slow, well-converged online method to high Rayleigh-quotient stability.

## Code framework

The primitives that already exist: dense linear algebra (`numpy`/JAX), an exact reference
solver to check against, a data source that can be read in full batch or as a stream of
minibatches, and the standard online-PCA building blocks (a matrix-vector product, a unit-norm
projection). The skeleton below is a bare top-k online-PCA harness. The single empty slot is
the per-vector update that the method will define.

```python
import numpy as np

# --- exact reference (small problems only) -----------------------------------
def reference_components(X, k):
    # eigenvectors of M = X^T X, sorted by eigenvalue (descending)
    M = X.T @ X
    w, U = np.linalg.eigh(M)
    order = np.argsort(w)[::-1]
    return U[:, order[:k]], w[order[:k]]

# --- generic online-PCA primitives that predate the method -------------------
def normalize_columns(V):
    return V / np.linalg.norm(V, axis=0, keepdims=True)

def rayleigh(X, V):
    XV = X @ V                       # variance carried by each column
    return np.sum(XV * XV, axis=0)

# --- the slot the method will fill -------------------------------------------
def component_update(X, V, i, lr):
    """Return the new estimate of the i-th component given the current
    estimates V (columns) and the data X.
    """
    # TODO: the per-vector objective / update we will design
    pass

def solve_topk(X, k, lr, iters, V0=None):
    d = X.shape[1]
    V = normalize_columns(np.random.randn(d, k)) if V0 is None else V0.copy()
    for _ in range(iters):
        for i in range(k):
            V[:, i] = component_update(X, V, i, lr)   # one step per component
    return V

if __name__ == "__main__":
    X = np.random.randn(200, 50)
    V = solve_topk(X, k=8, lr=1e-3, iters=2000)
    V_true, w_true = reference_components(X, 8)
    print(np.abs(np.sum(V * V_true, axis=0)))   # |cos angle| per component
```
