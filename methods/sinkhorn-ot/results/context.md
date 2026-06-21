# Context: Fast computation of transportation distances between histograms

## Research question

Given two histograms (probability vectors) `r` and `c` on `d` bins, both living in the simplex
`Σ_d = {x ∈ R^d_+ : x^T 1 = 1}`, and a `d × d` ground cost matrix `M` whose entry `m_ij` is the
cost of moving a unit of mass from bin `i` to bin `j`, we want the cheapest way to morph `r` into
`c`. The cost of a transport plan `P` — a nonnegative matrix whose row sums are `r` and column
sums are `c` — is the Frobenius inner product `⟨P, M⟩ = Σ_ij p_ij m_ij`, and the transportation
distance is

```
d_M(r, c) = min_{P ∈ U(r,c)} ⟨P, M⟩,   U(r,c) = {P ∈ R^{d×d}_+ : P 1 = r, P^T 1 = c}.
```

This family of distances is the only one among the usual simplex distances (Hellinger, χ², KL,
total variation) that is *parameterized* by a ground metric `M`, which lets it exploit relationships
between bins — synonyms in a bag of words, neighboring pixels in an image. Empirically it is a strong
distance for histogram retrieval and classification. Computing one such distance is a linear program,
and for general `M` every known exact algorithm scales at least cubically in `d`. The question is
how to compute a transportation-style distance between histograms for general `M`.

## Background

**The transportation polytope and its probabilistic meaning.** `U(r,c)` is the set of all
nonnegative `d × d` matrices with row sums `r` and column sums `c`. If `X, Y` are random variables
on `{1,…,d}` with marginals `r` and `c`, then `U(r,c)` is exactly the set of joint distributions
(contingency tables) of `(X, Y)`. One distinguished member is the **independence table** `rc^T`,
the joint law of independent `X, Y`.

**Entropy and the basic information inequality.** With `h(P) = −Σ p_ij log p_ij` the entropy of a
table and `h(r) = −Σ r_i log r_i` the entropy of a marginal, a basic information-theoretic
inequality (Cover & Thomas, *Elements of Information Theory*, §2) holds for every joint
distribution:

```
∀ P ∈ U(r,c),   h(P) ≤ h(r) + h(c),
```

and the bound is tight precisely at the independence table, since `h(rc^T) = h(r) + h(c)`. The gap
is the mutual information:

```
KL(P || rc^T) = h(r) + h(c) − h(P) = I(X;Y) ≥ 0.
```

So among all couplings with marginals `r, c`, the independence table is the unique maximum-entropy
one, and low mutual information ⇔ high entropy ⇔ "close to independent".

**Where the LP optimum lives.** By the theory of linear programming the optimum of `d_M` is
attained at a **vertex** of `U(r,c)`, and a vertex of the transportation polytope is a sparse table
with at most `2d − 1` nonzero entries (Brualdi, *Combinatorial Matrix Classes*, §8.1.3). The
optimal plan is therefore a near-deterministic, "extreme" coupling: most of the time `X = i` forces
a single destination `j`.

**When the LP optimum is a metric.** `d_M(r,c)` satisfies the distance axioms when `M` is itself a
metric matrix — nonnegative, symmetric, zero on the diagonal, and satisfying
`m_ij ≤ m_ik + m_kj` (Villani, *Optimal Transport: Old and New*, §6.1). The classical proof of the
triangle inequality goes through the **gluing lemma** (Villani, *Topics in Optimal Transportation*,
Lemma 7.6): given a coupling of `(x,y)` and one of `(y,z)`, one can "glue" them along the shared `y`
marginal into a coupling of `(x,z)` whose cost is bounded by the sum.

**The maximum-entropy principle.** Jaynes (1957) and, in the regularized-estimation form,
Dudík & Schapire (2006): when many configurations are compatible with the constraints, prefer the
one of maximum entropy — the least committal, most plausible.

**Matrix scaling — Sinkhorn & Knopp (1967).** For a nonnegative matrix `A` whose positive entries
have enough support (in particular, every positive entry lies on a positive diagonal; strictly
positive `A` is the simple case), there exist positive diagonal scalings `D₁, D₂` such that
`D₁ A D₂` has prescribed positive row sums and column sums. The scaled matrix is unique, and the
diagonal factors are unique up to the reciprocal rescaling `D₁ -> sD₁`, `D₂ -> D₂/s`. Moreover the
simple iterative procedure of alternately rescaling the rows to the target row sums and the columns
to the target column sums **converges** to it (Sinkhorn & Knopp, *Pacific J. Math.* 21:343–348).
When `A > 0` the support condition is automatic. The same alternating-fit iteration appeared
independently many times — iterative
proportional fitting (Deming & Stephan 1940), the RAS method (Bacharach 1965), and, in
transportation economics, the **gravity model** for estimating origin–destination flows
(Erlander & Stewart, *The Gravity Model in Transportation Analysis*). Knight (2008) gives a modern
convergence analysis (Franklin & Lorenz 1989).

**A precedent for regularizing transport.** Ferradans et al. (2013), *Regularized Discrete Optimal
Transport*, observe that in vision applications (color transfer) the raw optimal matching is too
irregular, and add a graph-based penalty on the transport plan to smooth it. Their construction is
tailored to vision and uses the underlying transport solver.

## Baselines

- **Network simplex / min-cost-flow solvers** (Ahuja, Magnanti & Orlin, *Network Flows*; Orlin
  1993). The exact LP for `d_M`. Returns a vertex (`≤ 2d − 1` nonzeros) optimum. General-`M`
  worst-case cost `O(d^3 log d)`.

- **Rubner et al. (1997) EMD solver** ("Earth Mover's Distance"). The standard transportation-LP
  implementation used in content-based image retrieval. Exact, general `M`.

- **Pele & Werman (2009) FastEMD** (`emd_hat_gd_metric`). Speeds up EMD by exploiting thresholded
  metric ground costs; the general-`M` complexity is `O(d^3 log d)` (their §2.1).

- **Approximate / restricted-ground-metric EMD** — embeddings and sketches (Indyk & Thaper 2003;
  Grauman & Darrell 2004; Shirdhonkar & Jacobs 2008; Andoni et al. 2009). Achieve sub-cubic or even
  linear time by constraining `M` (e.g. to ℓ¹ / wavelet-tree structure).

- **Unparameterized simplex distances** — Hellinger, χ², Kullback–Leibler, total variation, and the
  Gaussian/squared-Euclidean kernel. Cheap and closed-form.

## Evaluation settings

- **MNIST digits.** 20×20 pixel-intensity images normalized to histograms on `d = 400` bins; the
  natural ground metric `M` is the Euclidean distance between the 400 grid points. Classification by
  one-vs-one SVM (libsvm) on a kernel `exp(−d/t)` built from a distance `d`, with the kernel
  bandwidth `t` chosen by cross-validation over data-driven distance quantiles, the SVM constant by
  inner cross-validation, and non-positive-definite kernels regularized by a diagonal shift. Protocol:
  4-fold (1 train / 3 test) cross-validation repeated 6 times.

- **Synthetic timing.** Histograms drawn uniformly from the `d`-simplex (Smith & Tromble 2004) and
  random ground-metric matrices `M` (from `d` points spread by a spherical Gaussian in dimension
  `d/10`, then divided by their median), sweeping `d`. Wall-clock per distance is measured against
  the public Rubner and Pele–Werman EMD codes, on a single CPU core and on a GPU.

## Code framework

What already exists: dense linear algebra (`numpy` / GPU array libraries), elementwise vector ops,
and exact LP transportation solvers to use as a slow baseline. The pieces below are the empty slots
for a fast histogram-distance routine.

```python
import numpy as np

def emd(a, b, M):
    """Exact transportation LP via network simplex.
    Returns a vertex plan with <= 2d-1 nonzeros. Cost ~ O(d^3 log d)."""
    pass

def transport_plan(a, b, M, num_iter=1000, stop_thr=1e-9):
    """Compute a fast transport-style plan P with row sums a and column sums b.
    b is one target histogram.
    """
    # TODO
    pass

def transport_cost(a, b, M, num_iter=1000, stop_thr=1e-9):
    """Return <P, M>; b may be one target or a matrix of target histograms."""
    pass

def stable_transport_plan(a, b, M, num_iter=1000, stop_thr=1e-9):
    """Numerically stable variant for the harder-to-compute regime."""
    pass
```
