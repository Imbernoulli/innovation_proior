Let me start from what actually hurts. I have observational samples of `X = (X_1, ..., X_d)` from a
nonlinear additive-noise model `X_j = f_j(X_{pa(j)}) + N_j`, independent noise, and I want the directed
DAG. I already accept the field's hard-won framing: the DAG is identifiable here because the nonlinearity
breaks the cause/effect symmetry, and the irreducibly causal part is the *topological order* — get a
correct order and the rest is per-node regression and variable selection. So the real question is how to
recover the order, and every method I know does it the same indirect way: posit candidate parent sets,
fit regressions, and compare a fit-quality score. CAM does it greedily by residual variance — repeatedly
regress every remaining variable on the current prefix and append the lowest-residual-variance one;
RESIT does it by regressing and testing residual independence. Both are `O(d^2)` (or worse) fits, both
are noisy at modest sample size, and both commit to greedy choices they never revisit. I keep wanting to
ask: is there a quantity that pins down the order *directly*, read off the distribution itself, without a
single regression in the order-finding loop? If I had that, the order would stop being a search and
become a measurement.

Let me look at what the additive-noise structure does to the *score* of the distribution, because the
score is the most distribution-intrinsic object I can name — `s(x) = ∇_x log p(x)`, the gradient field
of the log-density. Write the joint log-density. Because the model is Markov to the DAG and the noise is
additive,

  `log p(x) = sum_j log p_{N_j}( x_j - f_j(x_{pa(j)}) )`.

Take the gradient component along `x_j`. Two kinds of terms can depend on `x_j`: the term `j` itself,
where `x_j` is the argument of its *own* noise density, and any term `c` for which `j ∈ pa(c)`, where
`x_j` enters through the *child's* mechanism `f_c`. So

  `s_j(x) = ∂/∂x_j log p_{N_j}(x_j - f_j(pa(j))) + sum_{c : j ∈ pa(c)} ∂/∂x_j log p_{N_c}(x_c - f_c(pa(c)))`.

The first term involves `x_j` and `x_j`'s parents (through `f_j`); the second involves `x_j`'s children
and their other parents. Already the score mixes parents and children, so reading the order off `s`
itself is murky. Let me push to the *second* derivative along `x_j`, the `j`-th diagonal entry of the
Hessian of `log p`,

  `H_{jj}(x) = ∂^2/∂x_j^2 log p(x)`.

Now specialize to the case the identifiability theory most cleanly covers: Gaussian noise,
`log p_{N_j}(z) = -z^2/(2σ_j^2) + const`. The first term is
`∂^2/∂x_j^2 [ -(x_j - f_j(pa(j)))^2 / (2σ_j^2) ]`. Crucially `f_j` does *not* depend on `x_j` — there are
no self-loops — so `x_j - f_j(pa(j))` is just `x_j` minus a constant-in-`x_j` quantity, and the second
derivative is exactly `-1/σ_j^2`, a *constant*. Every bit of `x`-dependence in `H_{jj}` therefore comes
from the *second* group of terms — the children. And here is the payoff: if `j` is a **leaf**, it has no
children, the second group is empty, and `H_{jj}(x) = -1/σ_j^2` is constant over the whole sample. Its
*variance over `x`* is zero. Conversely, if `j` has a child, that child's mechanism `f_c` is nonlinear in
`x_j`, so `H_{jj}` genuinely varies with `x` and its variance is strictly positive. So I have an exact,
distributional characterization I did not have before:

  **`j` is a leaf of the DAG  ⟺  Var_x[ H_{jj}(x) ] = 0.**

This is the thing I was hunting for — the order read directly off the score's Jacobian, no regression in
the loop. Let me make sure I believe the "only children matter" claim, because it is the whole method.
The diagonal Hessian entry `H_{jj}` is `∂_{x_j}^2 log p`. Under the factorization, the only factors whose
log depends on `x_j` are factor `j` (own noise) and the child factors. Factor `j` contributes the
constant `-1/σ_j^2` (Gaussian noise, no self-dependence of `f_j` on `x_j`). Child factor `c` contributes
`∂_{x_j}^2 [ -(x_c - f_c(pa(c)))^2 / (2σ_c^2) ]`, which expands to terms in `f_c` and its derivatives in
`x_j` — nonzero and `x`-dependent precisely because `f_c` is a nonlinear function of `x_j`. So
non-constancy of `H_{jj}` is equivalent to "has at least one child," i.e. "is not a leaf." Good. The leaf
characterization is exact in the population.

So the order algorithm writes itself, and it is a *global* measurement, not a greedy search. Estimate the
diagonal of the Hessian of `log p` at the sample points; the variable whose diagonal-Hessian column has
the *minimum* empirical variance is the current leaf (in finite samples the variance is not exactly zero
for the true leaf, but it is the smallest). Append that variable to the order as the *last* element. Now
remove it: a leaf deleted from an ANM leaves an ANM over the remaining `d-1` variables (the leaf's value
does not enter any other mechanism, since it has no children), so the reduced system is again a
Gaussian-noise ANM and its leaf is the second-to-last variable. Recompute the diagonal Hessian on the
reduced data, find its leaf, prepend, and recurse. After `d-1` removals one variable remains — the root —
and reversing the leaf sequence gives the topological order, roots first. This is strictly better-posed
than CAM's greedy residual-variance append: there is no commitment based on a noisy fit comparison that
compounds down the chain; each step is a direct read of a distributional quantity, and the problem shrinks
by one variable each time, so the small-sample behavior is far steadier than regressing on an
ever-growing predictor set.

Everything now hinges on estimating, from finite samples, the diagonal of the Hessian of `log p` — the
quantities `∂_{x_j} log p` (the score) and `∂_{x_j}^2 log p` (its diagonal Jacobian) at each data point. I
cannot differentiate a density I do not have. But I do not need the density; I need the score and the
diagonal of its Jacobian, and *score estimation* is a solved problem via Stein's identities with a kernel
— no density, no normalizing constant. Let me set it up. Use a Gaussian kernel; form the Gram matrix
`K_{ij} = exp(-||x_i - x_j||^2 / (2 s^2)) / s`, with the bandwidth `s` chosen as the *median* of the
pairwise distances (the standard heuristic that adapts the kernel to the data scale with no tuning). The
first-order Stein identity, regularized, gives the score at the sample points as the solution of a ridge
system,

  `G = (K + η_G I)^{-1} ∇K`,

where `∇K` is the sample gradient of the kernel and `G` is the `(n × d)` matrix of estimated scores. The
ridge `η_G` (small, `10^{-3}`) keeps the solve well-conditioned. The kernel gradient has the explicit
form `∇K[k, j] = -sum_i (x_{k,j} - x_{i,j}) K_{ik} / s^2`, which I can read straight off
`∂/∂x_{k,j} exp(-||x_k - x_i||^2/(2s^2))`. Then the *second*-order identity gives the diagonal Hessian.
Differentiating the kernel twice gives `∇^2 K[k, j] = sum_i (-1/s^2 + (x_{k,j} - x_{i,j})^2 / s^4) K_{ik}`,
and the second-order Stein estimate of the diagonal Hessian of `log p` is

  `H = -G^2 + (K + η_H I)^{-1} ∇^2 K`,

with `η_H = 10^{-3}`. The `-G^2` term is not optional decoration: the second-order Stein solve estimates
`∂_{jj} p(x) / p(x)`, and
`∂_{jj} p(x) / p(x) = ∂_{jj} log p(x) + (∂_j log p(x))^2`; subtracting the squared score estimate
therefore converts it to the genuine diagonal Hessian of `log p`. Both `G` and `H` are `O(n^3)` matrix solves,
entirely practical for the hundreds-to-few-thousand sample sizes and tens-of-nodes graphs in play.

Two finite-sample refinements before I take the variance. First, the columns of `H` live on different
scales (each `-1/σ_j^2` plus child-driven variation), so comparing raw variances across variables is
apples-to-oranges; normalize each column by its mean before computing the variance, so the leaf criterion
is scale-fair. Second, the dispersion measure: variance is the natural choice and works, though a robust
alternative (median-absolute-deviation around the column median) is available for heavy-tailed cases — I
keep variance as the default. The leaf is then `argmin` over remaining columns of the normalized variance.

Once the order is in hand the rest is the well-trodden, *regularized* edge-selection half — which I
explicitly do not want to reinvent, because the order was the hard part and the pruning literature is
mature. For each node in order position `pos`, fit a flexible nonlinear regression of it on its
predecessors and keep a parent only if it genuinely contributes. The canonical realization is CAM-style
pruning: fit a generalized additive model of each node on its current parents and drop any parent whose
covariate significance test exceeds a small p-value (e.g. `0.001`); this gives the screening property
that true edges survive with high probability. The order from the score-matching stage feeds straight in,
and because a correct order's super-DAG already yields consistent intervention distributions, the pruning
is pure refinement — it lowers SHD and improves readability without being needed for correctness.

So let me assemble the whole thing into the code I would actually run — the score/diagonal-Hessian
estimator via Stein, the iterative leaf-removal order recovery, and the CAM-style pruning along the
recovered order:

```python
import numpy as np


def stein_hessian_diagonal(X, eta_G, eta_H, s=None):
    """Diagonal of the Hessian of log p_X at the sample points, via first- and
    second-order Stein identities with a Gaussian kernel. Returns (n, d)."""
    n, d = X.shape
    X_diff = X[:, None, :] - X[None, :, :]                 # (n, n, d)
    sqdist = np.sum(X_diff ** 2, axis=2)                   # (n, n)
    if s is None:
        flat_dist = np.sqrt(np.maximum(sqdist, 0.0)).ravel()
        mid = (flat_dist.size - 1) // 2
        s = np.partition(flat_dist, mid)[mid]                 # torch median convention
    K = np.exp(-sqdist / (2 * s ** 2)) / s                 # Gaussian Gram matrix
    # kernel gradient: nablaK[k, j] = -sum_i (x_{k,j} - x_{i,j}) K[i,k] / s^2
    nablaK = -np.einsum('kij,ik->kj', X_diff, K) / s ** 2  # (n, d)
    I = np.eye(n)
    G = np.linalg.solve(K + eta_G * I, nablaK)             # score estimate (n, d)
    # second derivative: nabla2K[k, j] = sum_i (-1/s^2 + (x_{k,j}-x_{i,j})^2/s^4) K[i,k]
    nabla2K = np.einsum('kij,ik->kj',
                        -1.0 / s ** 2 + X_diff ** 2 / s ** 4, K)   # (n, d)
    # diagonal Hessian of log p = -G^2 + (K + eta_H I)^{-1} nabla2K
    return -G ** 2 + np.linalg.solve(K + eta_H * I, nabla2K)


def compute_topological_order(X, eta_G=1e-3, eta_H=1e-3):
    """Iterative leaf detection: leaf = min-variance diagonal-Hessian column."""
    n, d = X.shape
    data = X.copy()
    active = list(range(d))
    order = []
    while data.shape[1] > 1:
        H = stein_hessian_diagonal(data, eta_G, eta_H)
        H = H / H.mean(axis=0)                  # scale-fair column normalization
        leaf = int(np.argmin(H.var(axis=0)))    # constant column => leaf
        order.append(active[leaf])
        active.pop(leaf)
        data = np.delete(data, leaf, axis=1)    # removing a leaf leaves an ANM
    order.append(active[0])
    order.reverse()                             # roots first
    return order


def cam_prune(X, order, cutoff=1e-3):
    """CAM-style significance pruning along the known order (reused as-is):
    keep a parent only if its smooth-term significance test clears the cutoff."""
    d = X.shape[1]
    B = np.zeros((d, d))
    for pos in range(1, len(order)):
        j = order[pos]
        parents = order[:pos]
        pvals = gam_significance(X[:, j], X[:, parents])   # one p-value per parent
        for k, p in enumerate(parents):
            if pvals[k] <= cutoff:
                B[j, p] = 1.0    # B[child, parent] = 1 means parent -> child
    return B


def run_causal_discovery(X):
    """SCORE: score-matching order recovery + CAM pruning. Returns adjacency B
    with B[i, j] != 0 meaning j -> i."""
    order = compute_topological_order(X)
    return cam_prune(X, order)
```

Let me trace the causal chain so nothing is hand-waved. I wanted the topological order — the irreducibly
causal part of nonlinear ANM discovery — without the greedy, regression-based, error-compounding search
that CAM and RESIT use. Looking at the *score* of the distribution, I found that for Gaussian-noise ANMs
the `j`-th diagonal entry of the Hessian of `log p` is the constant `-1/σ_j^2` *plus* a term that varies
with `x` only through `x_j`'s children — so a variable is a leaf exactly when that diagonal Hessian entry
has zero variance over the data. That turns order recovery into a direct measurement: estimate the
diagonal Hessian, take the min-variance column as the leaf, remove it (which leaves a smaller ANM),
recurse, reverse. The diagonal Hessian is estimable from samples with no density via first- and
second-order Stein identities under a Gaussian kernel — score `G = (K+η_G I)^{-1}∇K`, diagonal Hessian
`-G^2 + (K+η_H I)^{-1}∇^2 K` — with a median-distance bandwidth and small ridge regularizers, all
`O(n^3)`. With the order recovered, edge selection is CAM-style significance pruning, which is
refinement, not correctness, since a correct order already yields consistent interventions. The result is
a fast, search-free, distributionally-grounded recovery of the nonlinear causal DAG.
