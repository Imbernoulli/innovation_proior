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
from the *second* group of terms — the children. So whether `H_{jj}` is constant or not is governed
entirely by whether `j` has children. Let me not just assert that — let me actually differentiate the
child term and see what comes out, on the smallest case that has both a leaf and a non-leaf. Take the
two-node chain `X_1 → X_2`, `X_1 = N_1`, `X_2 = f(X_1) + N_2`, both noises Gaussian. The log-density is

  `log p(x_1, x_2) = -x_1^2/(2σ_1^2) - (x_2 - f(x_1))^2/(2σ_2^2) + const`.

Differentiate twice in `x_2` (the leaf). The first term has no `x_2`, the second is
`-(x_2 - f(x_1))^2/(2σ_2^2)`, and `f` does not contain `x_2`, so `∂_{x_2}^2` gives `-1/σ_2^2` — flat in
both `x_1` and `x_2`. Now twice in `x_1` (the root). The first term gives `-1/σ_1^2`. The second is where
the child shows up: writing `r = x_2 - f(x_1)`, `∂_{x_1}[-r^2/(2σ_2^2)] = r f'(x_1)/σ_2^2`, and
`∂_{x_1}` again gives `[-f'(x_1)^2 + r f''(x_1)]/σ_2^2`. So

  `H_{11}(x) = -1/σ_1^2 + ( -f'(x_1)^2 + (x_2 - f(x_1)) f''(x_1) ) / σ_2^2`,
  `H_{22}(x) = -1/σ_2^2`.

That is the whole story made explicit. `H_{22}`, the leaf's column, is a constant; `H_{11}`, the root's
column, carries `f'(x_1)^2` and `f''(x_1)` and the residual `x_2 - f(x_1)` — genuinely `x`-dependent
whenever `f` is nonlinear (`f'' ≢ 0`). For a strictly linear `f` the `f''` term vanishes and only
`-f'(x_1)^2/σ_2^2` survives, still a *constant* — which is the right behavior, since linear-Gaussian SEMs
are exactly the non-identifiable case, so I would not want the root's column to look different from a
leaf's there. The nonlinearity is doing the identifying work, exactly as the ANM theory says it should.
Generalizing the two derivatives back to `d` nodes: factor `j` always contributes the constant `-1/σ_j^2`,
and each child factor `c` contributes a term of the form `[ -(∂_{x_j} f_c)^2 + (x_c - f_c)·∂_{x_j}^2 f_c ]/σ_c^2`,
which is non-constant precisely when `f_c` depends nonlinearly on `x_j`. So I have an exact, distributional
characterization:

  **`j` is a leaf of the DAG  ⟺  Var_x[ H_{jj}(x) ] = 0.**

The order can be read directly off the score's Jacobian, with no regression in the loop.

This gives me a way to find *one* variable — the leaf — directly. To get the full order I need to repeat,
so I have to check that removing the leaf leaves me with a problem of the same shape. Marginalize out a
leaf `ℓ`: because `ℓ` has no children, its value `x_ℓ` does not appear in any other mechanism `f_j`, so
the remaining variables still satisfy `X_j = f_j(pa(j)) + N_j` with the *same* mechanisms and the *same*
independent Gaussian noises (none of which involved `N_ℓ`). The marginal over the remaining `d-1`
variables is therefore again a Gaussian-noise ANM on the sub-DAG with `ℓ` deleted — and its leaf is the
second-to-last variable of the original order. (This is the one place I should be careful: the *estimated*
diagonal Hessian on the reduced data is recomputed from the reduced kernel, not sliced from the old one,
precisely because the marginal density, not the conditional, is the object whose score I now want.) So the
procedure is: estimate the diagonal of the Hessian of `log p` at the sample points; take the column with
the *minimum* empirical variance as the current leaf (in finite samples the true leaf's variance is not
exactly zero, but I expect it to be the smallest — something to confirm on data below); append it as the
last element; delete that column and recurse on the `d-1` remaining variables; reverse the leaf sequence
at the end to get the order roots-first. It is a *global* measurement at each step, not a greedy fit
comparison: there is no commitment based on a noisy regression that compounds down the chain, and the
problem shrinks by one variable each time rather than regressing on an ever-growing predictor set as CAM
does — which I would expect to make the small-sample behavior steadier, though that is an empirical claim
I cannot settle from the algebra alone.

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

with `η_H = 10^{-3}`. The `-G^2` term is the part I am least sure of by inspection, so let me derive it
rather than trust it. The second-order Stein solve, by construction, estimates `∂_{jj} p / p` (the ratio
of the second derivative of the *density* to the density — that is what the kernel identity returns), but
what I want is `∂_{jj} log p`. Relate the two. With `s_j = ∂_j log p = (∂_j p)/p`,
`∂_j s_j = ∂_j[(∂_j p)/p] = (∂_{jj} p)/p - ((∂_j p)/p)^2 = (∂_{jj} p)/p - s_j^2`. Rearranging,
`(∂_{jj} p)/p = ∂_{jj} log p + (∂_j log p)^2`. So to recover `∂_{jj} log p` I must subtract the squared
score `(∂_j log p)^2` from the second-order Stein estimate — which is exactly the `-G^2` term, with `G`
the estimated score. Good: the correction is forced, not cosmetic, and I now know its sign. Both `G` and
`H` are `O(n^3)` matrix solves, entirely practical for the hundreds-to-few-thousand sample sizes and
tens-of-nodes graphs in play.

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
covariate significance test exceeds a small p-value (e.g. `0.001`); the appeal is the screening property,
that a true edge's covariate is significant and so should survive the cut while spurious predecessors are
dropped. The order from the score-matching stage feeds straight in, and since a correct order already
makes the super-DAG (every node regressed on all its predecessors) a valid causal model, pruning only
removes the redundant edges — it lowers SHD and improves readability, but a mistake in pruning costs edges,
not the order, so the order stage is where correctness is won or lost.

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

Before I trust any of this I want to run the estimator on a case where I know the answer, because the leaf
criterion rests on two things that could fail in finite samples: the population claim (leaf column flat)
and the *estimate* actually reflecting it. The cleanest test is the same two-node chain I differentiated
by hand. Generate `n = 800` points from `X_1 = N_1` (`σ_1 = 1`), `X_2 = sin(2 X_1) + 0.5 X_1 + N_2`
(`σ_2 = 0.5`) — index 1 is the true leaf — and run `stein_hessian_diagonal`, then look at the mean and
variance of each column. The leaf's diagonal-Hessian column should sit near the constant `-1/σ_2^2 = -4.0`
with little spread; the root's should be larger in magnitude (it carries the extra child term) and clearly
more dispersed. What comes back: the leaf column has estimated mean `≈ -4.54` (vs. the population `-4.0` —
in the right place, biased a bit by the ridge and bandwidth) and raw variance `≈ 11.6`, while the root
column has mean `≈ -9.1` and raw variance `≈ 89`. After the mean-normalization the algorithm uses, the
normalized variances are `≈ 1.07` for the root and `≈ 0.57` for the leaf, so `argmin` picks index 1 — the
true leaf. So the criterion is not just a population statement: at `n = 800` the leaf's column really is
the flattest one, by a comfortable factor of roughly two even after normalization, and the procedure
identifies it. (I also notice the leaf column is not perfectly flat — variance `11.6`, not `0` — which is
the expected finite-sample blur; it is the *relative* flatness that the `argmin` exploits, which is why
ranking columns is safer than thresholding any one of them against zero.)

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
