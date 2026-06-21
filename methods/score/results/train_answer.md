We observe samples of a random vector $X = (X_1, \dots, X_d)$ drawn from a nonlinear additive-noise model $X_j = f_j(X_{\mathrm{pa}(j)}) + N_j$ with mutually independent noise and nonlinear mechanisms, and we want the fully *directed* acyclic graph of who causes whom — oriented edges, not an undirected skeleton, from purely observational data. The identifiability theory makes this well-posed: the nonlinearity breaks the cause/effect symmetry, so the DAG is recoverable from $P_X$ even under Gaussian noise, and the irreducibly causal part is the *topological order*. Once a correct order is in hand the structured Markov-equivalence-class ambiguity collapses: the structural equations become triangular and what remains is per-node regression plus variable selection, an efficiency problem rather than a correctness one. So the whole difficulty concentrates on recovering the order, and every standard method does it the same indirect way. CAM searches the order greedily by residual variance — repeatedly regressing every remaining variable on the current prefix and appending the one with lowest residual variance — which is an $O(d^2)$ pile of fits on ever-growing predictor sets, noisy at modest sample size, and committed to greedy choices it never revisits. RESIT regresses each variable on candidates and tests residual independence, but its kernel independence tests are expensive and do not scale much past twenty nodes. The continuous-constraint neural methods (GraN-DAG, NOTEARS-MLP) optimize the graph directly under a smooth acyclicity penalty but are non-convex, initialization-sensitive, and prone to over-connection. What I want is a quantity that pins down the order *directly*, read off the distribution itself, with not a single regression inside the order-finding loop — so that order recovery becomes a measurement rather than a search.

I propose SCORE: read the topological order off the **score** of the data distribution, $s(x) = \nabla \log p(x)$, and specifically off the diagonal of its Jacobian. The load-bearing fact is a clean characterization of leaves. Because the model is Markov to the DAG with additive noise, the joint log-density factorizes as $\log p(x) = \sum_j \log p_{N_j}\!\big(x_j - f_j(x_{\mathrm{pa}(j)})\big)$. Differentiating along $x_j$, only two kinds of factors depend on $x_j$: factor $j$ itself, where $x_j$ is the argument of its own noise density, and any child factor $c$ with $j \in \mathrm{pa}(c)$, where $x_j$ enters through the child's mechanism $f_c$. The score therefore already mixes parents and children, which is too murky to order on. So I push to the second derivative, the $j$-th diagonal entry of the Hessian of $\log p$, $H_{jj}(x) = \partial^2 \log p / \partial x_j^2$, and specialize to the Gaussian-noise case the identifiability theory most cleanly covers, $\log p_{N_j}(z) = -z^2/(2\sigma_j^2) + \text{const}$. The own-noise term contributes $\partial_{x_j}^2\big[-(x_j - f_j(\mathrm{pa}(j)))^2/(2\sigma_j^2)\big]$, and because there are no self-loops, $f_j$ does not depend on $x_j$, so this is exactly the *constant* $-1/\sigma_j^2$. Every bit of $x$-dependence in $H_{jj}$ thus comes from the child factors, each of which contributes $\partial_{x_j}^2\big[-(x_c - f_c(\mathrm{pa}(c)))^2/(2\sigma_c^2)\big]$, nonzero and genuinely $x$-varying precisely because $f_c$ is nonlinear in $x_j$. The consequence is exact in the population:
$$j \text{ is a leaf of the DAG} \iff \mathrm{Var}_x\big[H_{jj}(x)\big] = 0.$$
A leaf has no children, so its diagonal Hessian entry is the flat constant $-1/\sigma_j^2$; any non-leaf has a child whose nonlinear mechanism injects $x$-dependence and a strictly positive variance.

This turns order recovery into iterative leaf removal — a global measurement, not a greedy fit comparison. Estimate the diagonal Hessian at the sample points, take the variable whose column has the *minimum* empirical variance as the current leaf (in finite samples the true leaf's variance is smallest rather than exactly zero), and append it as the last element of the order. Then delete that variable: a leaf removed from an ANM leaves an ANM over the remaining $d-1$ variables, since the leaf entered no other mechanism, so the reduced system is again a Gaussian-noise ANM whose leaf is the second-to-last variable. Recurse until one variable — the root — remains, then reverse to get the order roots-first. This is strictly better-posed than CAM's greedy append: there is no commitment based on a noisy fit comparison that compounds down the chain, each step is a direct read of a distributional quantity, and the problem shrinks by one variable per step, so the small-sample behavior is far steadier than regressing on a growing predictor set.

Everything then hinges on estimating, from finite samples, the score and the diagonal of its Jacobian without ever forming the density. That is exactly what Stein's identities with a reproducing kernel give. Use a Gaussian kernel with Gram matrix $K_{ij} = \exp(-\|x_i - x_j\|^2/(2s^2))/s$, bandwidth $s$ set to the *median* pairwise distance — the standard heuristic that adapts the kernel to the data scale with no tuning. The first-order Stein identity, ridge-regularized, gives the score at the sample points as the solution of a linear system,
$$G = (K + \eta_G I)^{-1}\,\nabla K,$$
where $\nabla K[k,j] = -\sum_i (x_{k,j} - x_{i,j})\,K_{ik}/s^2$ is read straight off the kernel gradient and the small ridge $\eta_G = 10^{-3}$ keeps the solve well-conditioned. Differentiating the kernel twice gives $\nabla^2 K[k,j] = \sum_i \big(-1/s^2 + (x_{k,j}-x_{i,j})^2/s^4\big) K_{ik}$, and the second-order identity yields the diagonal Hessian as
$$H = -G^2 + (K + \eta_H I)^{-1}\,\nabla^2 K,$$
with $\eta_H = 10^{-3}$. The $-G^2$ term is not decoration: the second-order Stein solve estimates $\partial_{jj} p / p$, and since $\partial_{jj} p / p = \partial_{jj} \log p + (\partial_j \log p)^2$, subtracting the squared score estimate converts it into the genuine diagonal Hessian of $\log p$. Both solves are $O(n^3)$, entirely practical for the hundreds-to-few-thousand samples and tens of nodes in play. Two finite-sample refinements precede the variance: the columns of $H$ live on different scales (each $-1/\sigma_j^2$ plus child-driven variation), so I normalize each column by its mean before taking the variance, making the leaf criterion scale-fair; and variance is kept as the default dispersion measure, with a median-absolute-deviation alternative available for heavy-tailed cases. The leaf is then the $\arg\min$ over remaining columns of the normalized variance.

With the order recovered, edge selection is the well-trodden, regularized half I deliberately do not reinvent. Because a correct order's super-DAG already yields consistent intervention distributions, pruning is pure refinement, not correctness: for each node in order position, fit a flexible nonlinear regression on its predecessors and keep a parent only if it genuinely contributes. The canonical realization is CAM-style significance pruning — fit a generalized additive model of each node on its current parents and drop any parent whose covariate-significance test exceeds a small p-value (e.g. $10^{-3}$) — which has the screening property that true edges survive with high probability, lowering SHD and improving readability without being needed for correctness. The result is a fast, search-free, distributionally grounded recovery of the nonlinear causal DAG.

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
    nablaK = -np.einsum('kij,ik->kj', X_diff, K) / s ** 2  # (n, d)
    I = np.eye(n)
    G = np.linalg.solve(K + eta_G * I, nablaK)             # score estimate (n, d)
    nabla2K = np.einsum('kij,ik->kj',
                        -1.0 / s ** 2 + X_diff ** 2 / s ** 4, K)   # (n, d)
    return -G ** 2 + np.linalg.solve(K + eta_H * I, nabla2K)       # diagonal Hessian


def compute_topological_order(X, eta_G=1e-3, eta_H=1e-3):
    """Iterative leaf detection: leaf = min-variance diagonal-Hessian column."""
    data = X.copy()
    active = list(range(X.shape[1]))
    order = []
    while data.shape[1] > 1:
        H = stein_hessian_diagonal(data, eta_G, eta_H)
        H = H / H.mean(axis=0)                  # scale-fair normalization
        leaf = int(np.argmin(H.var(axis=0)))    # constant column => leaf
        order.append(active[leaf])
        active.pop(leaf)
        data = np.delete(data, leaf, axis=1)    # removing a leaf leaves an ANM
    order.append(active[0])
    order.reverse()                             # roots first
    return order


def run_causal_discovery(X, eta_G=1e-3, eta_H=1e-3, cutoff=1e-3):
    """SCORE: score-matching order recovery + CAM-style pruning.
    Returns adjacency B with B[i, j] != 0 meaning j -> i."""
    d = X.shape[1]
    order = compute_topological_order(X, eta_G, eta_H)
    B = np.zeros((d, d))
    for pos in range(1, len(order)):
        j = order[pos]
        parents = order[:pos]
        pvals = gam_significance(X[:, j], X[:, parents])   # CAM significance pruning
        for k, p in enumerate(parents):
            if pvals[k] <= cutoff:
                B[j, p] = 1.0                              # B[child, parent]=1: parent->child
    return B
```
