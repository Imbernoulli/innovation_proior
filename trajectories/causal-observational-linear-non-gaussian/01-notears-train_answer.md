The harness leaves one small function open: take the data matrix $X$ and return an adjacency matrix $B$ whose support is the recovered DAG, with the convention $B[i,j]\neq 0 \Rightarrow j\to i$. The data come from a linear structural equation model — each variable is a signed linear combination of its parents plus its own independent noise, $x_i = \sum_j w_{ji}\,x_j + e_i$ — so if I stack the incoming-weight vectors into a matrix $W$ whose column $j$ holds the weights flowing into node $j$, then the *support* of $W$ literally is the graph, and fitting it is a plain least-squares problem: minimize $\ell(W) = \tfrac{1}{2n}\lVert X - XW\rVert_F^2$, a smooth convex score with gradient $-\tfrac{1}{n}X^\top(X-XW)$, whose minimizer is known to recover a true DAG with high probability, in high dimensions, with or without Gaussian noise. The single thing standing between that trivial fit and the answer is the requirement that the graph $W$ induces be **acyclic** — and acyclicity is a combinatorial property of a discrete object whose count grows superexponentially in the number of nodes. Every classical method is some way of moving *inside* the discrete set of acyclic graphs without leaving it: exact parent-set solvers (globally optimal, dead past a few dozen nodes), greedy equivalence search adding or deleting one edge at a time (fast only when each node has a handful of parents, which these dense, hub-heavy benchmark graphs violate), order-based search over the $d!$ topological orderings. All of them inherit that discrete region's curse — bad scaling or a bounded-in-degree assumption — and none is "write down an objective and call a solver." I am deliberately starting here, with a method that *ignores* the non-Gaussianity the whole benchmark is built around, because I want a strong, general-purpose floor from a different method family so that whatever the later non-Gaussian rungs add is measured against a real competitor rather than against nothing.

I propose **NOTEARS-linear**: turn acyclicity from a combinatorial constraint on a discrete object into a single smooth scalar equality on real matrices, so the whole problem becomes an ordinary continuous program $\min_W \ell(W)$ subject to $h(W)=0$ that a black-box solver can handle. What makes this possible is an algebraic handle on acyclicity. Forget weights for a moment and take a binary adjacency $B$: the elementary fact is that $(B^k)_{ii}$ counts the closed walks of length $k$ from node $i$ back to itself, so $\mathrm{tr}(B^k)$ is the total number of length-$k$ closed walks, and a directed graph is acyclic *precisely* when it has no closed walk of any length — i.e. $\mathrm{tr}(B^k)=0$ for every $k\ge 1$. I need to package "all of these vanish" into one smooth nonnegative scalar that I can take a gradient of. The naive packagings fail: summing the raw traces $\sum_k \mathrm{tr}(B^k)$ has the closed form $\mathrm{tr}((I-B)^{-1})$ via the Neumann series, but that series only converges when the spectral radius is below one, which a non-DAG iterate along the optimization path easily violates, and then $I-B$ can be singular; the finite version $\sum_{k=1}^d \mathrm{tr}(B^k)$ fixes convergence but the walk counts grow like $(\text{degree})^k$ and overflow floating point by $k=d$. The fix is to *reweight* each trace by $1/k!$, because then
$$\sum_{k\ge 0}\frac{\mathrm{tr}(B^k)}{k!} = \mathrm{tr}(e^B),$$
the matrix exponential, which converges for **every** square matrix (the factorial dominates any fixed matrix's growth), keeps every term nonnegative for binary $B$, and shrinks the dangerous high-$k$ terms hard. So $\mathrm{tr}(e^B) = d + \sum_{k\ge 1}\mathrm{tr}(B^k)/k!$ equals $d$ exactly when every trace vanishes — $B$ is a DAG $\iff \mathrm{tr}(e^B)=d$ — and $e^B$ is a workhorse every scientific library computes in $O(d^3)$ by scaling-and-squaring.

That argument used $B$'s nonnegativity (walk counts are nonnegative), but I need $h$ on *signed* real $W$, where $(W^k)_{ii}$ can be negative and terms can cancel, so $\mathrm{tr}(e^W)=d$ no longer forces acyclicity. The repair is to replace $W$ by its entrywise square: the Hadamard product $W\circ W$ is nonnegative everywhere and has the *same support* as $W$ (the same graph), so running the binary argument on it gives the smooth acyclicity function
$$h(W) = \mathrm{tr}\!\big(e^{\,W\circ W}\big) - d,$$
which is nonnegative, zero iff $W$'s graph is acyclic, and — because it is the factorial-reweighted total of weighted closed walks — automatically *quantifies* how cyclic the graph is, which is exactly what a continuous solver needs to descend rather than chase a flat indicator. Its gradient is one line: $\partial\,\mathrm{tr}(e^M)/\partial M = (e^M)^\top$ and $M=W\circ W$ gives $\partial M_{kl}/\partial W_{ij} = 2W_{ij}$ on the diagonal, so $\nabla h(W) = (e^{W\circ W})^\top \circ 2W$ — the same matrix exponential, Hadamard-multiplied by $2W$ and transposed (the transpose matters because $W\circ W$ need not be symmetric). A quick check: the path $1\to2\to3$ has $h=0$, and closing it into a 3-cycle by adding $3\to1$ makes $\mathrm{tr}(B^3)=3>0$ so $h>0$.

With $h$ in hand the program is $\min_W \tfrac{1}{2n}\lVert X-XW\rVert^2 + \lambda\lVert W\rVert_1$ subject to $h(W)=0$ — the whole matrix updated at once (global, not edge-by-edge), the score distribution-agnostic, no in-degree assumption. It is not convex (the constraint set is nonconvex), so I reach stationary points, not certified optima, like every non-exact method. The right tool for a single smooth equality constraint is the **augmented Lagrangian**: a pure penalty $\ell + \tfrac{\rho}{2}h^2$ only enforces $h=0$ as $\rho\to\infty$, and large $\rho$ is badly conditioned, so I add a multiplier, $L^\rho(W,\alpha) = \ell(W) + \tfrac{\rho}{2}h(W)^2 + \alpha\,h(W)$, where $\alpha$ drives $h$ to zero without sending $\rho$ to infinity. By the envelope theorem the dual gradient is $\nabla D(\alpha)=h(W^\star_\alpha)$, so dual ascent is $\alpha\leftarrow\alpha+\rho\,h(W^\star_\alpha)$ with $\rho$ doubling as the step size; the outer loop solves the inner problem, nudges $\alpha$, and raises $\rho$ (by $10\times$) only when feasibility stalls — when the new infeasibility is not below $0.25\,h_{\text{old}}$. The inner subproblem minimizes the smooth $f = \ell + \tfrac{\rho}{2}h^2 + \alpha h$ plus the nonsmooth $\lambda\lVert W\rVert_1$; I keep it in plain smooth optimization by splitting each variable into nonnegative positive and negative parts $W=W^+ - W^-$ (at any optimum at most one side is positive, so $\lVert W\rVert_1 = \mathbf{1}^\top(W^+ + W^-)$ becomes linear), handing the resulting bound-constrained smooth problem to L-BFGS-B — a quasi-Newton method chosen because each evaluation needs an $O(d^3)$ matrix exponential and I want the fewest, most informative steps. The smooth gradient is $G_{\text{smooth}} = \nabla\ell + (\rho h+\alpha)\nabla h$, becoming $+G_{\text{smooth}}+\lambda$ for the $W^+$ block and $-G_{\text{smooth}}+\lambda$ for the $W^-$ block; bounds pin the diagonal to exactly zero (no self-loops) and I center the columns of $X$ so the intercept does not leak into edge weights. After the outer loop converges, $h(\tilde W)$ is at machine tolerance — *almost* a DAG, with a handful of tiny spurious weights — so I round with a hard threshold, zeroing every entry with $\lvert\tilde W_{ij}\rvert\le\omega$ (justified because near-feasibility puts the residual cyclic mass in tiny edges, and hard-thresholding provably reduces false discoveries). One harness-specific detail is load-bearing: the reference implementation reads $W[i,j]\neq0$ as $i\to j$, but this task's convention is the opposite, $B[i,j]\neq0\Rightarrow j\to i$, so the returned adjacency must be **transposed** — that single line is the whole difference between a correct graph and a fully reversed one, and since the metric scores direction, getting it wrong would zero the score. The fixed constants are $\rho=1,\alpha=0,h=\infty$ to start, outer iterations capped at 100 and $\rho$ at $10^{16}$, feasibility at $h\le10^{-8}$, threshold $\omega=0.3$, and $\lambda=0.1$.

```python
def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)
    """
    import numpy as np
    import scipy.linalg as sla
    from scipy.optimize import minimize
    from sklearn.utils import check_array

    X = check_array(X)
    n, d = X.shape

    # Reference defaults from Zheng et al. 2018 reference impl.
    lambda1 = 0.1
    max_iter = 100
    h_tol = 1e-8
    rho_max = 1e16
    w_threshold = 0.3

    def _loss_and_grad(W):
        # Squared-error regression loss: 1/(2n) * ||X - X W||^2
        R = X - X @ W
        loss = 0.5 / n * (R ** 2).sum()
        G = -1.0 / n * X.T @ R
        return loss, G

    def _h_and_grad(W):
        # h(W) = tr(exp(W*W)) - d  (Zheng 2018 smooth acyclicity)
        M = W * W
        E = sla.expm(M)
        h = np.trace(E) - d
        G = E.T * 2 * W
        return h, G

    def _obj(w_pm, rho, alpha):
        w_pm = w_pm.reshape(2, d * d)
        W = (w_pm[0] - w_pm[1]).reshape(d, d)
        loss, G_loss = _loss_and_grad(W)
        h, G_h = _h_and_grad(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w_pm.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g = np.concatenate([
            (G_smooth + lambda1).flatten(),
            (-G_smooth + lambda1).flatten(),
        ])
        return obj, g

    w_est = np.zeros(2 * d * d)
    rho, alpha, h = 1.0, 0.0, np.inf
    # Non-negative bounds; force diagonal to zero (no self-loops)
    bnds = [(0, 0) if (i == j) else (0, None)
            for _ in range(2) for i in range(d) for j in range(d)]

    for _ in range(max_iter):
        while rho < rho_max:
            sol = minimize(_obj, w_est, args=(rho, alpha),
                           method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            W_new = (w_new[:d * d] - w_new[d * d:]).reshape(d, d)
            h_new, _ = _h_and_grad(W_new)
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_final = (w_est[:d * d] - w_est[d * d:]).reshape(d, d)
    W_final[np.abs(W_final) < w_threshold] = 0.0

    # NOTEARS: W[i, j] != 0 means i -> j
    # causal-learn / this task: B[i, j] != 0 means j -> i. Transpose.
    return W_final.T
```
