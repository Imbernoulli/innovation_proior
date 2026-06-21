We are given $n$ i.i.d. observations of a $d$-dimensional vector stacked into a data matrix $X \in \mathbb{R}^{n \times d}$, and we want to recover the directed acyclic graph (DAG) behind the joint distribution. Modeling the data by a linear structural equation model, each variable is a linear function of its parents plus independent noise, $X_j = w_j^\top X + z_j$, so the weighted adjacency matrix $W \in \mathbb{R}^{d \times d}$ — with column $j$ holding the incoming weights of node $j$, and $W_{ij} \neq 0$ meaning an edge $i \to j$ — *is* the graph. Fitting $W$ to the data is genuinely easy: the natural fit measure is the squared reconstruction error of regressing every variable on the others at once, $\ell(W) = \tfrac{1}{2n}\|X - XW\|_F^2$, a smooth function with gradient $-\tfrac{1}{n}X^\top(X - XW)$, and we have full statistical cover for it — the least-squares minimizer recovers a true DAG with high probability even in high dimensions, for Gaussian *and* non-Gaussian noise alike, with no faithfulness assumption. The statistics, in short, are settled. The entire difficulty lives in one place: the requirement that the graph $W$ induces be acyclic.

That requirement is brutal in a specific way. Acyclicity is a property of the *support pattern* of $W$ — a combinatorial property of a discrete object — and the set of DAGs on $d$ nodes grows superexponentially in $d$, so optimizing a score over it is NP-hard. Every existing method is therefore some clever way of moving around inside the discrete set of acyclic graphs without ever leaving it: exact integer-program solvers enumerate parent sets and are globally optimal but die past a few dozen nodes; greedy methods like greedy equivalence search add or delete one edge at a time and re-check for cycles, which is fast only when each node has a handful of parents and collapses on dense hub graphs; order-based methods fix a topological ordering and search the $d!$ orderings instead. All of them enforce acyclicity *operationally* by staying in the legal region of a discrete space, and all of them pay for it — with poor scaling, with bounded-in-degree assumptions that real scale-free networks violate, or with bespoke graphical-model machinery that demands expertise to implement. None is "write down an objective and call a solver." That is exactly what transformed the two neighboring problems: undirected structure learning became a convex log-determinant program (the graphical lasso), and deep nets are SGD on a differentiable loss. The directed-acyclic case never got this, because its acyclicity constraint has no smooth closed form. So the question sharpens to a single concrete target: find a function $h(W)$ on real matrices — smooth, with a cheap gradient — that is exactly zero when $W$'s graph is a DAG and nonzero otherwise. Then $\min_W F(W)$ subject to $h(W) = 0$ is an ordinary smooth equality-constrained problem, and the whole discrete-search apparatus can be thrown away.

I propose NOTEARS — Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for Structure learning — and its core is precisely that function $h$. The construction starts from an elementary algebraic fact. For a binary adjacency matrix $B$, the entry $(B^k)_{ii}$ counts the number of closed walks of length $k$ that leave node $i$ and return to it, so $\operatorname{tr}(B^k) = \sum_i (B^k)_{ii}$ is the total number of length-$k$ closed walks. A directed graph is acyclic exactly when it has no closed walks of any length, so acyclicity is equivalent to $\operatorname{tr}(B^k) = 0$ for every $k \geq 1$. This already converts the combinatorial property into an algebraic one; what remains is to bundle "all these traces vanish" into a single smooth scalar. The obvious first try sums them via the Neumann series, $\sum_{k\geq 0} B^k = (I-B)^{-1}$, giving $\operatorname{tr}((I-B)^{-1}) = d + \sum_{k\geq 1}\operatorname{tr}(B^k)$, so $B$ is a DAG iff $\operatorname{tr}((I-B)^{-1}) = d$ — but the Neumann series converges only when the spectral radius $r(B) < 1$, and along the optimization path $B$ need not be acyclic, $r(B)$ can exceed $1$, and the inverse blows up; maintaining $r(B) < 1$ is itself a hard constraint to project onto. Retreating to the *finite* sum $\sum_{k=1}^{d}\operatorname{tr}(B^k) = 0$ (valid for all $B$, since a DAG has no cycle longer than $d$) removes the convergence issue but introduces a numerical one: the number of length-$k$ walks grows like (average degree)$^k$, so by $k = d$ the entries of $B^k$ overflow floating point and both value and gradient become garbage. The lesson of these two walls is that I need a *reweighting* of the trace counts that keeps every term nonnegative, converges for all matrices, and shrinks the high-$k$ terms hard.

Reweighting $\operatorname{tr}(B^k)$ by $1/k!$ does all three at once, because $\sum_{k\geq 0}\operatorname{tr}(B^k)/k! = \operatorname{tr}(e^B)$, the matrix exponential, which converges for *every* square matrix since the factorial dominates any fixed matrix's growth. The terms stay individually nonnegative for binary $B$, so

$$\operatorname{tr}(e^B) = d + \sum_{k\geq 1}\frac{\operatorname{tr}(B^k)}{k!} \quad\Longrightarrow\quad B \text{ is a DAG} \iff \operatorname{tr}(e^B) = d,$$

with no spectral-radius condition and the magnitude explosion defused by the factorial. The matrix exponential is moreover a workhorse of numerical linear algebra with an $O(d^3)$ scaling-and-squaring algorithm in every scientific library, so cheapness is essentially free. But $B$ is binary and I need $h$ on *real* $W$ to take gradients, and I cannot simply write $\operatorname{tr}(e^W) = d$: the nonnegativity-of-terms argument used that $B$'s entries are nonnegative, whereas for signed $W$ the diagonal walk contributions can cancel and a cyclic graph could hit $d$ by accident. The fix is to make the argument of the exponential entrywise nonnegative while preserving the support of $W$ — and the Hadamard square $W \circ W$, with entries $w_{ij}^2 \geq 0$ that are zero exactly where $W$ is zero, does precisely that. Running the closed-walk argument on the nonnegative weighted matrix $W \circ W$ (each closed walk now contributing a product of squared weights, still nonnegative) gives the function

$$h(W) = \operatorname{tr}\!\left(e^{W \circ W}\right) - d, \qquad W \text{ is a DAG} \iff h(W) = 0.$$

This $h$ is smooth (a composition of the entrywise square, the matrix exponential, and the trace), it is nonnegative everywhere so the DAGs are exactly its global minima, and it does not merely indicate acyclicity but *quantifies* it: $h(W)$ is the $1/k!$-reweighted sum of weighted closed walks, larger when there are more cycles or heavier cycles, which is exactly the graded signal a continuous solver can descend rather than a flat indicator. Its gradient comes from one chain-rule step: $\partial\operatorname{tr}(e^M)/\partial M = (e^M)^\top$ (the transpose is load-bearing because $M = W \circ W$ need not be symmetric), and with $M = W \circ W$ the inner derivative $\partial M_{kl}/\partial W_{ij}$ is $2W_{ij}$ when $(k,l) = (i,j)$ and zero otherwise, so

$$\nabla h(W) = \left(e^{W \circ W}\right)^\top \circ\, 2W,$$

which reuses the very same matrix exponential — value and gradient together cost a single $e^{W\circ W}$ evaluation.

The program I have been chasing now exists: minimize $F(W) = \tfrac{1}{2n}\|X - XW\|_F^2 + \lambda\|W\|_1$ subject to $h(W) = \operatorname{tr}(e^{W\circ W}) - d = 0$, a smooth score with a smooth scalar equality constraint over real-matrix variables. It is nonconvex (the feasible set is nonconvex), so I am honest that it yields stationary points, not certified global optima — but it is solvable by standard machinery. The right tool for a single smooth equality constraint is the augmented Lagrangian, $L^\rho(W,\alpha) = F(W) + \tfrac{\rho}{2}h(W)^2 + \alpha\,h(W)$. A plain quadratic penalty would enforce $h = 0$ only as $\rho \to \infty$, which wrecks conditioning; the multiplier term lets $\alpha$ converge to the correct Lagrange multiplier and drive $h$ to zero with $\rho$ kept *finite*. Because the constraint is the single scalar $h$, the envelope theorem makes the dual derivative just the constraint value at the inner optimizer, $\nabla D(\alpha) = h(W^*_\alpha)$, so dual ascent is

$$\alpha \leftarrow \alpha + \rho\, h(W^*_\alpha),$$

with $\rho$ doubling as the step size. The outer loop alternates this multiplier update with an inner solve, and $\rho$ is multiplied by $10$ only when an inner solve fails to shrink infeasibility below $0.25\, h_{\text{prev}}$ — raising the penalty only when feasibility actually stalls.

The inner subproblem, with $\rho, \alpha$ fixed, is to minimize $f(w) + \lambda\|w\|_1$ over $w = \operatorname{vec}(W)$, where $f$ collects the smooth parts and its $W$-gradient is $G_{\text{smooth}} = \nabla\ell + (\rho h + \alpha)\nabla h$ (the coefficient $\rho h + \alpha$ on $\nabla h$ comes from differentiating the $\tfrac{\rho}{2}h^2 + \alpha h$ terms). The principled route is proximal quasi-Newton — a quadratic L-BFGS model of the smooth part with the exact $\ell_1$ term, whose coordinate updates have the closed-form soft-threshold $z^* = -c + S(c - b/a,\, \lambda/a)$ — and I reach for a *second-order* inner method deliberately, because each evaluation of $f$ requires an $O(d^3)$ matrix exponential, so I want the fewest, most informative steps. But the $\ell_1$ term has structure that lets me avoid composite optimization entirely: split each variable into nonnegative positive and negative parts, $W = W^+ - W^-$ with $W^+, W^- \geq 0$. At any optimum at most one side of a coordinate is positive (otherwise subtract their minimum from both, leaving $W^+ - W^-$ unchanged while strictly reducing $W^+ + W^-$), so $\|W\|_1 = \mathbf{1}^\top(W^+ + W^-)$ becomes *linear*, and the subproblem turns into a smooth objective with simple bound constraints, handed directly to L-BFGS-B with no proximal machinery — at the cost of merely doubling the variable count from $d^2$ to $2d^2$. In the doubled variables the gradient is $+G_{\text{smooth}} + \lambda$ on the $w^+$ block and $-G_{\text{smooth}} + \lambda$ on the $w^-$ block, and I pin the diagonal box constraints to $(0,0)$ so the model can never place a self-loop.

After the outer loop converges, $h(\tilde W)$ sits at machine tolerance ($\leq 10^{-8}$) rather than exactly zero, so a few tiny spurious weights may technically induce a cycle; I hard-threshold them away, zeroing every entry with $|\tilde W_{ij}| \leq \omega$. This is justified both statistically (hard thresholding of regression estimates provably reduces false discoveries) and structurally (because $h$ quantifies cyclicity, a near-feasible solution's residual cyclic mass lives precisely in tiny edges). A fixed $\omega = 0.3$ works as a default; with $\lambda = 0$ and $\omega$ fixed, no tuning is needed beyond solver tolerances. The remaining numerical constants are: start $\rho = 1$, $\alpha = 0$, $h = \infty$; cap outer iterations at $100$ and $\rho$ at $10^{16}$; declare feasibility at $h \leq 10^{-8}$; use the progress factor $0.25$ and the $\times 10$ penalty multiplier; and center the columns of $X$ for the least-squares loss so the intercept does not leak into the edge weights. The same smooth wrapper carries logistic and Poisson node-wise losses without touching the acyclicity machinery; the demo call uses $\lambda = 0.1$ with the $\ell_2$ loss.

```python
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid


def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8,
                   rho_max=1e16, w_threshold=0.3):
    """min_W L(W;X) + lambda1*||W||_1  s.t.  h(W) = tr(e^{W∘W}) - d = 0,  augmented Lagrangian.
    Returns W_est [d, d]; W_est[i, j] != 0 means edge i -> j."""
    n, d = X.shape

    def _loss(W):
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / n * (R ** 2).sum()
            G_loss = -1.0 / n * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / n * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / n * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / n * (S - X * M).sum()
            G_loss = 1.0 / n * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        # smooth acyclicity h(W) = tr(e^{W∘W}) - d; one matrix exponential gives value + gradient
        E = slin.expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * W * 2                       # ∇h = (e^{W∘W})^T ∘ 2W
        return h, G_h

    def _adj(w):
        # doubled variables [w_pos, w_neg] (each d^2) -> W = w_pos - w_neg
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()   # l1 = 1^T(w_pos + w_neg)
        G_smooth = G_loss + (rho * h + alpha) * G_h                       # coeff (rho*h + alpha) on ∇h
        g_obj = np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)
        return obj, g_obj

    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)          # center columns
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:                            # feasibility stalled -> raise penalty
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h                                    # dual ascent
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0                 # round to a clean DAG
    return W_est
```
