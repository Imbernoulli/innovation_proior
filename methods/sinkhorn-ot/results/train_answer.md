We want the cheapest way to morph one histogram into another while respecting the geometry of the bins. Given two probability vectors $r, c$ on the $d$-bin simplex $\Sigma_d$ and a ground cost matrix $M$ whose entry $m_{ij}$ is the cost of moving a unit of mass from bin $i$ to bin $j$, the transportation distance is the minimal Frobenius cost of a coupling whose marginals are $r$ and $c$,

$$d_M(r,c) = \min_{P \in U(r,c)} \langle P, M\rangle, \qquad U(r,c) = \{P \ge 0 : P\mathbf{1} = r,\; P^\top \mathbf{1} = c\}.$$

This is the only one of the usual simplex distances that is parameterized by a ground metric, so unlike Hellinger, $\chi^2$, KL, or total variation it can recognize that two nearby bins — synonyms in a bag of words, neighboring pixels in an image — are almost the same feature. That is exactly why it wins on high-dimensional histogram retrieval and classification. The trouble is cost. The feasible set $U(r,c)$ is the transportation polytope, and minimizing a linear functional over a polytope lands the optimum on a vertex, which is a sparse table with at most $2d-1$ nonzero entries. The combinatorial search for which $2d-1$ cells to light up is precisely what makes the LP hard: every exact solver for general $M$ — network simplex, the Rubner EMD code, Pele–Werman FastEMD — costs at least $O(d^3 \log d)$, and a single pair of histograms with a few hundred bins can take seconds, which puts the distance out of reach for any learning loop over thousands of pairs. The sub-cubic alternatives only buy their speed by restricting $M$ to special structure (an $\ell^1$/wavelet-tree embedding, thresholded metrics) and accepting an approximation error and a measurable drop in retrieval quality. And the vertex itself is a second disease, not just a symptom: it is near-deterministic and brittle, so a tiny nudge in $r$ can jump the optimum to a different vertex, leaving the objective piecewise-linear, non-differentiable, and impossible to put inside a gradient-based pipeline. So the question is whether we can get a transport-flavored distance for *arbitrary* $M$ that is orders of magnitude cheaper, smooth in its inputs, and trivially parallelizable, without giving up the metric structure.

I propose the Sinkhorn distance: entropy-regularized optimal transport. The move is to refuse the vertex. The cost and the brittleness are the same affliction — the LP is forced to a corner because its objective is flat — so instead of the extreme plan we ask for the cheapest *smooth* one. Smoothness of a coupling has an exact meter, its entropy $h(P) = -\sum_{ij} p_{ij}\log p_{ij}$. The vertex has tiny entropy; the smoothest coupling, ignoring cost entirely, is the independence table $rc^\top$ where $p_{ij} = r_i c_j$. The lever connecting them is the information inequality $h(P) \le h(r) + h(c)$, tight exactly at $P = rc^\top$, whose gap is the mutual information,

$$h(r) + h(c) - h(P) = \mathrm{KL}(P \,\|\, rc^\top) = I(X;Y) \ge 0.$$

High entropy thus means low mutual information, means close to independence. That gives a knob: restrict to couplings with $\mathrm{KL}(P\|rc^\top) \le \alpha$, a set $U_\alpha(r,c)$, and define $d_{M,\alpha}(r,c) = \min_{P \in U_\alpha(r,c)} \langle P, M\rangle$. This is the maximum-entropy principle made into a distance — for a given level of cost, prefer the least committal, most plausible plan. As $\alpha \to \infty$ the constraint becomes vacuous (every coupling satisfies $\mathrm{KL} \le \tfrac12(h(r)+h(c))$, so $U_\alpha = U$ and we recover $d_M$); at $\alpha = 0$ we are forced to $P = rc^\top$ and $d_{M,0} = r^\top M c$ in closed form. So it is a genuine interpolation between independence and exact EMD.

Before earning the right to make it fast, the regularized object must still be a distance. Symmetry is immediate from symmetric $M$. The triangle inequality is the real work, and the plain-transport proof through the gluing lemma carries over only if the glued plan stays inside the entropy ball. Take the optimal coupling $P$ of $(x,y)$ and $Q$ of $(y,z)$ and glue along $y$ via $t_{ijk} = p_{ij}q_{jk}/y_j$, with $S_{ik} = \sum_j t_{ijk}$. Direct summation gives $\sum_i S_{ik} = z_k$ and $\sum_k S_{ik} = x_i$, so $S \in U(x,z)$. Reading $t_{ijk}$ as the law of a triple, $p(X,Y,Z) = p(X)\,p(Y|X)\,p(Z|Y)$ is a Markov chain $X \to Y \to Z$, so the data-processing inequality gives $I(X;Z) \le I(X;Y) \le \alpha$; since $S$ is the $(X,Z)$ marginal, $S \in U_\alpha(x,z)$. The cost bound then follows from $m_{ik} \le m_{ij} + m_{jk}$ and $\sum_k q_{jk} = \sum_i p_{ij} = y_j$, yielding $d_{M,\alpha}(x,z) \le d_{M,\alpha}(x,y) + d_{M,\alpha}(y,z)$ for every $\alpha \ge 0$. (For small $\alpha$ the coincidence axiom fails, because forcing entropy keeps $d_{M,\alpha}(r,r) > 0$ when $h(r) > 0$; multiplying by $\mathbf{1}_{r\ne c}$ restores it if a true metric is needed.)

Now the computation. A hard $\mathrm{KL}$-ball is awkward, so move the entropy into the objective with a multiplier and *penalize* low-entropy plans instead of constraining them:

$$P^\lambda = \arg\min_{P \in U(r,c)} \langle P, M\rangle - \tfrac{1}{\lambda} h(P).$$

This already changes the character of the problem before we count a single operation. The negative entropy $-h(P) = \sum p_{ij}\log p_{ij}$ is *strictly* convex, so where the flat LP sat at a corner with possible ties, the penalized minimizer is **unique**, positive on every allowed cell, and smooth — the corner-jumping non-differentiability is gone. To find it, write the Lagrangian for the two marginal equalities with multiplier vectors $\alpha$ (rows) and $\beta$ (columns), and set the derivative in each $p_{ij}$ to zero:

$$\frac{\partial L}{\partial p_{ij}} = \tfrac{1}{\lambda}(\log p_{ij} + 1) + m_{ij} + \alpha_i + \beta_j = 0 \;\Longrightarrow\; p_{ij} = e^{-1}\, e^{-\lambda\alpha_i}\, e^{-\lambda m_{ij}}\, e^{-\lambda\beta_j}.$$

The $(i,j)$ dependence factors completely. Absorbing the stationary constant $e^{-1}$ into a one-dimensional factor (legitimate because the scalings are only identified up to reciprocal rescaling) and setting $u_i = e^{-\lambda\alpha_i}$, $K_{ij} = e^{-\lambda m_{ij}}$, the optimal plan is a fixed Gibbs kernel rescaled by one positive vector per side,

$$P^\lambda = \mathrm{diag}(u)\, K\, \mathrm{diag}(v), \qquad K = e^{-\lambda M}.$$

This is exactly a matrix-scaling object, and Sinkhorn & Knopp (1967) settled it: for strictly positive $K$ — and $K = e^{-\lambda M} > 0$ entrywise, so the support condition is free — there is a unique scaled matrix $\mathrm{diag}(u)K\,\mathrm{diag}(v)$ with prescribed row and column sums, with $u,v$ unique up to $u \to su,\; v \to v/s$. The combinatorial search for $2d-1$ active cells has evaporated; all that remains is to find the scalings. Writing the row-sum constraint $\mathrm{diag}(u)K\,\mathrm{diag}(v)\mathbf{1} = r$ and the column-sum constraint elementwise gives the coupled fixed point

$$u = r/(Kv), \qquad v = c/(K^\top u),$$

which we solve by alternation: fix $v$, set $u = r/(Kv)$; fix that $u$, set $v = c/(K^\top u)$; repeat. Each half-step is a single matrix–vector product plus an elementwise divide, so one iteration is $O(d^2)$ — no factorization, no pivoting, no bookkeeping. It converges because each half-step is a KL (Bregman) projection onto an affine marginal set: $u = r/(Kv)$ is the minimal row-scaling adjustment that fixes the row marginal, $v = c/(K^\top u)$ the analogous column projection, and alternating Bregman projections onto two affine sets converge to their intersection. The rate is geometric — the row-then-column map is a contraction in Hilbert's projective metric (Birkhoff / nonlinear Perron–Frobenius, made precise by Franklin & Lorenz), with ratio $< 1$ set by $K$ — so we get linear convergence cheaply for any $M$.

The multiplier $\lambda = 1/\varepsilon$ is an inverse temperature that slides the whole construction continuously. As $\lambda \to 0$, $K \to \mathbf{1}\mathbf{1}^\top$ and every scaling pushes toward the independence table $rc^\top$, with the distance degenerating to $r^\top M c$. As $\lambda \to \infty$, $K$ concentrates on the smallest-cost cells, $P^\lambda$ sharpens toward the LP vertex, and $d_M^\lambda \to d_M$. But there is a real reason not to crank $\lambda$: as $K$ becomes diagonally dominant the Hilbert-metric contraction ratio approaches $1$ so convergence slows, and the entries $e^{-\lambda m_{ij}}$ underflow to literal zeros, after which $Kv$ hits a zero and the divide blows up. Moderate $\lambda$ is therefore both faster and numerically safer, and it is precisely the regime where the distance is smooth and robust; when very small $\varepsilon$ is genuinely needed, the fix is to carry $\log u, \log v$ and do the products with a log-sum-exp. One more gift falls out of the form: everything is matrix–vector products against the *same* $K$, so the distance from one $r$ to a whole family $C = [c_1,\dots,c_N]$ runs by replacing the vector $v$ with a matrix, as $O(d^2 N)$ dense linear algebra that is embarrassingly parallel and GPU-friendly. I keep the theorem straight: the metric statement belongs to the hard-constrained $d_{M,\alpha}$ (with the $\mathbf{1}_{r\ne c}$ caveat), while the fixed-$\lambda$ quantity $d_M^\lambda$ is the fast smooth surrogate we actually compute, linked to $d_{M,\alpha}$ by a pair-dependent multiplier when the entropy constraint is active. Two practical details from the loop: drop any zero rows of the source and put zeros back into the returned plan, and for a whole matrix of targets compute the cost vector directly rather than materializing a $d\times d\times N$ stack of plans.

```python
import warnings
import numpy as np

def transport_plan(a, b, M, reg, num_iter=1000, stop_thr=1e-9, warn=True):
    """Entropy-regularized OT plan for one target histogram; reg = eps = 1/lambda."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    M = np.asarray(M, dtype=float)
    if b.ndim != 1:
        raise ValueError("transport_plan expects one target histogram; use transport_cost for many targets.")

    active = a > 0
    if not np.any(active):
        raise ValueError("source histogram has no positive mass")
    if not np.all(active):
        P = np.zeros_like(M, dtype=float)
        P[active, :] = transport_plan(a[active], b, M[active, :], reg, num_iter, stop_thr, warn)
        return P

    K = np.exp(-M / reg)                       # Gibbs kernel K = exp(-lambda M)

    u = np.ones(M.shape[0]) / M.shape[0]
    v = np.ones(M.shape[1]) / M.shape[1]
    Kp = (1.0 / a)[:, None] * K                # fold diag(1/a) into K for the u-update

    for it in range(num_iter):
        uprev, vprev = u.copy(), v.copy()
        KtU = K.T @ u
        v = b / KtU                            # column projection: enforce marginal b
        u = 1.0 / (Kp @ v)                     # row projection: enforce marginal a
        if KtU.min() == 0 or not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            if warn:
                warnings.warn(f"numerical errors at iteration {it}")
            u, v = uprev, vprev
            break
        if it % 10 == 0:
            marginal = v * (K.T @ u)           # (diag(u) K diag(v))^T 1
            if np.linalg.norm(marginal - b) < stop_thr:
                break

    return u[:, None] * K * v[None, :]         # P = diag(u) K diag(v)

def transport_cost(a, b, M, reg, num_iter=1000, stop_thr=1e-9, warn=True):
    """Return <P, M>. If b has target columns, return one cost per column."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    M = np.asarray(M, dtype=float)
    if b.ndim == 1:
        return float(np.sum(transport_plan(a, b, M, reg, num_iter, stop_thr, warn) * M))
    if b.ndim != 2:
        raise ValueError("target histogram must be a vector or a matrix with target columns")

    active = a > 0
    if not np.any(active):
        raise ValueError("source histogram has no positive mass")
    a = a[active]
    M = M[active, :]

    K = np.exp(-M / reg)
    u = np.ones((M.shape[0], b.shape[1])) / M.shape[0]
    v = np.ones((M.shape[1], b.shape[1])) / M.shape[1]
    Kp = (1.0 / a)[:, None] * K

    for it in range(num_iter):
        uprev, vprev = u.copy(), v.copy()
        KtU = K.T @ u
        v = b / KtU
        u = 1.0 / (Kp @ v)
        if np.any(KtU == 0) or not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            if warn:
                warnings.warn(f"numerical errors at iteration {it}")
            u, v = uprev, vprev
            break
        if it % 10 == 0:
            marginal = np.einsum("ik,ij,jk->jk", u, K, v)
            if np.linalg.norm(marginal - b) < stop_thr:
                break

    return np.einsum("ik,ij,jk,ij->k", u, K, v, M)


def stable_transport_plan(a, b, M, reg, num_iter=1000, stop_thr=1e-9):
    """Log-domain stabilization for one target when reg is small."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    M = np.asarray(M, dtype=float)
    if b.ndim != 1:
        raise ValueError("stable_transport_plan expects one target histogram")

    active = a > 0
    if not np.any(active):
        raise ValueError("source histogram has no positive mass")
    if not np.all(active):
        P = np.zeros_like(M, dtype=float)
        P[active, :] = stable_transport_plan(a[active], b, M[active, :], reg, num_iter, stop_thr)
        return P

    Mr = -M / reg
    loga = np.log(a)
    logb = np.full_like(b, -np.inf, dtype=float)
    logb[b > 0] = np.log(b[b > 0])
    u = np.zeros(M.shape[0])                    # log row scaling
    v = np.zeros(M.shape[1])                    # log column scaling

    def logsumexp(A, axis):
        m = A.max(axis=axis, keepdims=True)
        return (m + np.log(np.exp(A - m).sum(axis=axis, keepdims=True))).squeeze(axis)

    for it in range(num_iter):
        v = logb - logsumexp(Mr + u[:, None], axis=0)
        u = loga - logsumexp(Mr + v[None, :], axis=1)
        if it % 10 == 0:
            logP = Mr + u[:, None] + v[None, :]
            if np.linalg.norm(np.exp(logP).sum(axis=0) - b) < stop_thr:
                break
    return np.exp(Mr + u[:, None] + v[None, :])         # P = diag(e^u) K diag(e^v)
```
