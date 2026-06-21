Shannon's channel capacity C = max_p I(X;Y) and rate-distortion function R(D) = min I(X;X̂) are defined as optima of mutual information over probability distributions. For symmetric channels or simple sources we can write the answer in closed form, but for an arbitrary discrete channel matrix Q or distortion matrix d there is no formula. One is left with a constrained convex program over the simplex. Generic solvers such as projected gradient ascent or interior-point methods can handle it in principle, but they require projection steps or large linear-system solves, they ignore the log-sum structure of mutual information, and they give no cheap certificate of how far the current iterate is from the optimum.

The real obstacle is that maximizing I(p,Q) over the input distribution p directly has no closed-form step. The KKT conditions are implicit: the optimal support is unknown and the stationarity equations couple all variables through the output distribution q = pQ. We need an iterative scheme that exploits the special structure of mutual information and supplies both monotone improvement and a computable gap bound.

The method is the Arimoto–Blahut algorithm. It is based on a variational, or double-optimization, representation of mutual information. Rewrite the ratio inside the logarithm as Q(j|i)/q_j = φ*(i|j)/p_i, where φ*(i|j) = p_i Q(j|i)/q_j is the Bayes posterior of the input given the output. If we promote this posterior to a free reverse channel φ(i|j), then I(p,Q) = max_φ Ĩ(p,Q,φ), where Ĩ(p,Q,φ) = Σ_i,j p_i Q(j|i) log(φ(i|j)/p_i). The gap I − Ĩ is a weighted sum of KL divergences between the true posterior and φ, so it is zero exactly when φ equals the Bayes posterior. Capacity therefore becomes a double maximization C = max_p max_φ Ĩ(p,Q,φ). The crucial point is that each inner maximization is closed form. Fixing p and optimizing over φ gives the posterior by Bayes' rule. Fixing φ and optimizing over p gives a linear-plus-entropy objective whose maximizer is a Gibbs distribution, p_i ∝ exp(Σ_j Q(j|i) log φ(i|j)).

Alternating between these two closed-form updates makes the mutual information non-decreasing: the φ-step raises Ĩ to I(p^t,Q), and the p-step can only raise it further. Folding the two half-steps together yields a particularly simple multiplicative update on p. Let q^t = p^t Q be the current output distribution and let D(Q_i‖q^t) be the KL divergence from row i of Q to q^t. Then p_i^{t+1} ∝ p_i^t exp(D(Q_i‖q^t)). Input letters whose output distributions are most distinguishable from the average output are up-weighted; letters that look like the crowd are suppressed. Because the capacity program is concave, any limit point of this iteration satisfies the KKT conditions and is therefore the global optimum.

A second benefit is a built-in stopping certificate. At every iterate one has the sandwich I(p^t,Q) ≤ C ≤ max_i D(Q_i‖q^t). The upper bound follows from writing C as Σ_i p*_i D(Q_i‖q^t) − D(q*‖q^t) and dropping the non-positive KL term. So the gap max_i D(Q_i‖q^t) − I(p^t,Q) is a computable error bar, and the algorithm can stop once it falls below a prescribed tolerance.

The same construction works for rate distortion, with signs flipped. For a slope s > 0, minimize I(X;X̂) + s E[d(X,X̂)] by alternating between the reproduction marginal r(x̂) and the test channel p(x̂|x). The updates are p(x̂|x) ∝ r(x̂) exp(−s d(x,x̂)) and r(x̂) = Σ_x p(x) p(x̂|x). Sweeping s traces the entire R(D) curve. Both capacity and rate-distortion versions are instances of alternating information projection between convex families of distributions, which is why monotone global convergence is guaranteed.

```python
import numpy as np

def _kl_rows_to_vec(Q, q):
    """D(Q_i || q) for each row i of Q; shape (N,)."""
    nz = Q > 0
    ratio = np.zeros_like(Q)
    ratio[nz] = np.log(Q[nz] / q[np.where(nz)[1]])
    return (Q * ratio).sum(axis=1)

def arimoto_blahut_capacity(Q, max_iter=1000, tol=1e-9):
    """Capacity (bits) and capacity-achieving input of a DMC. Q[i,j] = P(y=j|x=i)."""
    N = Q.shape[0]
    p = np.full(N, 1.0 / N)
    I_lower = 0.0
    for _ in range(max_iter):
        q = p @ Q
        d = _kl_rows_to_vec(Q, q)
        I_lower = float((p * d).sum())
        C_upper = float(d.max())
        if C_upper - I_lower < tol:
            break
        p = p * np.exp(d)
        p /= p.sum()
    return I_lower / np.log(2), p

def arimoto_blahut_rate_distortion(p_x, dist, s, max_iter=1000, tol=1e-9):
    """One R(D) point. p_x: source pmf; dist[x,xhat]=d(x,xhat); s>0 slope."""
    Lhat = dist.shape[1]
    p_x = p_x / p_x.sum()
    W = np.exp(-s * dist)
    p_cond = np.tile(p_x, (Lhat, 1)).T
    p_cond /= p_cond.sum(axis=1, keepdims=True)
    D_prev = np.inf
    R = D = 0.0
    for _ in range(max_iter):
        r = p_x @ p_cond
        p_cond = r * W
        p_cond /= p_cond.sum(axis=1, keepdims=True)
        ratio = np.log(p_cond / r[None, :])
        R = float((p_x[:, None] * p_cond * ratio).sum())
        D = float((p_x[:, None] * p_cond * dist).sum())
        if abs(D - D_prev) < tol:
            break
        D_prev = D
    return R / np.log(2), D
```
