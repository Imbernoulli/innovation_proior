# The Arimoto–Blahut algorithm

## Problem

Channel capacity C = max_p I(X;Y) of a discrete memoryless channel and the rate-distortion function R(D) = min_{p(x̂|x): E[d]≤D} I(X;X̂) are each the optimum of mutual information over a set of distributions. Except for symmetric special cases they have **no closed form**, so for an arbitrary channel matrix Q or distortion matrix d they must be computed numerically. The capacity program is a concave maximization over the probability simplex (R(D) a convex minimization), hence convex "in principle," but the simplex-constrained problem has no closed-form step and generic solvers give no cheap certificate of optimality.

## Key idea

Mutual information has a **variational (double-optimization) form**. Write the integrand ratio as Q(j|i)/q_j = φ*(i|j)/p_i, where for q_j > 0, φ*(i|j) = p_i Q(j|i)/q_j is the Bayes posterior of the input given the output, and q_j = Σ_i p_i Q(j|i). Outputs with q_j = 0 carry zero weight in the sums. Promoting the posterior to a free reverse channel φ(i|j) and defining

  Ĩ(p,Q,φ) = Σ_i Σ_j p_i Q(j|i) log( φ(i|j)/p_i ),

one gets I(p,Q) − Ĩ(p,Q,φ) = Σ_j q_j D(φ*(·|j)‖φ(·|j)) ≥ 0, so

  **I(p,Q) = max_φ Ĩ(p,Q,φ)**, maximized at φ = φ* (the posterior, for every output with q_j > 0).

Therefore C = max_p max_φ Ĩ(p,Q,φ) is a **double maximization**. Ĩ is concave in p for fixed φ and concave in φ for fixed p (biconcave, not jointly concave), and **both inner maxima are closed form**:

- Fix p → optimal φ is the posterior:  φ(i|j) = p_i Q(j|i) / Σ_k p_k Q(j|k).
- Fix φ → maximize Σ_i p_i a_i + H(p) with a_i = Σ_j Q(j|i) log φ(i|j); the Gibbs/softmax solution is p_i = r_i / Σ_k r_k with r_i = exp(a_i), and the optimal value is log Σ_i r_i.

**Alternating maximization** (Bayes-step, then softmax-step) makes I(p^t,Q) monotonically non-decreasing — because the φ-max gives Ĩ(p^t,Q,φ^{t+1}) = I(p^t,Q) and the p-max can only raise Ĩ further — and, being bounded by C, it converges. Folding the two half-steps together (q^t = p^t Q):

  **p_i^{t+1} = p_i^t · exp( D(Q_i ‖ q^t) ) / Σ_k p_k^t · exp( D(Q_k ‖ q^t) )**,

where Q_i is the i-th row of Q and D(Q_i‖q^t) = Σ_j Q(j|i) log(Q(j|i)/q_j^t). The update reinforces input letters whose output distribution is most distinguishable from the current average output.

**Stopping / convergence certificate.** For any iterate, with q = pQ,

  I(p,Q) ≤ C ≤ max_i D(Q_i ‖ q),

proved via C = Σ_i p*_i D(Q_i‖q) − D(q*‖q) ≤ max_i D(Q_i‖q). The gap max_i D(Q_i‖q) − I(p,Q) is a computable error bar; squeezing it to zero forces D(Q_i‖q*) = C for all active letters and ≤ C for inactive ones — exactly the KKT condition for the capacity-achieving input — so the fixed point is the **global** optimum of the concave program.

**Proximal view.** p^{t+1} = argmax_p ( Σ_i p_i D(Q_i‖q^t) − D(p‖p^t) ): a first-order model of I with a KL trust region — a mirror/proximal-ascent step, which is why no step size is needed.

## Rate-distortion analogue

For slope s > 0 (Lagrange multiplier; s relates to the curve slope), minimize I(X;X̂) + s·E[d] as a **double minimization** over the test channel p(x̂|x) and the reproduction marginal r(x̂). Both inner minima are closed form:

  p(x̂|x) = r(x̂) exp(−s d(x,x̂)) / Σ_{x̂'} r(x̂') exp(−s d(x,x̂')),   r(x̂) = Σ_x p(x) p(x̂|x).

Alternating minimization decreases the Lagrangian monotonically and converges; sweeping s > 0 traces R(D). The exp(−s d) factor keeps low-distortion reproductions and exponentially suppresses far ones.

Both algorithms are one instance of **alternating minimization of an I-divergence (KL) between two convex families** of distributions — the structure that guarantees monotone convergence to the global optimum.

## Implementation

```python
import numpy as np

def _kl_rows_to_vec(Q, q):
    """D(Q_i || q) for each row i of Q; shape (N,)."""
    nz = Q > 0
    ratio = np.zeros_like(Q)
    ratio[nz] = np.log(Q[nz] / q[np.where(nz)[1]])
    return (Q * ratio).sum(axis=1)

def blahut_arimoto_capacity(Q, max_iter=1000, tol=1e-9):
    """Capacity (bits) and capacity-achieving input of a DMC. Q[i,j] = P(y=j|x=i)."""
    N = Q.shape[0]
    p = np.full(N, 1.0 / N)                  # uniform, full-support start
    I_lower = 0.0
    for _ in range(max_iter):
        q = p @ Q                            # induced output q = pQ
        d = _kl_rows_to_vec(Q, q)            # D(Q_i || q) per input letter
        I_lower = float((p * d).sum())       # mutual information I(p,Q), in nats
        C_upper = float(d.max())             # certified upper bound on C
        if C_upper - I_lower < tol:          # sandwich gap < tol  =>  C pinned
            break
        p = p * np.exp(d)                    # multiplicative reweight ...
        p /= p.sum()                         # ... renormalized over the simplex
    return I_lower / np.log(2), p

def blahut_arimoto_rate_distortion(p_x, dist, s, max_iter=1000, tol=1e-9):
    """One R(D) point. p_x: source pmf; dist[x,xhat]=d(x,xhat); s>0 slope.
    Returns (rate in bits, average distortion)."""
    Lhat = dist.shape[1]
    p_x = p_x / p_x.sum()                       # normalize source weights
    W = np.exp(-s * dist)                     # closeness reweighting exp(-s d)
    p_cond = np.tile(p_x, (Lhat, 1)).T
    p_cond /= p_cond.sum(axis=1, keepdims=True)
    D_prev = np.inf
    R = D = 0.0
    for _ in range(max_iter):
        r = p_x @ p_cond                      # marginal r = sum_x p(x) p(xhat|x)
        p_cond = r * W                        # p(xhat|x) ~ r(xhat) exp(-s d)
        p_cond /= p_cond.sum(axis=1, keepdims=True)
        ratio = np.log(p_cond / r[None, :])
        R = float((p_x[:, None] * p_cond * ratio).sum())   # rate (nats)
        D = float((p_x[:, None] * p_cond * dist).sum())    # average distortion
        if abs(D - D_prev) < tol:
            break
        D_prev = D
    return R / np.log(2), D
```

Checks: on the binary symmetric channel with crossover β, `blahut_arimoto_capacity` returns 1 − H₂(β) bits; on the standard Z-channel check it returns 0.3219 bits; on a Bernoulli(p) source with Hamming distortion, sweeping s in `blahut_arimoto_rate_distortion` reproduces R(D) = H₂(p) − H₂(D) bits.
