# Context

## Research question

Shannon's noisy-channel coding theorem gives the operational meaning of a single number: for a discrete memoryless channel (DMC) with input alphabet of size N, output alphabet of size M, and transition matrix Q with entries Q(j|i) = P(Y=j | X=i), the largest rate at which information can be sent with arbitrarily small error probability is the **channel capacity**

  C = max_p I(X;Y),   I(X;Y) = Σ_i Σ_j p_i Q(j|i) log( Q(j|i) / Σ_k p_k Q(j|k) ),

where p = (p_1,…,p_N) ranges over input probability distributions (the probability simplex). Dually, for a source X with reproduction alphabet X̂ and a per-symbol distortion measure d(x,x̂), the smallest rate at which the source can be encoded with average distortion at most D is the **rate-distortion function**

  R(D) = min_{ p(x̂|x): E[d(X,X̂)] ≤ D } I(X;X̂).

These two quantities are the boundary lines of what is achievable: capacity is the ceiling on reliable communication, R(D) is the floor on lossy compression. Both are defined as the optimum of mutual information over a set of distributions.

The question is computational. For a handful of symmetric channels (binary symmetric, the symmetric DMC, the erasure channel) and a handful of textbook sources (Bernoulli with Hamming distortion, Gaussian with squared error) the optimum can be written down in closed form. For an **arbitrary** Q or an **arbitrary** distortion matrix d, the optimum is a constrained optimization — maximize (or minimize) mutual information over a simplex — to be solved numerically. The task is to compute these optima for any finite channel or source.

## Background

**Mutual information and its shape.** The functional being optimized is the mutual information I(p,Q) = Σ_{i,j} p_i Q(j|i) log( Q(j|i)/q_j ), with the induced output distribution q_j = Σ_i p_i Q(j|i). Two facts about its shape are load-bearing. First, for a fixed channel Q, I(p,Q) is a **concave** function of the input distribution p. This is because I(p,Q) = Σ_i p_i [Σ_j Q(j|i) log Q(j|i)] − Σ_j q_j log q_j: the first term is linear in p, and the second is the negative of Σ_j q_j log q_j, which is convex in q (the function t log t is convex), composed with the linear map q = pQ. So capacity is the maximum of a concave function over the simplex — a convex program, solvable in principle. Second, R(D) is, dually, the minimum of a convex functional, so it too is a convex program.

**Optimality conditions.** Lagrange multipliers give the Karush–Kuhn–Tucker (KKT) conditions for the capacity program: at the capacity-achieving p*, the quantity Σ_j Q(j|i) log(Q(j|i)/q*_j) equals a common value (the capacity) for every input letter i with p*_i > 0, and is no larger for letters with p*_i = 0. The variables are coupled through the normalization q = pQ. Generic convex-optimization machinery — projected gradient ascent, interior-point methods on the KKT system — applies to a concave maximization over the simplex. The existing closed-form capacities come from exploiting **symmetry**: a symmetric channel is maximized by the uniform input, by a symmetry/convexity argument, giving C = log M − H(row of Q).

**Kullback–Leibler divergence and Gibbs' inequality.** The basic tool for comparing two distributions is the relative entropy D(p‖q) = Σ_i p_i log(p_i/q_i), which satisfies D(p‖q) ≥ 0 with equality iff p = q. The elementary proof uses ln x ≤ x − 1: −D(p‖q) = Σ_i p_i log(q_i/p_i) ≤ Σ_i p_i (q_i/p_i − 1) = Σ_i q_i − Σ_i p_i = 0. Mutual information is itself a divergence, I(X;Y) = D( p_{XY} ‖ p_X p_Y ).

**Rate-distortion theory.** Shannon's 1959 source-coding-with-fidelity result establishes that R(D) is the operational compression limit. Its Lagrangian form replaces the hard distortion constraint E[d] ≤ D with a penalty: for a slope parameter s > 0, minimize I(X;X̂) + s·E[d(X,X̂)]; sweeping s > 0 traces out the entire R(D) curve, with s the (negative) slope of the curve at the corresponding point. The minimization is again over a set of conditional distributions p(x̂|x), with the reproduction marginal r(x̂) = Σ_x p(x) p(x̂|x) determined by them.

**Coordinate ascent / alternating optimization.** A standard idea for an objective of two argument-blocks is to fix one block and optimize the other, then swap, and repeat. The geometric prototype is finding the minimum distance between two convex sets by alternately projecting.

## Baselines

**Symmetry-based closed forms.** For a symmetric DMC — every row of Q a permutation of every other row, every column likewise — the uniform input distribution achieves capacity by a symmetry/convexity argument, and C = log M − H(row of Q). For the binary symmetric channel with crossover β, C = log 2 − H(β). Core idea: exploit a group symmetry so that the maximizer is forced to the uniform distribution.

**Generic convex programming.** Because the capacity program is a concave maximization over the simplex (and R(D) a convex minimization), one can apply gradient ascent with simplex projection, or a Lagrangian/Newton interior-point solver to the KKT system. Core idea: treat it as a black-box convex problem; each gradient step uses a projection onto the simplex, and interior-point methods factor systems whose size grows with N·M.

**Hand analysis of the KKT conditions.** One can solve the stationarity conditions Σ_j Q(j|i) log(Q(j|i)/q_j) = γ (a constant) for all active i, together with q = pQ and Σ_i p_i = 1, by guessing the support and solving the coupled equations. Core idea: solve the optimality equations directly, letter by letter.

## Evaluation settings

The natural objects on which any such algorithm would be exercised are small, fully specified discrete channels and sources where the answer is independently known, so correctness can be checked:

- **Channels.** The binary symmetric channel with crossover β (capacity log 2 − H(β)); the binary erasure channel; small asymmetric DMCs with a given N×M transition matrix Q where capacity has no closed form and must be computed.
- **Sources / distortion.** The Bernoulli(p) source with Hamming distortion, whose rate-distortion function is the closed form R(D) = H(p) − H(D) for 0 ≤ D ≤ min(p,1−p); the Gaussian source with squared-error distortion, R(D) = ½ log(σ²/D); and general finite sources with an arbitrary distortion matrix d(x,x̂) where R(D) has no closed form.
- **Metrics / protocol.** Mutual information of the current iterate measured in bits (or nats); for rate-distortion, the (D, R) pair obtained at a given slope s, compared against the closed-form curve where one exists. The protocol is to initialize an input/reproduction distribution, iterate the update to convergence, and read off the limiting value.

## Code framework

The computational frame consists of probability-vector utilities, a mutual-information functional, KL divergence, and an iterate-until-converged loop. The update that maps the current input distribution to the next candidate distribution is left to be filled in.

```python
import numpy as np

def mutual_information(p, Q):
    # p: (N,) input distribution; Q: (N,M) channel, Q[i,j] = P(y=j | x=i)
    q = p @ Q                      # induced output distribution, (M,)
    nz = Q > 0
    # I = sum_{i,j} p_i Q_ij log(Q_ij / q_j)
    ratio = np.zeros_like(Q)
    ratio[nz] = np.log(Q[nz] / q[np.where(nz)[1]])
    return float((p[:, None] * Q * ratio).sum())

def kl_divergence(a, b):
    # D(a || b) = sum_i a_i log(a_i / b_i)
    nz = a > 0
    return float((a[nz] * np.log(a[nz] / b[nz])).sum())

def capacity_update(p, Q):
    # TODO
    pass

def compute_capacity(Q, max_iter=1000, tol=1e-9):
    N = Q.shape[0]
    p = np.full(N, 1.0 / N)        # agnostic, full-support start
    for _ in range(max_iter):
        p_new = capacity_update(p, Q)
        # TODO: stopping rule
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new
            break
        p = p_new
    return mutual_information(p, Q), p
```
