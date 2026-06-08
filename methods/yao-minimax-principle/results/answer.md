# Yao's minimax principle

## Problem

Lower-bounding a randomized algorithm is hard because its cost on a fixed input is a random variable and its intrinsic cost is the expected cost on its *worst* input: the deterministic trick "exhibit one hard input" fails, since the algorithm can re-randomize so as to be fast on any single fixed input. We want a method that certifies a randomized lower bound while reasoning only about deterministic algorithms.

## Key idea

View algorithm design as a finite two-person zero-sum game. The designer chooses a deterministic algorithm (a row), an adversary chooses an input (a column), and the payoff is the cost r(A,x). A randomized algorithm is a mixed strategy for the designer (a distribution q over algorithms); a hard input distribution is a mixed strategy for the adversary (a distribution d over inputs). By von Neumann's minimax theorem the two players' values coincide, so the best randomized algorithm's worst-case expected cost equals the best deterministic algorithm's average cost against the worst input distribution. Hence to prove a randomized lower bound it suffices to exhibit one hard input distribution and bound deterministic algorithms on it.

## Setup and quantities

Finite decision-tree / comparison model. 𝒜 = deterministic algorithms; 𝒳 = inputs; r(A,x) = cost (probes / comparisons) of deterministic A on input x — the payoff matrix entry.

- Randomized algorithm R ≡ distribution q over 𝒜; expected cost on input x is E(R,x) = Σ_A q(A) r(A,x); worst-case cost max_x E(R,x).
- Input distribution d over 𝒳; average cost of deterministic A is C(A,d) = Σ_x d(x) r(A,x).
- **Randomized complexity** F₂ = inf_R max_x E(R,x).
- **Distributional complexity** F₁ = sup_d min_A C(A,d).

## The principle (errorless / Las Vegas)

**Theorem (equality).** F₂ = F₁. That is,

  inf_R max_{x} E(R,x) = sup_d min_{A} C(A,d).

Two-sided form: max_D min_{A} E_{x∼D}[c(A,x)] = min_R max_{x} E[c(R,x)].

**Proof.** The cost matrix r(A,x) defines a finite zero-sum game; the designer minimizes, the adversary maximizes. A randomized algorithm is the designer's mixed strategy, so F₂ = min_q max_x Σ_A q(A)r(A,x), the minmax value. An input distribution is the adversary's mixed strategy, so F₁ = max_d min_A Σ_x d(x)r(A,x), the maxmin value. Von Neumann's minimax theorem (1928) gives minmax = maxmin for any finite zero-sum matrix game (equivalently, strong LP duality: the optimal mixed strategies solve a primal/dual LP pair). Hence F₂ = F₁. ∎

**The easy direction, used directly for lower bounds (no minimax needed).** For any randomized R and any input distribution d,

  max_x E(R,x) ≥ Σ_x d(x) E(R,x) = Σ_A q(A) C(A,d) ≥ min_A C(A,d),

using max ≥ weighted-average, reordering the finite double sum, then average ≥ min. The right-hand side has no R, so it bounds F₂ = inf_R max_x E(R,x) from below.

## Lower-bound recipe

To lower-bound the best randomized algorithm's worst-case expected cost on a problem:

1. Model deterministic algorithms as rows, inputs as columns, cost r(A,x) as the payoff.
2. Exhibit **one** input distribution d (use any symmetry of the problem to make d uniform on isomorphism / relabelling classes — such a symmetric d is provably an optimal hard distribution; this also reduces F₁ to a small linear program). For selection problems the uniform distribution over all n! orderings is the hardest.
3. Prove **every** deterministic algorithm has average cost ≥ b under d — a deterministic argument, no coin-flips.
4. Conclude **every** randomized algorithm has worst-case expected cost ≥ b. In the errorless case this is tight (F₂ = F₁).

## Error (Monte Carlo) caveat

Allow the algorithm to err with probability ≤ λ. Let ε(A,x) ∈ {0,1} flag a wrong answer; q is "λ-tolerant" if sup_x Σ_A q(A) ε(A,x) ≤ λ. Define

  F_{1,λ} = sup_d min_{A : Σ_x d(x)ε(A,x) ≤ λ} C(A,d),  F_{2,λ} = inf over λ-tolerant R of max_x E(R,x).

The worst-case-on-every-input error constraint on q and the under-d error constraint on A are asymmetric, so the two feasible sets are not dual mixed-strategy simplices of one matrix game; von Neumann does not give equality. The surviving bound is one-sided:

  **F_{2,λ} ≥ ½ · F_{1,2λ},  for 0 ≤ λ ≤ ½.**

Reason for the factor and doubled error: fix a λ-tolerant q, any d, and T = max_x Σ_A q(A)r(A,x). Then Σ_A q(A)C(A,d) ≤ T and Σ_A q(A)err_d(A) ≤ λ. Therefore some deterministic A has C(A,d) ≤ 2T and err_d(A) ≤ 2λ: for λ,T > 0, average C(A,d)/(2T)+err_d(A)/(2λ); if λ = 0, supported trees have zero d-average error, and if T = 0, supported trees have zero d-average cost while the error average still gives err_d(A) ≤ 2λ. Thus min_{err_d(A)≤2λ} C(A,d) ≤ 2T for every d, so T ≥ ½F_{1,2λ}; inf over λ-tolerant q gives the bound.

The gap is genuine: finding a "mediocre" element (rank in [n/3, 2n/3]) needs 2n/3 comparisons for any deterministic algorithm in the worst case, but a randomized algorithm with error λ can sample O(log(1/λ)) elements and return their sample median using O(log(1/λ)) comparisons. The error probability decays exponentially in the sample size, so the order-of-magnitude advantage of randomization here lives entirely in the error-allowed regime and equality must fail there.

## Worked driver (small finite instance)

For a finite problem the principle is literally a pair of optimizations over the cost matrix; this computes F₁ and F₂ and checks that they coincide.

```python
import numpy as np
from scipy.optimize import linprog

# R[a, x] = cost (payoff) of deterministic algorithm a on input x.
# Designer minimizes (rows), adversary maximizes (columns).

def distributional_value(R):
    """F1 = max_d min_a  sum_x d[x] R[a,x]  (best deterministic algo vs worst input dist).
    LP: maximize v s.t. for every algorithm a:  sum_x d[x] R[a,x] >= v,  d a distribution."""
    m, n = R.shape                       # m algorithms, n inputs
    # variables: [d_1..d_n, v];  maximize v  ->  minimize -v
    c = np.concatenate([np.zeros(n), [-1.0]])
    # for each algorithm a:  v - sum_x d[x] R[a,x] <= 0
    A_ub = np.column_stack([-R, np.ones(m)])
    b_ub = np.zeros(m)
    # sum_x d[x] = 1
    A_eq = np.concatenate([np.ones(n), [0.0]])[None, :]
    b_eq = np.array([1.0])
    bounds = [(0, None)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    if not res.success:
        raise RuntimeError(res.message)
    return res.x[-1]                     # = v = F1

def randomized_value(R):
    """F2 = min_q max_x  sum_a q[a] R[a,x]  (best randomized algo, worst input).
    LP: minimize u s.t. for every input x:  sum_a q[a] R[a,x] <= u,  q a distribution."""
    m, n = R.shape
    c = np.concatenate([np.zeros(m), [1.0]])           # minimize u
    A_ub = np.column_stack([R.T, -np.ones(n)])          # sum_a q[a] R[a,x] - u <= 0
    b_ub = np.zeros(n)
    A_eq = np.concatenate([np.ones(m), [0.0]])[None, :] # sum_a q[a] = 1
    b_eq = np.array([1.0])
    bounds = [(0, None)] * m + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    if not res.success:
        raise RuntimeError(res.message)
    return res.x[-1]                                    # = u = F2

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    R = rng.integers(1, 9, size=(4, 5)).astype(float)   # 4 algorithms, 5 inputs
    f1, f2 = distributional_value(R), randomized_value(R)
    print("F1 (distributional) =", round(f1, 6))
    print("F2 (randomized)      =", round(f2, 6))
    # Yao / von Neumann: the two values coincide (up to LP tolerance).
    assert abs(f1 - f2) < 1e-6
    # Lower-bound recipe in action: the optimal hard distribution's value F1
    # is a valid (and here exact) lower bound on every randomized algorithm.
```
