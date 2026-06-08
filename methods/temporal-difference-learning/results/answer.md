# Temporal-Difference Learning (TD)

## Problem
Learn to predict a quantity that depends on the future evolution of an incompletely known dynamical system — the eventual outcome of a sequence (will this state lead to a win?) or a discounted sum of future signals — from a *stream* of experience, cheaply and from limited data. The natural supervised approach pairs each observation with the final outcome and runs the delta rule; but the outcome is available only at the end of the sequence (non-incremental, O(M) memory and peak compute) and is a single noisy sample of the expected value.

## Key idea
Drive learning from the difference between *temporally successive predictions* rather than from the difference between a prediction and the final outcome. The total supervised error telescopes exactly into a sum of one-step prediction changes,

  z − P_t = Σ_{k=t}^{m} (P_{k+1} − P_k),   with P_{m+1} ≝ z,

so the outcome-based update can be reorganized into incremental, change-by-change updates. Pushing this to its extreme replaces the noisy final outcome z by the next prediction P_{t+1} as the learning target (*bootstrapping*). Because the goal is to match the *expected* outcome, and the next prediction is an already-learned, lower-variance estimate of that expectation, bootstrapping is not merely cheaper but often a better use of the data.

## The TD(λ) family (linear predictor P_t = wᵀx_t, ∇P_t = x_t)
General update with an exponentially decaying eligibility trace:

  Δw_t = α (P_{t+1} − P_t) Σ_{k=1}^{t} λ^{t−k} ∇P_k,   0 ≤ λ ≤ 1,

with the trace e_t = Σ_{k=1}^{t} λ^{t−k} ∇P_k computed incrementally as **e_{t+1} = ∇P_{t+1} + λ e_t** (O(1) memory in sequence length).

- **TD(1)** (λ = 1): identical per-sequence weight change to the Widrow-Hoff / LMS rule Δw_t = α(z − wᵀx_t)x_t — but computed incrementally instead of waiting for z.
- **TD(0)** (λ = 0): Δw_t = α(P_{t+1} − P_t)∇P_t — the supervised rule with the actual outcome z replaced by the next prediction P_{t+1}.

## Discounted / cumulative case — the TD error
To predict z_t = Σ_{k=0}^{∞} γ^k c_{t+k+1} (0 ≤ γ < 1; γ = 1 for finite episodes), telescoping gives the Bellman consistency relation P_t = c_{t+1} + γ P_{t+1}, whose violation is the **temporal-difference error**

  **δ_t = c_{t+1} + γ P_{t+1} − P_t**   (equivalently δ = r + γV(s′) − V(s)),

with update Δw_t = α δ_t Σ_{k=1}^{t} (γλ)^{t−k} ∇P_k. Tabular TD(0): V(s_t) ← V(s_t) + α[r_{t+1} + γV(s_{t+1}) − V(s_t)].

## Why it works
- **Unification (samples *and* bootstraps).** Monte-Carlo/outcome targets sample but don't bootstrap (model-free, high variance, must wait for the end). Dynamic programming bootstraps but doesn't sample (needs the full model, sweeps the state space). The TD target r + γV(s′) samples one real successor *and* bootstraps off the current estimate there — model-free, incremental, lower variance.
- **Convergence.** For sequences from an absorbing Markov chain with linearly independent observation vectors, there is an ε > 0 such that for all 0 < α < ε, linear TD(0) converges in the mean to the ideal predictions (I − Q)^{-1}h. The proof uses the mean iteration I − αXᵀXD(I − Q). For B = D(I − Q), the symmetric part S = B + Bᵀ has S_ii = 2d_i(1 − p_ii) > 0 and S_ij = −d_i p_ij − d_j p_ji ≤ 0. Using dᵀ = μᵀ(I − Q)^{-1}, its row sums are d_i(1 − Σ_{j∈N}p_ij) + μ_i, giving diagonal dominance with a strict row in each retained connected component, hence S and B are positive definite. Then every eigenvalue μ = a + bi of XᵀXB has a > 0, and the iteration eigenvalue satisfies |1 − αμ|² = 1 − 2αa + α²(a² + b²) < 1 exactly when 0 < α < 2a/(a² + b²).
- **Optimality / better finite-data use.** Under repeated presentation of a finite training set, linear TD(0) converges to the **maximum-likelihood / certainty-equivalence** predictions (I − Q̂)^{-1}ĥ of the underlying Markov process, whereas Widrow-Hoff converges to the predictions that merely minimize RMS error on the training set. Example: from A→B→0 once and B→{1,1,1,1,1,1,0}, outcome-fitting gives V(A)=0 (the one A-trajectory ended in 0) while TD gives V(A)=V(B)=0.75 by using the A→B structure, which shows the difference between fitting raw returns and fitting the best Markov model.
- **Biological/computational naturalness.** δ is the real-time generalization of the Rescorla-Wagner conditioning rule ΔV_i = β(λ − V̄)·α_iX_i: promote λ from "reward this trial" to "reward next step + discounted next prediction," λ_{t+1} + γV̄_{t+1}. A prediction then acts as its own secondary reinforcer (resolving second-order conditioning), and δ behaves as a discrete-time derivative of the prediction — letting credit propagate backward along a chain without waiting for the final outcome.

## Implementation (linear TD(λ))
```python
import numpy as np

class TDLambda:
    """Linear TD(lambda): predictions P(x,w)=w·x learned from the difference
    between successive predictions (bootstrapping)."""

    def __init__(self, n_features, alpha=0.1, gamma=1.0, lam=0.0):
        self.w = np.zeros(n_features)
        self.alpha, self.gamma, self.lam = alpha, gamma, lam
        self.e = np.zeros(n_features)               # eligibility trace

    def predict(self, x):
        return float(self.w @ x)                    # linear: grad_w P = x

    def step(self, x, reward, x_next):
        v      = self.predict(x)
        v_next = 0.0 if x_next is None else self.predict(x_next)   # terminal: reward/cost carries outcome
        delta  = reward + self.gamma * v_next - v   # TD error: r + gamma*V(s') - V(s)
        self.e = self.gamma * self.lam * self.e + x # trace recurrence e <- gamma*lam*e + grad P
        self.w += self.alpha * delta * self.e       # one incremental update, O(1) memory in length
        return delta

    def end_episode(self):
        self.e = np.zeros_like(self.e)              # reset trace at sequence boundary

# lam=0 -> TD(0): w += alpha*(r + gamma*V(s') - V(s)) * x
# gamma=1 and lam=1 -> incremental Widrow-Hoff (the telescoped supervised rule)
# gamma=1 -> undiscounted episodic outcome prediction
```
