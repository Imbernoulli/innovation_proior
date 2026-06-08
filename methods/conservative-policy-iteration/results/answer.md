# Conservative Policy Iteration (CPI)

## Problem

In an MDP too large to enumerate, approximate greedy policy iteration can *degrade* performance, and on-policy policy gradient can need sample complexity exponential in the state-space size. Both failures share a cause: performance is scored under the *current* policy's state-visitation distribution, which starves the states whose improvement is needed to reach optimality. CPI is a sample-based algorithm — using only a **restart distribution** μ (draw the next state from a fixed μ) — that improves a more uniform measure η_μ(π) = E_{s∼μ}[V_π(s)] with a **monotone, verifiable, per-step improvement guarantee** whose cost is polynomial in the halt tolerance ε and **independent of |S|**.

## Setup (normalized, γ-discounted)

V_π(s) = (1−γ)E[Σ_t γ^t R(s_t,a_t)] ∈ [0,R]; Q_π(s,a) = (1−γ)R(s,a) + γE_{s'}[V_π(s')]; A_π = Q_π − V_π ∈ [−R,R] (centered: Σ_a π(a|s)A_π(s,a)=0). Discounted future-state distribution d_{π,μ}(s) = (1−γ)Σ_t γ^t Pr(s_t=s; π,μ).

## Key idea

1. **Performance-difference lemma.** For any policies π̃, π and start distribution μ,
   η_μ(π̃) − η_μ(π) = (1/(1−γ)) E_{s∼d_{π̃,μ}} E_{a∼π̃(·|s)}[A_π(s,a)].
   The improvement rides the *new* policy's visitation d_{π̃,μ} — unobservable before committing. A large policy change moves d off d_{π,μ}, so advantages measured under the old distribution stop predicting the change. This is why greedy degrades.

2. **Policy advantage** (the measurable surrogate, under the old distribution):
   𝔸_{π,μ}(π') = E_{s∼d_{π,μ}} E_{a∼π'(·|s)}[A_π(s,a)],  with  ∂η_μ/∂α|_{α=0} = (1/(1−γ)) 𝔸_{π,μ}(π').

3. **Conservative mixture update.** π_new(a|s) = (1−α)π(a|s) + α π'(a|s). It keeps d_{π_new,μ} within O(α) of d_{π,μ}, demoting the off-distribution error to O(α²).

## Main result — improvement lower bound

**Theorem.** Let 𝔸 = 𝔸_{π,μ}(π') and ε_inf = max_s |E_{a∼π'(·|s)}[A_π(s,a)]|. For the mixture update and all α ∈ [0,1],
  η_μ(π_new) − η_μ(π) ≥ (α/(1−γ)) ( 𝔸 − 2αγε_inf/(1−γ(1−α)) ).
At α=1 this reduces to 𝔸/(1−γ) − 2γε_inf/(1−γ), recovering the approximate-greedy max-norm penalty shape.

*Proof.* Per state, E_{a∼π_new}[A_π(s,·)] = αΣ_a π'(a|s)A_π(s,a) = α A_π(s,π') (since Σ_a π A_π = 0). Following π_new, deviations to π' are i.i.d. with prob α; let c_t count them before time t, Pr(c_t=0)=(1−α)^t, ρ_t=1−(1−α)^t. Conditioned on c_t=0 the path is distributed as under π: Pr(s_t=s | π_new, c_t=0)=Pr(s_t=s | π). Hence
  E_{s∼Pr(s_t|π_new)}[A_π(s,π_new)] = α[(1−ρ_t)E_{c_t=0}[A_π(s,π')] + ρ_t E_{c_t≥1}[A_π(s,π')]]
  ≥ α E_{s∼Pr(s_t|π)}[A_π(s,π')] − 2αρ_t ε_inf   (worst-case the c_t≥1 branch and the dropped (1−ρ_t) each cost αρ_t ε_inf).
By the performance-difference lemma, η_μ(π_new)−η_μ(π) = Σ_t γ^t E_{Pr(s_t|π_new)}[A_π(s,π_new)] ≥ αΣ_t γ^t E_{Pr(s_t|π)}[A_π(s,π')] − 2αε_inf Σ_t γ^t(1−(1−α)^t). With Σ_t γ^t = 1/(1−γ) and Σ_t γ^t(1−α)^t = 1/(1−γ(1−α)), the first term is (α/(1−γ))𝔸 and the penalty bracket is 1/(1−γ) − 1/(1−γ(1−α)) = γα/((1−γ)(1−γ(1−α))). ∎

## Optimal step size

Using ε_inf ≤ R and 1−γ(1−α) ≥ 1−γ: η_μ(π_new)−η_μ(π) ≥ (α𝔸)/(1−γ) − (2α²R)/(1−γ)². Maximizing over α gives
  **α\* = (1−γ)𝔸/(4R)**,  and  **η_μ(π_new) − η_μ(π) ≥ 𝔸²/(8R).**
(Values in [0,1], horizon H=1/(1−γ): α\* = (1−γ)𝔸/4, improvement ≥ 𝔸²/8.)

## Algorithm and guarantees

**Exact CPI.** Initialize π. Repeat: call the greedy *policy chooser* for π' = argmax_{h∈Π} 𝔸_{π,μ}(h); if 𝔸_{π,μ}(π') > ε, set α = (1−γ)𝔸/(4R) and update π ← (1−α)π + απ', else HALT and return π.

- **(i) Monotone improvement:** every accepted update raises η_μ by ≥ 𝔸²/(8R) ≥ ε²/(8R) > 0.
- **(ii) Termination:** since η_μ ∈ [0,R], exact CPI accepts at most 8R²/ε² updates. The sample-based bookkeeping keeps the same order and inflates the loop constant to 72R²/ε², with trajectory counts polynomial in 1/ε and independent of |S|.
- **(iii) Output quality:** exact halt gives OPT(𝔸_{π,μ}) ≤ ε. Measure-mismatch bound — for an optimal π* and any μ̄,
   η_{μ̄}(π*) − η_{μ̄}(π) ≤ (ε/(1−γ)) ||d_{π*,μ̄}/d_{π,μ}||_∞ ≤ (ε/(1−γ)²) ||d_{π*,μ̄}/μ||_∞
   (using d_{π,μ}(s) ≥ (1−γ)μ(s)). So an *exploratory, uniform-ish* μ keeps the mismatch bounded and carries the η_μ guarantee back to η_D.

The policy chooser only needs *average* (L1) advantage-regression accuracy, E_{s∼d_{π,μ}} max_a |A_π(s,a)−f(s,a)| ≤ ε/2 — far weaker than the L∞ accuracy that approximate greedy requires.

## Implementation

```python
import numpy as np

class RestartMDP:
    """Simulator with restart access: next state can be drawn from mu."""
    def __init__(self, mu, gamma, n_actions, R):
        self.mu, self.gamma, self.n_actions, self.R = mu, gamma, n_actions, R
    def restart(self): ...      # draw s ~ mu
    def step(self, s, a): ...   # -> (s_next, reward in [0, R])

def sample_future_state(mdp, policy):
    # s ~ d_{policy, mu}: roll out from mu, stop each step w.p. (1 - gamma).
    s = mdp.restart()
    while np.random.rand() >= (1 - mdp.gamma):
        s, _ = mdp.step(s, policy.sample(s))
    return s

def estimate_Q(mdp, policy, s, a):
    # Unbiased normalized Q_policy(s,a): reward observed at a geometric time.
    s, r = mdp.step(s, a)
    while np.random.rand() < mdp.gamma:
        s, r = mdp.step(s, policy.sample(s))
    return r

def policy_advantage(mdp, policy, candidate, k):
    # A_{policy,mu}(candidate) = E_{s~d}[ sum_a (cand-pol)(a|s) Q(s,a) ],
    # estimated via uniform action sampling + importance weight n_a.
    est = []
    for _ in range(k):
        s = sample_future_state(mdp, policy)
        a = np.random.randint(mdp.n_actions)
        Qsa = estimate_Q(mdp, policy, s, a)
        est.append(mdp.n_actions * Qsa * (candidate.prob(s, a) - policy.prob(s, a)))
    return float(np.mean(est))

def fit_candidate(mdp, policy):
    # policy chooser: fit advantages (average L1 error set by the tolerance) and return
    # a policy that on average chooses high-advantage actions under d_{policy,mu}.
    ...                                                  # regression -> candidate pi'

class MixturePolicy:
    # pi_new(a|s) = (1 - alpha) pi(a|s) + alpha pi'(a|s)
    def __init__(self, base, cand, alpha): self.base, self.cand, self.alpha = base, cand, alpha
    def prob(self, s, a):
        return (1 - self.alpha) * self.base.prob(s, a) + self.alpha * self.cand.prob(s, a)
    def sample(self, s):
        return self.cand.sample(s) if np.random.rand() < self.alpha else self.base.sample(s)

def conservative_policy_iteration(mdp, policy, eps, k):
    while True:
        candidate = fit_candidate(mdp, policy)
        A = policy_advantage(mdp, policy, candidate, k)  # scalar verification
        if A <= eps:                                     # break point
            return policy
        alpha = (1 - mdp.gamma) * A / (4 * mdp.R)        # alpha* from the bound
        policy = MixturePolicy(policy, candidate, alpha) # conservative mixture update
        # guaranteed: eta_mu rises by >= A^2 / (8 R) this step
```
