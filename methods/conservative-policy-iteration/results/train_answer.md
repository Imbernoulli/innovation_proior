In reinforcement learning problems too large to enumerate, the two standard approximate approaches both have a fatal weakness that is not just a question of sample noise. Approximate greedy policy iteration, which builds an estimated Q-function and switches to the greedy policy, only gives a max-norm bound that allows the value to drop by O(ε/(1−γ)); a single greedy swap can degrade the policy because it commits to errors everywhere at once. Policy gradient, by contrast, moves the policy infinitesimally and so improves the return to first order, but its gradient is weighted by the current policy's state-visitation distribution d_{π,D}. In problems that require exploration, the states whose improvement is necessary for optimality may be visited exponentially rarely, so the gradient estimate is accurate in magnitude yet useless in direction. Both failures come from the same source: we are scoring improvement under a distribution that the current policy itself controls, and that distribution systematically hides the states that most need improvement.

The way out is to change the objective and the update. We introduce Conservative Policy Iteration, or CPI. Instead of optimizing the return from the true start distribution, η_D(π) = E_{s∼D}[V_π(s)], CPI optimizes η_μ(π) = E_{s∼μ}[V_π(s)] under an exploratory restart distribution μ. The optimal policy still maximizes η_μ, because it maximizes V_π at every state, but η_μ weights states more uniformly and is not starved by the current policy's bad habits. The restart distribution is assumed available: the agent can reset its next state to a draw from μ, which is weaker than a full generative model but strong enough to gather information about states the current policy would never reach.

The core of CPI is a measurable surrogate called the policy advantage. For a candidate policy π' relative to the current policy π and measure μ, it is defined as 𝔸_{π,μ}(π') = E_{s∼d_{π,μ}} E_{a∼π'(·|s)}[A_π(s,a)], where d_{π,μ} is the discounted future-state distribution under π starting from μ and A_π is the advantage. This quantity is an average under the old policy's visitation, so it can be estimated from samples without ever observing the new policy's distribution. It is also the first-order coefficient of η_μ: if we interpolate from π toward π' with weight α, then ∂η_μ/∂α at α = 0 equals (1/(1−γ))𝔸_{π,μ}(π').

Rather than taking a full greedy step, CPI updates by a mixture: π_new(a|s) = (1−α)π(a|s) + απ'(a|s). This means a trajectory under π_new looks like a trajectory under π with occasional deviations to π'. Conditioned on no deviation having occurred, the state distribution is exactly the same as under π, so the off-distribution drift is controlled by the probability α of ever deviating. A careful bound shows that the improvement satisfies η_μ(π_new) − η_μ(π) ≥ (α/(1−γ))(𝔸 − 2αγε_inf/(1−γ(1−α))), where ε_inf bounds the per-state advantage magnitude. The gain is first order in α while the off-distribution penalty is second order, which is precisely why the mixture is safe. Maximizing this bound with ε_inf ≤ R gives the closed-form step size α* = (1−γ)𝔸/(4R) and a guaranteed per-step improvement of 𝔸²/(8R).

The algorithm is therefore a simple loop. Fit a candidate policy π' that has large policy advantage, estimate the scalar 𝔸 from samples, and halt if 𝔸 is below a tolerance ε. Otherwise take the mixture step with α*. Every accepted step improves η_μ by at least ε²/(8R), and because η_μ is bounded by R, the number of accepted updates is at most O(R²/ε²), independent of the state-space size. When the algorithm halts, the small policy advantage under μ translates to near-optimality under any measure μ̄ up to a mismatch ratio ||d_{π*,μ̄}/μ||_∞, which is why μ should be chosen exploratory and uniform-ish.

```python
import numpy as np

class RestartMDP:
    """Simulator with restart access: next state can be drawn from mu."""
    def __init__(self, mu, gamma, n_actions, R):
        self.mu, self.gamma, self.n_actions, self.R = mu, gamma, n_actions, R

    def restart(self):
        # draw s ~ mu
        raise NotImplementedError

    def step(self, s, a):
        # -> (s_next, reward in [0, R])
        raise NotImplementedError


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
    # A_{policy, mu}(candidate) = E_{s~d}[ sum_a (cand(a|s)-pol(a|s)) Q(s,a) ],
    # estimated via uniform action sampling + importance weight n_a.
    est = []
    for _ in range(k):
        s = sample_future_state(mdp, policy)
        a = np.random.randint(mdp.n_actions)
        Qsa = estimate_Q(mdp, policy, s, a)
        est.append(
            mdp.n_actions * Qsa * (candidate.prob(s, a) - policy.prob(s, a))
        )
    return float(np.mean(est))


def fit_candidate(mdp, policy):
    # Policy chooser: fit advantages with average L1 error set by the tolerance,
    # and return a policy that on average picks high-advantage actions under d.
    raise NotImplementedError  # regression -> candidate pi'


class MixturePolicy:
    # pi_new(a|s) = (1 - alpha) pi(a|s) + alpha pi'(a|s)
    def __init__(self, base, cand, alpha):
        self.base, self.cand, self.alpha = base, cand, alpha

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
        # guaranteed: eta_mu rises by >= A^2 / (8 * R) this step
```
