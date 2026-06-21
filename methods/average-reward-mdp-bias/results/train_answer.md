When a decision process has no terminal state, the discounted return is the wrong objective. The discount factor γ was introduced only to keep an infinite sum finite, but it also imposes an effective horizon of about 1/(1−γ) steps. For a genuinely continuing task, that horizon is a fiction, and it can rank policies incorrectly: a policy that pays rewards sooner can beat a policy with a strictly higher per-step average just because γ downweights the future. As γ approaches 1 the values diverge, so the comparison becomes the difference of two infinities. The honest criterion is the long-run average reward per step, g = lim_{N→∞} (1/N) E[Σ_{t=0}^{N-1} r_t], but the standard discounted Bellman machinery breaks down at γ=1 because I−P_π becomes singular.

Existing approaches each leave a gap. Discounted value iteration and policy iteration require γ<1 and inherit the horizon bias. Howard's average-reward policy iteration finds a policy of maximal gain, but gain is a single scalar that is state-independent for unichain policies, so it cannot rank states and does not discriminate among policies that tie on the long-run average but differ in the finite reward accumulated along the way. Computing only the stationary distribution gives the scalar g as well, but no relative value to guide improvement. Schwartz's R-learning is an average-adjusted learning heuristic without a worked-out optimality theory or a systematic way to break ties among equal-gain policies.

The right framework is the average-reward MDP, also called the gain/bias characterization. For a stationary policy π, the long-run average reward g is the gain, and for a unichain policy it is the same constant for every starting state. Because a scalar cannot rank states, we introduce a second function, the bias or relative value h(s), defined as the finite offset obtained by subtracting the steady stream N·g from the N-step expected reward. Intuitively, h(s) measures how much better or worse it is to start in state s rather than in the steady state. Together (g, h) satisfy the average-reward Poisson equation g + h(s) = r_π(s) + Σ_{s'} p(s'|s,π(s)) h(s'), or in vector form g·1 + (I−P_π)h = r_π. This equation is solvable because choosing g equal to the stationary expectation d·r_π puts r_π−g·1 in the range of the singular matrix I−P_π; the remaining additive freedom in h is harmless and is pinned by setting h(reference)=0.

The same pair emerges naturally from the discounted limit. The discounted value V_γ has a pole at γ=1 and admits a Laurent expansion V_γ(s) = g/(1−γ) + h(s) + o(1) as γ→1. The coefficient of the 1/(1−γ) pole is the gain, and the constant term is the bias. Taking the maximum over actions yields the average-reward Bellman optimality equation g* + h*(s) = max_a [r(s,a) + Σ_{s'} p(s'|s,a) h*(s')]. A stationary deterministic policy greedy with respect to h* attains the optimal gain g*. The framework also gives a systematic optimality ladder: (−1)-discount-optimal is gain-optimal; 0-discount-optimal is gain-optimal plus bias-optimal, which breaks ties by preferring policies with larger transient rewards; and ∞-discount-optimal, or Blackwell-optimal, is a single policy that is discount-optimal for all γ sufficiently close to 1, maximizing the Laurent coefficients lexicographically.

Computationally, policy iteration alternates between solving the Poisson equation for (g, h) and improving the policy greedily with respect to r + P h; it terminates in finitely many steps with a gain-optimal policy. Relative value iteration avoids the per-iteration linear solve by iterating the backup T(h)(s) = max_a [r(s,a) + Σ p(s'|s,a) h(s')], subtracting a reference state at each step to kill the common drift, and stopping when the span seminorm sp(T(h)−h) is small. The drift itself recovers g. For periodic chains an aperiodicity transform P̃=(1−τ)I+τP with scaled rewards r̃=τr leaves the optimal policy unchanged while ensuring convergence.

```python
import numpy as np

class MDP:
    """Finite MDP. P[a] is an (n,n) row-stochastic transition matrix; R[a] is length-n reward."""
    def __init__(self, P, R):
        self.P = P
        self.R = R
        self.n = P[0].shape[0]
        self.m = len(P)

def evaluate_policy(mdp, pi, ref=0):
    """Solve the average-reward Poisson equation g + h = r_pi + P_pi h, with h(ref)=0."""
    n = mdp.n
    Ppi = np.stack([mdp.P[pi[s]][s] for s in range(n)])
    rpi = np.array([mdp.R[pi[s]][s] for s in range(n)])
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    A[:n, :n] = np.eye(n) - Ppi
    A[:n, n] = 1.0
    b[:n] = rpi
    A[n, ref] = 1.0
    x = np.linalg.solve(A, b)
    return x[n], x[:n]  # gain, bias

def policy_iteration(mdp):
    """Howard-style average-reward policy iteration."""
    pi = np.zeros(mdp.n, dtype=int)
    while True:
        g, h = evaluate_policy(mdp, pi)
        Q = np.stack([mdp.R[a] + mdp.P[a] @ h for a in range(mdp.m)], axis=0)
        pi_new = Q.argmax(axis=0)
        if np.array_equal(pi_new, pi):
            return g, h, pi
        pi = pi_new

def relative_value_iteration(mdp, ref=0, eps=1e-9, tau=0.5):
    """Relative value iteration with aperiodicity transform and span seminorm stopping."""
    P = [(1 - tau) * np.eye(mdp.n) + tau * mdp.P[a] for a in range(mdp.m)]
    R = [tau * mdp.R[a] for a in range(mdp.m)]
    V = np.zeros(mdp.n)
    while True:
        Q = np.stack([R[a] + P[a] @ V for a in range(mdp.m)], axis=0)
        raw = Q.max(axis=0)
        drift = raw - V
        V_next = raw - raw[ref]
        if drift.max() - drift.min() < eps:
            g = 0.5 * (drift.max() + drift.min()) / tau
            return g, V_next, Q.argmax(axis=0)
        V = V_next
```
