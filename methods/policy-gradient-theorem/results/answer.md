# The Policy Gradient Theorem

## Problem it solves

Train an agent in a Markov decision process by gradient ascent on long-run reward, using a
function approximator, *without* the failure modes of estimating a value function and acting
greedily (discontinuous policies that can diverge under function approximation, and an inability
to represent stochastic optima). Parameterize the policy directly as a differentiable stochastic
map π(s,a,θ) = Pr{a_t=a | s_t=s, θ} and ascend the performance gradient ∂ρ/∂θ. The obstacle is
that performance ρ(θ) depends on θ through the environment's state-visitation distribution
d^π(s), whose derivative ∂d^π/∂θ involves the unknown transition dynamics and cannot be
estimated from samples.

## Key idea

The gradient of expected return with respect to the policy parameters does **not** contain the
derivative of the state weighting:

  ∂ρ/∂θ = Σ_s d^π(s) Σ_a [∂π(s,a)/∂θ] Q^π(s,a),

so it is built from states the agent visits by acting and actions it samples. In the
average-reward case d^π is the stationary distribution; in the start-state discounted case it is
the unnormalized discounted occupancy, implemented by γ^t weighting along a trajectory. Writing
(1/π)∂π/∂θ = ∇_θ log π gives the **score-function form**

  ∂ρ/∂θ = Σ_s d^π(s) E_{a∼π(·|s)}[ ∇_θ log π(a|s) Q^π(s,a) ],

which can be ascended from sampled trajectories. Any state-only **baseline** subtracts out
without bias (because Σ_a ∂π/∂θ = 0), reducing variance. A learned **compatible** critic that
satisfies ∂f_w/∂w = ∇_θ log π and is trained to its least-squares fixed point replaces the
unknown Q^π in the gradient *exactly*, yielding a convergence guarantee for the idealized
policy-iteration update with general differentiable function approximation.

## Setup

MDP with transitions P^a_{ss'} = Pr{s_{t+1}=s' | s_t=s, a_t=a} and expected rewards
R^a_s = E{r_{t+1} | s_t=s, a_t=a}. Policy π(s,a,θ) differentiable in θ ∈ ℝ^l. Two performance
measures, both covered by one theorem:

- **Average reward.** ρ(π) = lim_{n→∞}(1/n) E{r_1+…+r_n | π} = Σ_s d^π(s) Σ_a π(s,a) R^a_s, where
  d^π(s) = lim_{t→∞} Pr{s_t=s | s_0,π} is the stationary distribution. Differential value
  Q^π(s,a) = Σ_{t=1}^∞ E{r_t − ρ(π) | s_0=s, a_0=a, π}.
- **Start state (discounted).** ρ(π) = E{Σ_{t=1}^∞ γ^{t-1} r_t | s_0, π} = V^π(s_0),
  Q^π(s,a) = E{Σ_{k=1}^∞ γ^{k-1} r_{t+k} | s_t=s, a_t=a, π}, with discounted state weighting
  d^π(s) = Σ_{t=0}^∞ γ^t Pr{s_t=s | s_0, π}; this weighting is not normalized.

## Theorem 1 (Policy Gradient)

For any MDP, in either formulation,

  ∂ρ/∂θ = Σ_s d^π(s) Σ_a [∂π(s,a)/∂θ] Q^π(s,a).      (no ∂d^π/∂θ term)

**Proof (average reward).** With V^π(s) = Σ_a π(s,a) Q^π(s,a),
  ∂V^π(s)/∂θ = Σ_a [(∂π/∂θ)Q^π + π ∂Q^π/∂θ].
Average-reward Bellman: Q^π(s,a) = R^a_s − ρ + Σ_{s'} P^a_{ss'} V^π(s'), so
∂Q^π/∂θ = −∂ρ/∂θ + Σ_{s'} P^a_{ss'} ∂V^π(s')/∂θ. Substitute and use Σ_a π = 1 to pull out
−∂ρ/∂θ, then isolate:
  ∂ρ/∂θ = Σ_a (∂π/∂θ)Q^π + Σ_a π Σ_{s'} P^a_{ss'} ∂V^π(s')/∂θ − ∂V^π(s)/∂θ.   (★)
Sum (★) against d^π(s). LHS = ∂ρ/∂θ (since Σ_s d^π = 1). In the middle term the coefficient of
∂V^π(s')/∂θ is Σ_s d^π(s) Σ_a π(s,a) P^a_{ss'} = d^π(s') by stationarity, so it equals
Σ_{s'} d^π(s') ∂V^π(s')/∂θ, which cancels the last term Σ_s d^π(s) ∂V^π(s)/∂θ. Hence
∂ρ/∂θ = Σ_s d^π(s) Σ_a (∂π/∂θ) Q^π(s,a). ∎

**Proof (start state).** ∂V^π(s)/∂θ = Σ_a [(∂π/∂θ)Q^π + π Σ_{s'} γ P^a_{ss'} ∂V^π(s')/∂θ].
Unrolling repeatedly,
  ∂V^π(s)/∂θ = Σ_x Σ_{k=0}^∞ γ^k Pr(s→x,k,π) Σ_a (∂π(x,a)/∂θ) Q^π(x,a).
With ∂ρ/∂θ = ∂V^π(s_0)/∂θ and d^π(s) = Σ_k γ^k Pr(s_0→s,k,π),
∂ρ/∂θ = Σ_s d^π(s) Σ_a (∂π/∂θ) Q^π(s,a). ∎

## Score-function estimator and baseline (consequence)

Under the average-reward stationary distribution, or under the discounted occupancy with γ^t
time weighting, and with (1/π)∂π/∂θ = ∇_θ log π,

  ∂ρ/∂θ = Σ_s d^π(s) E_{a∼π(·|s)}[ ∇_θ log π(a|s) Q^π(s,a) ].

The **1/π** corrects oversampling of probable actions. Plugging in the actual return for Q^π
gives **REINFORCE**: Δθ_t ∝ ∇_θ log π(a_t|s_t) R_t, R_t = Σ_{k≥1} γ^{k-1} r_{t+k} (or
Σ_{k≥1}(r_{t+k}−ρ)). **Baseline invariance:** for any b(s),
E_{a∼π}[∇_θ log π(a|s) b(s)] = b(s) E_{a∼π}[∇_θ log π(a|s)] = b(s)·0 = 0, because
Σ_a ∂π(s,a)/∂θ = ∂/∂θ Σ_a π(s,a) = ∂/∂θ(1) = 0. So
∂ρ/∂θ = Σ_s d^π(s) E_{a∼π(·|s)}[∇_θ log π(a|s)(Q^π(s,a) − b(s))] for any state-only b; choose b ≈ V^π so the
multiplier is the advantage A^π = Q^π − V^π, reducing variance with zero bias. (This is the
multi-step lift of Williams' 1992 reinforcement-baseline result: Σ_ξ ∂g_i(ξ)/∂w = ∂(1)/∂w = 0.)

## Theorem 2 (Policy Gradient with Compatible Function Approximation)

Learn f_w ≈ Q^π by following π and minimizing squared error to an unbiased estimate of Q^π. At
the least-squares fixed point,

  Σ_s d^π(s) Σ_a π(s,a)[Q^π(s,a) − f_w(s,a)] ∂f_w(s,a)/∂w = 0.   (3)

If, in addition, f_w is **compatible** with the policy,

  ∂f_w(s,a)/∂w = (∂π(s,a)/∂θ)(1/π(s,a)) = ∇_θ log π(s,a),   (4)

then ∂ρ/∂θ = Σ_s d^π(s) Σ_a [∂π(s,a)/∂θ] f_w(s,a).   (5)

**Proof.** Substituting (4) into (3) gives Σ_s d^π Σ_a (∂π/∂θ)[Q^π − f_w] = 0 (6): the error is
orthogonal, under d^π, to the policy score. Subtract (6) (=0) from Theorem 1:
∂ρ/∂θ = Σ d^π Σ (∂π/∂θ)Q^π − Σ d^π Σ (∂π/∂θ)[Q^π − f_w] = Σ d^π Σ (∂π/∂θ) f_w. ∎

**Compatible features.** For a soft-max policy π(s,a) = e^{θ^T φ_sa} / Σ_b e^{θ^T φ_sb},
∇_θ log π(s,a) = φ_sa − Σ_b π(s,b) φ_sb, so the natural compatible critic is linear and
state-centered: f_w(s,a) = w^T[φ_sa − Σ_b π(s,b) φ_sb]. Then Σ_a π(s,a) f_w(s,a) = 0, i.e. f_w
is mean-zero per state — it estimates the **advantage** A^π(s,a) = Q^π(s,a) − V^π(s), not Q^π.
The gradient is invariant to adding any v(s): ∂ρ/∂θ = Σ_s d^π Σ_a (∂π/∂θ)[f_w(s,a) + v(s)],
since Σ_a ∂π/∂θ = 0; pick v ≈ V^π for variance control.

## Theorem 3 (Convergent Policy Iteration with Function Approximation)

With differentiable π and compatible f_w satisfying (4), max_{θ,s,a,i,j}|∂²π/∂θ_i∂θ_j| < B < ∞,
step sizes α_k → 0 and Σ_k α_k = ∞, and bounded rewards, the iteration

  w_k : Σ_s d^{π_k}(s) Σ_a π_k(s,a)[Q^{π_k}(s,a) − f_w(s,a)] ∂f_w/∂w = 0,
  θ_{k+1} = θ_k + α_k Σ_s d^{π_k}(s) Σ_a [∂π_k(s,a)/∂θ] f_{w_k}(s,a),

defines a performance sequence {ρ(π_k)} that converges with lim_{k→∞} ∂ρ(π_k)/∂θ = 0. **Proof.**
Theorem 2 makes the θ-update the exact performance gradient; bounded ∂²π and rewards bound
∂²ρ/∂θ_i∂θ_j; the step-size conditions then satisfy the hypotheses of the standard convergence
result. ∎

## Worked instantiation (compatible actor-critic)

Equation-faithful discounted start-state sketch. The average-reward version replaces `G` with a
differential return and drops the `gamma**t` occupancy weight.

```python
def discounted_returns(traj, gamma):
    G = 0.0
    out = [0.0] * len(traj)
    for t in reversed(range(len(traj))):
        G = traj[t][2] + gamma * G
        out[t] = G
    return out

def compatible_policy_iteration(env, policy, critic, alpha_theta, alpha_w,
                                critic_steps, gamma=1.0):
    """policy.grad_log_prob(s,a) = d log pi(a|s) / d theta.
       Compatibility requires critic.grad_w(s,a) == policy.grad_log_prob(s,a)."""
    while True:
        # Act under the current policy; gamma**t supplies discounted occupancy.
        traj = []
        s, done = env.reset(), False
        while not done:
            a = policy.sample(s)
            s2, r, done = env.step(a)
            traj.append((s, a, r))
            s = s2

        returns = discounted_returns(traj, gamma)

        # Fit f_w toward the least-squares condition (3) for the current policy.
        for _ in range(critic_steps):
            for t, (s, a, _) in enumerate(traj):
                residual = returns[t] - critic.value(s, a)
                critic.w = critic.w + alpha_w * (gamma**t) * residual \
                                      * critic.grad_w(s, a)

        # Use (5): sum_s d^pi(s) sum_a d pi/d theta * f_w(s,a).
        grad = 0.0
        for t, (s, a, _) in enumerate(traj):
            grad = grad + (gamma**t) * critic.value(s, a) \
                         * policy.grad_log_prob(s, a)
        policy.theta = policy.theta + alpha_theta * grad
```

For a soft-max policy the eligibility is `grad_log_prob(s,a) = phi(s,a) - sum_b pi(s,b) phi(s,b)`;
the compatible critic uses `critic.value(s,a) = dot(w, grad_log_prob(s,a))`, so
`sum_a pi(s,a) critic.value(s,a) = 0` in every state and the learned signal is an advantage.
