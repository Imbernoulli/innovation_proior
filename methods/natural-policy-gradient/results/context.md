# Context: covariant search in the space of parameterized policies

## Research question

We want to find a good control policy for a large Markov decision process by **directly
adjusting the parameters of a stochastic policy** and following the gradient of long-run reward,
instead of building an approximate value function and acting greedily with respect to it. Direct
policy search is attractive precisely where value-function methods are fragile: a small parameter
change produces a small change in the policy and in the state-visitation distribution, so the
method degrades gracefully, can encode domain knowledge in the policy class, and comes with
convergence guarantees that fitted-value methods lack.

The pain point is the update rule itself. The standard rule is

    Δθᵢ = α ∂η/∂θᵢ,

where η is the average reward and θ ∈ ℝᵐ parameterizes the policy π(a;s,θ). This rule is
**non-covariant**: it depends on the arbitrary choice of coordinates θ. Crudely, it is
dimensionally inconsistent — the left side carries units of θᵢ and the right side units of 1/θᵢ,
and different θᵢ need not even share units. Concretely, if we re-parameterize the policy by some
smooth change of variables, the gradient transforms as a covector, not as the tangent-vector move
we apply to the parameters, so the direction we call "steepest ascent" rotates to point somewhere
else. A solution would have to make the ascent direction a property of the *policy* — of the
distribution the parameters encode — and not of the labels we happened to give the parameters. It
must do this while keeping the one property that made policy gradients usable in the first place:
that the gradient can be estimated from sampled trajectories without differentiating the (unknown)
state distribution.

## Background

**Markov decision processes and the average-reward setting.** A finite MDP is a tuple
(S, s₀, A, R, P) with a stochastic policy π(a;s) giving the probability of action a in state s.
Assuming every policy is ergodic with stationary distribution ρ^π(s), the **average reward** is
η(π) = Σ_{s,a} ρ^π(s) π(a;s) R(s,a). The differential state–action value is
Q^π(s,a) = E_π[ Σ_{t≥0} (R(s_t,a_t) − η(π)) | s₀=s, a₀=a ], and V^π(s) = E_{a∼π}[Q^π(s,a)].
The advantage is A^π(s,a) = Q^π(s,a) − V^π(s). The goal is to maximize η over a smoothly
parameterized class { π_θ : θ ∈ ℝᵐ }.

**Information geometry and the natural gradient (Amari).** When the parameters of a model index a
family of probability distributions, the parameter space is not flat Euclidean space but a
**Riemannian manifold**. The squared length of an infinitesimal step is not Σ(dθᵢ)² but the
quadratic form |dθ|² = Σ_{ij} Gᵢⱼ(θ) dθᵢ dθⱼ = dθᵀ G(θ) dθ, where G is a position-dependent metric
tensor. The steepest-ascent direction of a function under such a metric — the dθ that maximizes
the function subject to |dθ|² fixed — is **G⁻¹∇** rather than ∇. Amari (1996, 1998) identified the
right metric for a family of distributions: the **Fisher information matrix**
Gᵢⱼ = E[ ∂ᵢ log p · ∂ⱼ log p ]. By a classical result of information geometry (Cencov), the Fisher
metric is, up to scale, the **unique invariant metric** on a space of probability distributions —
it assigns the same distance between two distributions no matter which coordinates parameterize
them, and locally it equals the Hessian of the Kullback–Leibler divergence,
D_KL(p_θ ‖ p_{θ+dθ}) = ½ dθᵀ F dθ + O(dθ³). Amari called G⁻¹∇ the **natural gradient** and proved
that natural-gradient online learning is asymptotically Fisher-efficient (it attains the
Cramér–Rao bound) in parameter-estimation problems, where the Fisher information converges to the
Hessian of the loss. A documented motivation was that natural-gradient learning escapes the
plateaus that trap ordinary gradient descent.

**The vanilla policy gradient and its score-function structure.** Writing ∇log π(a;s,θ) = ∇π/π
(the *score*), the gradient of any expectation under π carries a factor of the score, so policy
gradients are naturally expressed in terms of E[ ∇log π · (·) ].

## Baselines

**Vanilla policy gradient with function approximation (Sutton, McAllester, Singh & Mansour 1999).**
The **policy gradient theorem** states, for both the average-reward and start-state formulations,

    ∂η/∂θ = Σ_s d^π(s) Σ_a (∂π(s,a)/∂θ) Q^π(s,a),

with no ∂d^π/∂θ term — the effect of the policy on the state distribution vanishes, which is what
makes the gradient estimable from samples. They further show how to replace the unknown Q^π by a
learned linear critic f_w(s,a) = wᵀψ(s,a) **without biasing the gradient**, provided f_w is
*compatible*: ∂f_w/∂w = (∂π/∂θ)·(1/π), i.e. the critic's features are the policy's score,
ψ(s,a) = ∇log π(a;s,θ). For a Gibbs policy π ∝ exp(θᵀφ_sa) compatibility forces
ψ_sa = φ_sa − Σ_b π(s,b)φ_sb (the features, centered to mean zero under π), so the compatible
critic is best read as an approximation to the **advantage** A^π, not to Q^π itself. Theorem 2
there proves that if f_w is compatible and has converged (its error orthogonal to the score), then
substituting f_w for Q^π leaves the gradient exact. **Gap:** the update still moves along the raw
gradient ∂η/∂θ, so it remains non-covariant; the framework says *what* to approximate but not
*which direction* to move.

**Natural-gradient learning in supervised problems (Amari 1996/1998).** The natural gradient
G⁻¹∇ is derived and shown efficient for perceptron training, blind source separation, and linear
systems. **Gap:** it is developed for a *single* probability model with one well-defined Fisher
manifold and a loss whose Hessian the Fisher converges to; it is not formulated for a control
problem, where there is one conditional distribution π(·;s) *per state* and the objective η is an
expectation over the induced stationary distribution rather than a log-likelihood. Whether a
natural gradient even exists for the policy-search objective, and what metric to use, is open.

**Value-based policy iteration / approximate policy iteration.** Exact policy iteration alternates
greedy improvement (π'(s) = argmax_a Q^π(s,a)) with policy evaluation and converges fast, but with
*approximate* value functions it is notoriously non-monotone — it can improve a policy sharply for
a few iterations and then degrade it badly (documented on problems such as Tetris). **Gap:** it
takes discrete, unbounded greedy steps with no notion of a controlled, locally-safe move, and its
relationship to the smooth, monotone-but-slow policy gradient is not understood.

## Evaluation settings

The natural yardsticks are small MDPs where the exact gradient and value function can be computed
or closely estimated — a one-dimensional linear–quadratic regulator with Gaussian noise and a
quadratic-feature Gibbs policy, and small discrete MDPs with a handful of states and actions where
one parameter governs a near-deterministic action choice. The harder testbed is the MDP of
**Tetris** with a linear value/policy over standard board features, a domain where approximate
policy iteration is known to peak and then collapse, making it a stress test for any policy-search
direction. The metric of interest is average reward (game score) as a function of training time /
number of updates, with the conventional gradient as the reference direction.

## Code framework

Policy search over a parameterized stochastic policy. The data pipeline (rollouts under the current
policy), the policy parameterization, the score function, and the gradient estimator already exist;
the **direction** in which to move the parameters is the empty slot.

```python
import numpy as np

class Policy:
    """Differentiable stochastic policy π(a; s, θ)."""
    def __init__(self, theta):
        self.theta = theta                      # parameter vector θ ∈ R^m

    def prob(self, s, a):                        # π(a; s, θ)
        raise NotImplementedError

    def sample(self, s):                         # a ~ π(·; s, θ)
        raise NotImplementedError

    def score(self, s, a):
        # ∇_θ log π(a; s, θ) = ∇π/π  — the score / compatible feature vector ψ(s,a)
        raise NotImplementedError


def estimate_value(policy, mdp):
    """Differential state-action value Q^π(s,a) (or an estimate) under the average-reward setting."""
    raise NotImplementedError


def policy_gradient(policy, mdp, Q):
    # ∇η(θ) = Σ_{s,a} ρ^π(s) ∇π(a;s,θ) Q^π(s,a)   (Sutton et al. 1999)
    g = np.zeros_like(policy.theta)
    for s in mdp.states:
        rho = mdp.stationary_prob(s, policy)
        for a in mdp.actions:
            g += rho * policy.prob(s, a) * policy.score(s, a) * Q(s, a)
    return g


def ascent_direction(policy, mdp, grad):
    # TODO: turn the raw gradient into the direction we will actually step.
    #       The vanilla choice is to return `grad` unchanged, which moves under an implicit
    #       identity metric.
    pass


def train(policy, mdp, alpha, n_iters):
    for _ in range(n_iters):
        Q = estimate_value(policy, mdp)
        grad = policy_gradient(policy, mdp, Q)
        direction = ascent_direction(policy, mdp, grad)
        policy.theta = policy.theta + alpha * direction
    return policy
```
