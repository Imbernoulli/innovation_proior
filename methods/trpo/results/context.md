## Research question

Can we train large, nonlinear stochastic policies (neural networks with tens of
thousands of parameters) for sequential decision making in a way that makes
*reliable, monotonic* progress with little hyperparameter tuning?

The tension is sharp. Gradient-based optimization has far better sample/oracle
complexity guarantees than black-box search, and it has just made deep supervised
learning scale to millions of parameters. Yet in policy optimization, simple
derivative-free methods (the cross-entropy method, covariance-matrix adaptation)
were routinely competitive with or better than gradient methods on hard problems
like Tetris and locomotion. A gradient method that consistently beats random search
on large policies — and does so without per-task tuning — would unlock training of
rich, general-purpose controllers (e.g. neural-network locomotion policies, or
image-input game-playing policies). The crux is the *step*: how far should each
update move the policy so that the true return actually goes up, every time?

## Background

**The MDP setup and the performance objective.** An infinite-horizon discounted
MDP `(S, A, P, r, ρ0, γ)`. A stochastic policy `π(a|s)` has expected discounted
return `η(π) = E_{s0,a0,…}[ Σ_t γ^t r(s_t) ]`. Standard definitions: state-action
value `Q_π`, value `V_π`, advantage `A_π(s,a) = Q_π(s,a) − V_π(s)`. The
(unnormalized) discounted state-visitation frequency is
`ρ_π(s) = Σ_t γ^t P(s_t = s)`.

**The advantage decomposition (Kakade & Langford, 2002).** The return of any
policy `π̃` relative to a reference `π` is exactly the expected discounted advantage
of `π̃` accumulated under `π̃`'s own trajectories:
`η(π̃) = η(π) + E_{τ∼π̃}[ Σ_t γ^t A_π(s_t,a_t) ]`, equivalently
`η(π̃) = η(π) + Σ_s ρ_{π̃}(s) Σ_a π̃(a|s) A_π(s,a)`. The catch: the new visitation
`ρ_{π̃}` depends on `π̃` in a complicated, coupled way, so this exact expression is
hard to optimize directly.

**The local approximation L.** Replacing `ρ_{π̃}` by `ρ_π` gives
`L_π(π̃) = η(π) + Σ_s ρ_π(s) Σ_a π̃(a|s) A_π(s,a)`. For a differentiable
parameterization `π_θ`, `L` agrees with `η` to first order at the current point:
`L_{π_{θ0}}(π_{θ0}) = η(π_{θ0})` and
`∇_θ L_{π_{θ0}}(π_θ)|_{θ0} = ∇_θ η(π_θ)|_{θ0}`. So improving `L` by a *small enough*
step improves `η` — but nothing here says how big "small enough" is.

**The conservative-policy-iteration bound (Kakade & Langford, 2002).** This is the
only known result that turns "small enough" into an explicit number. For a *mixture*
update `π_new = (1−α)π_old + α π'` (with `π' = argmax L_{π_old}`), they prove
`η(π_new) ≥ L_{π_old}(π_new) − (2εγ/(1−γ)^2) α^2`, where
`ε = max_s |E_{a∼π'(·|s)}[A_π(s,a)]|`. A lower bound on improvement means: improving
the right-hand side guarantees real improvement. The limitation is the policy class
— it holds only for these mixture policies, which nobody parameterizes with a neural
network.

**Distribution geometry: KL and the Fisher metric.** Two facts about probability
distributions that the field already had. (i) Pinsker-type inequality relating total
variation to KL: `D_TV(p,q)^2 ≤ D_KL(p,q)`. (ii) The KL divergence between
`π_θ` and a nearby `π_{θ+Δ}` is, to second order, a quadratic form in `Δ` with the
*Fisher information matrix* as its Hessian:
`D_KL(π_θ‖π_{θ+Δ}) ≈ ½ Δ^T F(θ) Δ`, `F = E[∇log π ∇log π^T]`. The Fisher metric is
invariant to how the distribution is parameterized — it measures distance in the
space of distributions, not in raw coordinate space.

**Diagnostic observation about fixed-step methods.** Methods that take a *fixed*
step in the Fisher metric (a fixed Lagrange penalty / learning rate) work on easy
control tasks but stall on harder ones: a fixed step is simultaneously too large
somewhere (causing a collapse in performance after a single bad update) and too
small elsewhere (wasting samples). Euclidean-parameter steps are worse still,
because the same parameter displacement can mean a negligible or a ruinous change
in the policy distribution. This is the empirical pain that the step rule has to
contend with.

## Baselines

**Vanilla policy gradient (REINFORCE / GPOMDP; Williams 1992; Sutton et al. 2000;
Bartlett & Baxter 2011; Peters & Schaal 2008a).** Estimate
`∇_θ η = E[ ∇_θ log π_θ(a|s) A_π(s,a) ]` from trajectories and take a step
`θ ← θ + α ∇_θ η`. Core idea: ascend the return directly with the likelihood-ratio
(score-function) gradient; a baseline reduces variance. Gap: the step is steepest
ascent in *Euclidean* parameter space, which has no meaning for distributions; step
size is unprincipled and brittle (too large collapses the policy, too small crawls),
and variance is high.

**Natural policy gradient (Kakade 2002; Bagnell & Schneider 2003 covariant policy
search; Peters & Schaal 2008b natural actor-critic).** Precondition the gradient by
the inverse Fisher matrix: step direction `F^{-1} g`, with a fixed step size /
penalty `1/λ`. Core idea: steepest ascent in the *distribution* (Fisher) metric, so
the direction is reparameterization-invariant — a major conceptual improvement over
vanilla PG. Natural actor-critic estimates this direction with a compatible linear
critic. Gap: it uses a *fixed* penalty/step, so on hard problems it over- or
under-shoots; and forming or inverting `F` is infeasible for large networks if done
naively.

**Conservative policy iteration (Kakade & Langford 2002).** The mixture update with
the `α^2` improvement bound above. Core idea: the first method with an explicit
monotonic-improvement guarantee for approximate RL. Gap: restricted to mixture
policies, so the guarantee never transfers to neural-network policies.

**Relative entropy policy search (Peters, Mülling & Altün 2010).** Maximize expected
reward subject to a KL bound on the *state-action marginal* `p(s,a)` between
successive policies. Core idea: bound the information loss per update. Gap: requires
a costly nonlinear optimization in the inner loop, and constrains the marginal
rather than the conditional `π(a|s)`.

**Derivative-free search (cross-entropy method, Szita & Lőrincz 2006; CMA-ES,
Hansen & Ostermeier 1996).** Treat return as a black box over policy parameters and
evolve a search distribution. Core idea: simple, robust, no gradients. Gap: sample
complexity scales poorly with the number of parameters, so it does not scale to
large neural-network policies.

## Evaluation settings

Continuous control via the MuJoCo physics simulator: 2D *swimmer* (~10-dim state,
reward = forward velocity minus a small control penalty), *hopper* (~12-dim, with a
survival bonus and episode termination on falling), and *walker* (~18–20-dim, with
a foot-impact penalty). States are generalized positions/velocities; controls are
joint torques; the challenge is underactuation, high dimensionality, and contact
discontinuities. A linear-policy cart-pole balancing task (Barto et al. 1983) serves
as a small standard baseline easy for black-box methods. For perception-based
control: Atari games from raw images via the Arcade Learning Environment (Bellemare
et al. 2013), with the preprocessing protocol of Mnih et al. 2013 (downsampled
stacked frames). Discount `γ = 0.99`. Metric: total reward / return per episode over
training, comparing gradient and gradient-free methods at matched sample budgets,
and against reported scores for value-based and search-based agents on Atari.

## Code framework

A minimal neural policy-search harness already has a Gym-style environment loop, a
policy network that outputs an action distribution, a value-function baseline, an
advantage estimator, and a standard optimizer for value regression. The policy
improvement rule remains a single empty slot.

```python
import numpy as np, torch, torch.nn as nn

class PolicyNet(nn.Module):
    """Maps state -> parameters of an action distribution (e.g. Gaussian mean,
    or categorical logits). Provides log-prob and a distribution object."""
    def forward(self, obs):
        # TODO: map observations to distribution parameters.
        pass
    def distribution(self, obs):
        # TODO: return a torch.distributions object.
        pass
    def log_prob(self, obs, act):
        # TODO: evaluate action log-probabilities.
        pass

class ValueNet(nn.Module):
    """State-value baseline V(s), fit by regression to returns."""
    def forward(self, obs):
        # TODO: predict V(s).
        pass

def collect_trajectories(env, policy, steps):
    # TODO: roll out the current policy and store the data needed for an update.
    pass

def estimate_advantages(rewards, values, gamma, lam):
    # TODO: compute an advantage estimate and value targets.
    pass

def update_policy(policy, batch):
    # TODO: design the policy improvement rule.
    pass

def fit_value(value_net, obs, returns, opt, iters):
    # TODO: regress V(s) onto return targets.
    pass

def train(env, policy, value_net, epochs):
    for _ in range(epochs):
        batch = collect_trajectories(env, policy, steps=4000)
        estimate_advantages(...)
        update_policy(policy, batch)
        fit_value(value_net, ...)
```
