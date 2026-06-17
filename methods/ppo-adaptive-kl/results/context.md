## Research question

I have a stochastic policy `π_θ(a|s)` — a neural network mapping a state to a distribution
over continuous actions — and I want to improve it by gradient ascent on the expected return.
The hard constraint that governs everything is the *on-policy* one: the only data I have are
trajectories sampled from the *current* policy, and the moment I change `θ`, those samples
describe a policy that no longer exists. Sampling is expensive — every batch is a fresh set of
environment rollouts — so I want to extract as much improvement as I can from each batch,
ideally many gradient steps, not one. But there is a sharp tension: a large or repeated update
computed from a fixed batch can move the policy so far that the batch no longer reflects it, and
the result is not a small loss of accuracy but a *collapse* — the policy degrades, the new
rollouts are worse, and training does not recover. The precise goal is a policy-optimization
method that (1) is first-order and cheap enough to scale to large networks and to architectures
that share parameters between the policy and the value function, that use dropout, or that carry
auxiliary losses; (2) extracts multiple epochs of useful gradient steps from each sampled batch
without the destructive-update collapse; (3) keeps each policy update inside a controlled region
so improvement is reliable rather than occasionally catastrophic; (4) needs little
hyperparameter tuning, with whatever knob controls the update size staying meaningful both across
different problems and across the course of a single run, where the right update size drifts as
the policy moves from far-from-optimal to nearly converged. Each existing method below achieves
some of these; none achieves all at once, and the methods that best control the update size are
exactly the ones that are second-order and hard to scale.

## Background

By this time policy-gradient reinforcement learning with neural-network function approximators
is one of the main routes to continuous control — simulated robotic locomotion in MuJoCo
(Todorov et al. 2012), benchmark suites like the one of Duan et al. (2016), and parallel deep-RL
systems (Mnih et al. 2016). The dominant on-policy recipe rests on a few load-bearing ideas.

The **score-function policy gradient** (Williams 1992): for an objective `J(θ) = E_τ[R(τ)]`
the gradient can be written as an expectation that only needs sampled trajectories and the
log-probability of the chosen actions,

```
g = E_t[ ∇_θ log π_θ(a_t | s_t) · Â_t ],
```

where `Â_t` is an estimate of the advantage of action `a_t` at state `s_t`. Implementations
realize this by differentiating a surrogate objective `L^PG(θ) = E_t[ log π_θ(a_t|s_t) · Â_t ]`
whose gradient is `g`. This is unbiased but high-variance, and — crucially — it is derived for a
*single* gradient step at the current `θ`: the estimator is only valid in expectation under the
sampling policy, so nothing in it protects against the update moving the policy somewhere the
batch does not describe.

The **advantage** is the quantity that tells the gradient which actions to reinforce relative to
the state's baseline value `V(s)`. Estimating it well is a bias-variance problem. Generalized
advantage estimation (Schulman et al. 2015a) forms an exponentially λ-weighted sum of one-step
temporal-difference residuals `δ_t = reward_t + γ V(s_{t+1}) − V(s_t)`,

```
Â_t = δ_t + (γλ) δ_{t+1} + (γλ)^2 δ_{t+2} + … ,
```

with `λ ∈ [0,1]` trading bias (small λ, leans on the value function) against variance (large λ,
leans on the empirical returns). A learned value function `V_θ(s)` is fit alongside the policy to
supply the baseline; in parallel-actor implementations the policy is run for a fixed `T` steps in
each of `N` actors and the advantage estimator is truncated at the horizon `T`.

The **parallel-actor, fixed-horizon harness** (Mnih et al. 2016): rather than full episodes, run
the current policy for `T` timesteps in each of `N` parallel environments, collect `NT`
transitions, compute advantages that do not look beyond `T`, take an update, and discard the
data. An entropy bonus on the policy is commonly added to keep exploration alive. This is the
data-collection skeleton on-policy methods plug into.

The **observed failure mode that motivates the whole problem** is well documented: if one takes
the single-step surrogate `L^PG` and naively performs *several* epochs of gradient ascent on it
using the same batch, the policy update is not well justified and empirically leads to
destructively large policy changes — the policy moves far outside the region the batch describes,
performance collapses, and it does not recover. This is the concrete pain that any
multi-epoch-per-batch method has to defeat. A second, related observation: the *amount* a fixed
gradient step moves the policy distribution is not constant — early in training, far from the
optimum, a given step changes the action distribution a lot; late in training it changes it
little — so any fixed update-size knob is the wrong size at some point in the run.

The natural way to measure "how far the policy moved" is the Kullback-Leibler divergence between
the old and new action distributions, `KL[π_old(·|s), π_θ(·|s)]`, averaged over the visited
states. For a diagonal-Gaussian policy (the standard choice for continuous control, a network
emitting a mean `μ(s)` with a learned per-dimension standard deviation `σ`) this KL has a closed
form in `(μ, σ)`, so it can be computed analytically and cheaply, with no extra sampling. This
distributional distance, not the Euclidean distance in parameter space, is the quantity that
correlates with whether a batch still describes the updated policy.

## Baselines

These are the methods a new on-policy optimizer would be measured against and would react to.

**Vanilla policy gradient / REINFORCE (Williams 1992; the on-policy actor-critic of Mnih et al.
2016).** Estimate `g = E_t[∇_θ log π_θ(a_t|s_t) Â_t]`, take one ascent step, throw the batch away,
collect again. With a learned value baseline and GAE advantages this is a workable continuous-
control method. **Gap:** it uses each expensive batch for a single gradient step, so it is
sample-inefficient; and there is no mechanism preventing a step (or a learning rate) from being
too large — push the step size up to improve efficiency and the update becomes unreliable, with
no built-in notion of "you have moved the policy too far."

**Conservative policy iteration (Kakade & Langford 2002).** Works with the surrogate
`L^CPI(θ) = E_t[ r_t(θ) Â_t ]`, where `r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)` is the importance
ratio that re-weights old-policy samples to estimate the new policy's advantage; it improves the
policy by conservatively mixing the greedy policy into the current one. The ratio is what makes
the surrogate an off-policy estimate computable from a fixed batch. **Gap:** maximizing
`E_t[r_t Â_t]` with no leash drives `r_t` far from 1 — to `+∞` where `Â>0`, toward 0 where
`Â<0` — which is exactly the destructive update; the surrogate is only a faithful proxy for the
true improvement while the policy stays close to `π_old`.

**Trust region policy optimization (Schulman et al. 2015b).** Maximize the same surrogate but
under an explicit constraint on the policy change measured in KL,

```
maximize_θ  E_t[ r_t(θ) Â_t ]     subject to   E_t[ KL[π_old(·|s_t), π_θ(·|s_t)] ] ≤ δ.
```

This comes with monotonic-improvement theory and works well: the KL constraint is precisely the
leash CPI lacks. It is solved approximately by linearizing the objective and taking a quadratic
(Fisher-information) approximation to the KL constraint, then a conjugate-gradient step with a
line search. The theory actually suggests the *penalized* form `E_t[r_t Â_t] − β KL[π_old, π_θ]`
for some coefficient `β`, because a certain KL-penalized surrogate lower-bounds the true return;
but a hard constraint is used instead because it is hard to pick a single `β` that behaves well
across different problems, or even within one problem as its characteristics change over training.
**Gap:** the method is second-order. It needs Fisher-vector products and a conjugate-gradient
inner loop per update, which is complicated to implement and expensive; and the machinery does not
play well with architectures that share parameters between policy and value function, use dropout,
or carry auxiliary losses. It also does not naturally do *multiple epochs of plain SGD* on the
collected batch — the very thing that would buy sample efficiency. So the method that best
controls the update size is exactly the one that fails the scalability and simplicity goals; and
the penalized form it points to is left unused because a *fixed* `β` does not hold a target update
size steady across problems or across a run.

**Adaptive-stepsize vanilla policy gradient.** Run vanilla PG but, after each batch, adjust the
optimizer stepsize up or down by
looking at how much the update changed the policy, measured in KL. **Gap:** it inherits vanilla
PG's lack of any in-objective protection against bad directions; the stepsize feedback alone does
not stop a single large or ill-directed step from hurting, and there is no surrogate that remains
a valid proxy under multi-epoch reuse of the batch.

## Evaluation settings

The natural yardsticks already in use at the time:

- **MuJoCo continuous-control tasks** (Todorov et al. 2012) in OpenAI Gym (Brockman et al. 2016):
  a suite such as HalfCheetah, Hopper, InvertedDoublePendulum, InvertedPendulum, Reacher,
  Swimmer, Walker2d (the "-v1" set), each run for about one million timesteps, scored by average
  total reward over the last 100 episodes. Policies are fully-connected MLPs (e.g. two hidden
  layers of 64 units, tanh) emitting the mean of a Gaussian with learned standard deviation. Each
  algorithm is run with several random seeds and scores are normalized so a random policy is 0 and
  the best result is 1, then averaged across environments. A standard yardstick for comparing
  on-policy optimizers.
- **High-dimensional 3D-humanoid locomotion / steering tasks** (Roboschool-style): a simulated
  humanoid that must run, steer toward a target whose position changes periodically, and get up
  off the ground, possibly while being perturbed. Trained for tens to hundreds of millions of
  timesteps with many parallel actors and a per-dimension log-standard-deviation that is annealed
  over training. The high-dimensional stress test for whether an on-policy method scales.
- **Atari (Arcade Learning Environment, Bellemare et al. 2015)** with a shared convolutional
  policy/value network and an entropy bonus, scored by average episode reward over training and
  over the last 100 episodes — the discrete-action, parameter-sharing stress test.
- Protocol: for each method, search the update-size knob and the optimizer settings on a cheap
  subset, fix the rest (horizon `T`, epochs, minibatch size, discount `γ`, GAE `λ`), and compare
  learning curves and final scores across seeds.

## Code framework

The optimizer plugs into the standard parallel-actor on-policy harness that already exists: a
rollout buffer that stores `T` steps from each of `N` environments, a GAE return/advantage
computation, a diagonal-Gaussian actor-critic network, and an Adam optimizer. What is *not*
settled is the update rule that turns one collected batch into several reliable gradient steps.
That is the empty slot. The scaffold below shows the pieces that exist before the method and
leaves the update rule as a stub.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Diagonal-Gaussian policy + value function (the standard continuous-control net)."""

    def __init__(self, num_obs, num_actions, hidden=(256, 256, 256), init_std=1.0):
        super().__init__()
        # actor MLP -> action mean; critic MLP -> scalar value
        self.actor = _mlp(num_obs, hidden, num_actions)
        self.critic = _mlp(num_obs, hidden, 1)
        self.std = nn.Parameter(init_std * torch.ones(num_actions))  # learned per-dim std
        self.distribution = None

    def update_distribution(self, obs):
        mean = self.actor(obs)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def evaluate(self, obs):
        return self.critic(obs)


class RolloutStorage:
    """Fixed-horizon buffer for N parallel actors; computes GAE returns/advantages."""

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 \
                else self.values[step + 1]
            not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + not_terminal * gamma * next_values - self.values[step]
            advantage = delta + not_terminal * gamma * lam * advantage          # GAE recursion
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs):
        ...  # yields shuffled minibatches of (obs, actions, old_log_prob, advantages,
             #  returns, old_values, old_mu, old_sigma), iterating num_epochs times


class PolicyOptimizer:
    """Turns one collected batch into a policy update. Already owns: the actor-critic,
    the rollout storage, an Adam optimizer, and the GAE hyperparameters."""

    def __init__(self, actor_critic, num_learning_epochs, num_mini_batches,
                 gamma=0.99, lam=0.95, value_loss_coef=1.0, entropy_coef=0.0,
                 learning_rate=1e-3, max_grad_norm=1.0, device="cpu"):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma, self.lam = gamma, lam
        self.value_loss_coef, self.entropy_coef = value_loss_coef, entropy_coef
        self.learning_rate, self.max_grad_norm = learning_rate, max_grad_norm

    def update(self):
        for batch in self.storage.mini_batch_generator(self.num_mini_batches,
                                                       self.num_learning_epochs):
            # re-evaluate the current policy distribution on the stored states
            self.actor_critic.update_distribution(batch.observations)
            log_prob = self.actor_critic.get_actions_log_prob(batch.actions)
            value = self.actor_critic.evaluate(batch.observations)
            # TODO: the per-batch policy-update rule we will design.
            #       Given old-policy log-probs / distribution params, the advantages, and the
            #       re-evaluated current policy, form the loss and any update bookkeeping, then:
            #         self.optimizer.zero_grad(); loss.backward()
            #         nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            #         self.optimizer.step()
            pass
        self.storage.clear()
```

The harness supplies, for every stored transition, the state, the action that was taken, the
log-probability and distribution parameters under the policy that took it, the GAE advantage, and
the return target for the critic; `update()` is iterated for `K` epochs over shuffled minibatches.
The single empty slot is the per-batch update rule.
