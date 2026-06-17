# AIRL

AIRL (Adversarial Inverse Reinforcement Learning) learns a reward function from expert
demonstrations by training a GAN over single state-action transitions whose discriminator has
a special structure. The discriminator's learnable score is decomposed into a state-only
reward term `g(s)` and a potential-based shaping term `h`, which lets the recovered reward be
*disentangled* from the environment dynamics — transferable to environments with different
dynamics, where re-optimizing it still yields good policies. It is the scalable, single-
transition realization of the GAN/IRL/energy-based-model connection.

## Problem it solves

Recover a reward (not just a policy) from a fixed set of expert demonstrations on
high-dimensional continuous control with unknown dynamics, such that the reward is portable:
take it to a new environment whose dynamics differ, re-optimize a policy under it, and still
behave well. Direct imitation (GAIL) recovers only a policy — nothing to re-optimize — and
naive IRL recovers a reward entangled with the training dynamics, which silently breaks under
dynamics change.

## Key idea

Train an adversarial single-transition discriminator with the GAN-GCL structured form

```
D_{theta,phi}(s,a,s') = exp{ f_{theta,phi}(s,a,s') } / ( exp{ f_{theta,phi}(s,a,s') } + pi(a|s) ),
```

implemented as a binary classifier with logit `f(s,a,s') - log pi(a|s)` (expert = 1, policy =
0). Restrict `f` to a state reward term plus a potential-based shaping term:

```
f_{theta,phi}(s,a,s') = g_theta(s) + gamma * h_phi(s') - h_phi(s),
```

The policy is optimized (PPO/TRPO, with entropy regularization) on the reward

```
r_hat(s,a) = log D - log(1 - D) = f_{theta,phi}(s,a,s') - log pi(a|s).
```

Why each piece:

- **Single transitions, not trajectories.** The trajectory-level GAN-GCL discriminator's logit
  is a sum over the episode, so its variance compounds with the horizon and it learns poorly.
  Dropping to single `(s,a,s')` keeps the variance bounded; the single-transition discriminator
  still optimizes the MaxEnt IRL gradient (its loss gradient equals the guided-cost-learning
  cost gradient with `f` as the reward).
- **The `- log pi` logit.** Filling in the known generator density `pi(a|s)` makes the optimal
  discriminator independent of the generator (stabilizes training) and lets a reward be read
  out of `f` — unlike GAIL's generic discriminator, which is `0.5` everywhere at optimum.
- **The `g + gamma h(s') - h(s)` decomposition.** This is exactly the Ng-Harada-Russell (1999)
  potential-based shaping form, the *only* policy-invariant reward-transformation class when
  dynamics are unknown. The shaping ambiguity IRL cannot see through is poured entirely into
  `h`, leaving `g` unshaped.
- **State-only `g(s)`.** A reward is disentangled (same optimal policy under all dynamics) **iff**
  it is state-only (Theorems below). Making `g` state-only is what makes the recovered reward
  transferable.

## What it recovers

At the GAN optimum `pi = pi_E`, the discriminator is `1/2` everywhere, so `exp{f*} = pi_E(a|s)`,
giving `f*(s,a) = log pi_E(a|s) = A*(s,a)` — the (entangled) advantage. With the structured `f`,
deterministic dynamics, decomposable dynamics, and a state-only ground-truth reward `r(s)`:

```
g*(s) = r(s) + const        (the true, disentangled reward)
h*(s) = V*(s) + const        (the shaping term recovers the soft value function)
```

So `h` absorbs the value-function shaping (`f* = r(s) + gamma V*(s') - V*(s) = A*`), leaving `g`
as the clean reward. The recovery proof applies the chaining lemma to
`g*(s) - h*(s) + gamma h*(s') = r(s) - V*(s) + gamma V*(s')`, grouping
`a=g-h`, `b=gamma h`, `c=r-V`, and `d=gamma V`; therefore `g*=r+const` and
`h*=V+const`. In stochastic environments `f(s,a,s')` is a single-sample estimate of `A*(s,a)`.

## Disentanglement theory

Reward `r'` is **disentangled** wrt ground-truth `r` over a dynamics set if `pi*_{r',T} =
pi*_{r,T}` for all `T` in the set; under max causal entropy this is equivalent to
`Q*_{r',T}(s,a) = Q*_{r,T}(s,a) - f(s)`. Dynamics are **decomposable** if all states are linked
(transitive closure of "some state reaches both in one step").

**Chaining lemma.** Under decomposability, if `a(s)+b(s') = c(s)+d(s')` for all `s,s'`, then
`a = c + C1` and `b = d + C2` for constants `C1,C2`. (`a(s)-c(s) = d(s')-b(s')` is both
state-only and next-state-only; decomposability links all states, forcing both sides constant.)

**Sufficiency.** If `r(s)` is state-only, `T` decomposable, and a state-only `r'(s)` yields the
optimal policy in `T` (`Q*_{r',T} = Q*_{r,T} - f(s)`), then `r' = r + const`, so `r'` is
disentangled for all dynamics. The soft Bellman step is
`V_{r'}(s') = logsumexp_a(Q*_r(s',a)-f(s')) = V_r(s') - f(s')`, giving
`r'(s) = r(s) + gamma E_{s'}[f(s')] - f(s)`. In the deterministic/support-wise
form, `phi(s)+f(s)=gamma f(s')`, and decomposability forces `f`, hence `phi`, to be constant.

**Necessity.** If `r'(s,a,s')` is disentangled for all dynamics, it must be state-only.
(Counterexample: a 3-state MDP where an action-dependent `r'` matching `r` under one transition
structure induces a bad policy once the dynamics are permuted.)

## Algorithm

```
Obtain expert trajectories.
Initialize policy pi and discriminator D_{theta,phi} with f = g(s) + gamma*h(s') - h(s).
for step t in 1..N:
    Collect transitions (s, a, s') by executing pi.
    Train D_{theta,phi} by binary logistic regression to classify expert vs. policy
        transitions, with logit f(s,a,s') - log pi(a|s).
    Set reward r_{theta,phi}(s,a,s') <- log D - log(1-D) = f - log pi.
    Update pi w.r.t. r_{theta,phi} with any policy optimizer (PPO/TRPO, entropy-regularized).
```

Practical stabilizers: zero the potential at terminal states (`gamma*(1-done)*h(s')`) so
shaping stays policy-invariant across variable-length episodes; mix policy samples from the
previous ~20 iterations as extra discriminator negatives so the reward does not overfit to the
current policy; normalize the reward network's observation inputs with running mean/std so the
discriminator cannot separate expert from policy on raw scale.

## Working code

Fills the discriminator/reward slots of the adversarial-imitation harness: the reward network
computes the done-aware shaped score, and the trainer mirrors the standard AIRL implementation:
`logits_expert_is_high = reward_net(...) - log_policy_act_prob`, BCE labels expert samples as
1 and policy samples as 0, and the explicit sampler reward is `f - log pi`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim, hidden_sizes=(32, 32), out_dim=1):
    layers = []
    last = in_dim
    for width in hidden_sizes:
        layers += [nn.Linear(last, width), nn.ReLU()]
        last = width
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class RewardNetwork(nn.Module):
    """Shaped score f(s,a,s',done) = g(s) + gamma*(1-done)*h(s') - h(s)."""

    def __init__(self, obs_dim, action_dim, gamma=0.99, use_action=False):
        super().__init__()
        self.gamma = gamma
        self.use_action = use_action
        base_in = obs_dim + action_dim if use_action else obs_dim
        self.base = mlp(base_in, hidden_sizes=(32,))          # g, state-only by default
        self.potential = mlp(obs_dim, hidden_sizes=(32, 32))  # h

    def _base_reward(self, state, action):
        if self.use_action:
            x = torch.cat([state, action], dim=-1)
        else:
            x = state
        return self.base(x).squeeze(-1)

    def _potential(self, state):
        return self.potential(state).squeeze(-1)

    def forward(self, state, action, next_state, done):
        base_reward = self._base_reward(state, action)
        old_shaping = self._potential(state)
        new_shaping = self._potential(next_state)
        new_shaping = (1.0 - done.float()) * new_shaping
        return base_reward + self.gamma * new_shaping - old_shaping

    def unshaped(self, state, action):
        return self._base_reward(state, action)


class IRLAlgorithm:
    """Adversarial reward learner with AIRL's structured discriminator."""

    def __init__(self, reward_net, optimizer):
        self.reward_net = reward_net
        self.optimizer = optimizer

    def logits_expert_is_high(
        self,
        state,
        action,
        next_state,
        done,
        log_policy_act_prob,
    ):
        if log_policy_act_prob is None:
            raise TypeError("AIRL requires log pi(a|s) for the discriminator logit")
        f = self.reward_net(state, action, next_state, done)
        return f - log_policy_act_prob

    def policy_reward(self, state, action, next_state, done, log_policy_act_prob):
        f = self.reward_net(state, action, next_state, done)
        return f - log_policy_act_prob

    def update(self, expert_batch, policy_batch):
        state = torch.cat([expert_batch["obs"], policy_batch["obs"]])
        action = torch.cat([expert_batch["acts"], policy_batch["acts"]])
        next_state = torch.cat([expert_batch["next_obs"], policy_batch["next_obs"]])
        done = torch.cat([expert_batch["dones"], policy_batch["dones"]]).float()
        logp = torch.cat([
            expert_batch["log_policy_act_prob"],
            policy_batch["log_policy_act_prob"],
        ])

        logits = self.logits_expert_is_high(state, action, next_state, done, logp)
        labels = torch.cat([
            torch.ones(len(expert_batch["obs"]), device=logits.device),
            torch.zeros(len(policy_batch["obs"]), device=logits.device),
        ])
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "disc_loss": loss.detach(),
            "disc_acc": ((logits > 0) == labels.bool()).float().mean().detach(),
        }
```
