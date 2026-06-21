The problem is to recover a reward function, not just a policy, from a fixed set of expert demonstrations in high-dimensional continuous control. A policy clone can reproduce the expert in the training environment, but it cannot be re-optimized under new dynamics, so it is not portable. Maximum-entropy inverse reinforcement learning gives the right probabilistic objective, but its gradient contains an intractable partition-function term over trajectories. Earlier adversarial formulations either operate at the trajectory level, where importance weights blow up the variance, or use a generic single-transition discriminator that collapses to 1/2 everywhere and leaves no reward to extract. The remaining gap is a scalable, single-transition adversarial reward learner that explicitly separates the dynamics-invariant reward from the dynamics-dependent shaping that makes rewards non-transferable.

The method is Adversarial Inverse Reinforcement Learning (AIRL). It trains a binary discriminator over individual transitions, but instead of a free-form classifier it uses the structured discriminator D_{theta,phi}(s,a,s') = exp(f) / (exp(f) + pi(a|s)), implemented as a sigmoid whose input is f(s,a,s') - log pi(a|s). The score f is split into a state-only reward term g_theta(s) plus a potential-based shaping term gamma h_phi(s') - h_phi(s). The policy is optimized with PPO or TRPO on the reward log D - log(1 - D), which simplifies to f - log pi, exactly the entropy-regularized objective that pushes the policy toward the current reward model while maintaining exploration.

This construction solves all the pieces at once. Discriminating single transitions keeps the variance bounded while still optimizing the MaxEnt IRL gradient, because the loss gradient of this special discriminator matches the guided-cost-learning cost gradient with f playing the role of the reward. Subtracting log pi(a|s) from the logit makes the optimal discriminator independent of the generator, which stabilizes training and lets f be read as a reward at convergence. At the global optimum pi = pi_E, the discriminator is 1/2 everywhere and f* equals log pi_E(a|s), the advantage A*. Without further structure that advantage is useless for transfer because it embeds the training value function V*. The decomposition f = g(s) + gamma h(s') - h(s) is exactly the Ng-Harada-Russell potential-based shaping class, the only policy-invariant reward ambiguity that IRL cannot see through. The optimization can therefore pour all of the value-function shaping into h, leaving g as the clean, state-only reward. Under deterministic, decomposable dynamics g* recovers the true reward up to an additive constant and h* recovers V* up to a constant. To keep the shaping valid across variable-length episodes, the future potential is zeroed at terminal transitions using gamma (1 - done) h(s').

The training loop alternates three steps. First, collect on-policy transitions from the current policy. Second, train the discriminator with binary cross-entropy, labeling expert transitions as 1 and policy transitions as 0, with logits f - log pi. Third, convert the discriminator into the per-step reward f - log pi and take a policy-optimization step. In practice the reward network benefits from a running normalization on observations so the discriminator cannot separate expert from policy on raw scale, and mixing in policy samples from recent iterations prevents the reward from overfitting to the latest rollout.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim, hidden_sizes=(256, 256), out_dim=1):
    layers = []
    last = in_dim
    for width in hidden_sizes:
        layers += [nn.Linear(last, width), nn.ReLU()]
        last = width
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class RewardNetwork(nn.Module):
    """Shaped score f(s,a,s',done) = g(s) + gamma*(1-done)*h(s') - h(s)."""

    def __init__(self, obs_dim, action_dim, gamma=0.99):
        super().__init__()
        self.gamma = gamma
        # g: state-only reward (the transferable part)
        self.base = mlp(obs_dim, hidden_sizes=(256, 256))
        # h: potential function absorbing the value-function shaping
        self.potential = mlp(obs_dim, hidden_sizes=(256, 256))

    def _base_reward(self, state):
        return self.base(state).squeeze(-1)

    def _potential(self, state):
        return self.potential(state).squeeze(-1)

    def forward(self, state, action, next_state, done):
        base_reward = self._base_reward(state)
        old_shaping = self._potential(state)
        new_shaping = self._potential(next_state)
        new_shaping = (1.0 - done.float()) * new_shaping
        return base_reward + self.gamma * new_shaping - old_shaping

    def unshaped(self, state):
        return self._base_reward(state)


class IRLAlgorithm:
    """AIRL discriminator and reward provider."""

    def __init__(self, reward_net, optimizer):
        self.reward_net = reward_net
        self.optimizer = optimizer

    def logits_expert_is_high(self, state, action, next_state, done,
                              log_policy_act_prob):
        if log_policy_act_prob is None:
            raise TypeError("AIRL requires log pi(a|s)")
        f = self.reward_net(state, action, next_state, done)
        return f - log_policy_act_prob

    def policy_reward(self, state, action, next_state, done,
                      log_policy_act_prob):
        f = self.reward_net(state, action, next_state, done)
        return f - log_policy_act_prob

    def update(self, expert_batch, policy_batch):
        state = torch.cat([expert_batch["obs"], policy_batch["obs"]])
        action = torch.cat([expert_batch["acts"], policy_batch["acts"]])
        next_state = torch.cat([expert_batch["next_obs"],
                                policy_batch["next_obs"]])
        done = torch.cat([expert_batch["dones"],
                          policy_batch["dones"]]).float()
        logp = torch.cat([
            expert_batch["log_policy_act_prob"],
            policy_batch["log_policy_act_prob"],
        ])

        logits = self.logits_expert_is_high(state, action, next_state,
                                            done, logp)
        labels = torch.cat([
            torch.ones(len(expert_batch["obs"]), device=logits.device),
            torch.zeros(len(policy_batch["obs"]), device=logits.device),
        ])
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        acc = ((logits > 0) == labels.bool()).float().mean().detach()
        return {"disc_loss": loss.detach(), "disc_acc": acc}
```
