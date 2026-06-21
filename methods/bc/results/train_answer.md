The problem is to learn a competent control policy from expert demonstrations alone, with no reward signal and no environment interaction during training. In this setting the natural first instinct is inverse reinforcement learning: recover a reward that explains the expert and then optimize it. Earlier rungs of this ladder do exactly that. GAIL casts imitation as occupancy matching through a discriminator, but on clean expert demos the discriminator wins the min-max game, its logits saturate, and the reward it provides collapses into an uninformative flat signal. AIRL improves on GAIL by structuring the reward into a reward term plus potential-based shaping and by adding input and output normalization, yet the adversarial objective remains non-stationary by construction. On terminating bodies such as Hopper and Walker2d this instability dominates: most seeds get trapped in degenerate basins before a usable reward ever emerges. The covariate-shift error of naive imitation is real, but when the demonstrations are dense and the expert is competent, the price paid for adversarial instability can be larger than the price paid for supervised drift.

The simpler move is to stop learning a reward altogether and directly fit the policy to the expert's actions. This method is Behavioral Cloning, or BC. It reduces imitation to maximum-likelihood supervised learning on the expert's state-action pairs. For each demonstration state, the policy outputs a distribution over actions; the training objective is the negative log-probability of the expert action under that distribution. Because the policy is Gaussian, its log-probability depends on both the mean and the log-standard-deviation, so the loss trains the action center and the action spread together. That gives better coverage than a brittle point regression and prevents the policy from collapsing to a deterministic guess. The reward network is left as a dummy that returns zeros, so the fixed PPO loop still runs but contributes almost nothing; all meaningful learning happens in the supervised BC steps.

The implementation samples batches from the stored expert demonstrations, evaluates the log-probability of the expert actions under the current policy, and takes several gradient steps on the negative log-likelihood plus a small entropy bonus. Gradient clipping keeps the updates stable. There is no discriminator to balance, no inner RL loop, and no non-stationary reward to normalize. The trade-off is explicit: accept bounded covariate-shift drift in exchange for a clean, stationary objective. On clean dense demos that trade-off wins.

```python
import torch
import torch.nn as nn


class RewardNetwork(nn.Module):
    """Dummy reward network for Behavioral Cloning.

    BC does not learn a reward; this returns zeros so the fixed PPO loop
    runs but does not contribute any meaningful policy gradient.
    """

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.dummy = nn.Linear(1, 1)  # keeps the interface consistent

    def forward(self, state, action, next_state):
        return torch.zeros(state.shape[0], device=state.device)


class IRLAlgorithm:
    """Behavioral Cloning (BC): supervised maximum-likelihood cloning.

    Trains the policy network to reproduce expert actions by minimizing the
    negative log-probability under the policy's Gaussian distribution.
    """

    def __init__(self, reward_net, expert_demos, obs_dim, action_dim, device, args):
        self.reward_net = reward_net
        self.expert_demos = expert_demos
        self.device = device
        self.args = args
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.total_updates = 0
        self._policy = None
        self._policy_optimizer = None

    def set_policy(self, policy, optimizer):
        self._policy = policy
        self._policy_optimizer = optimizer

    def compute_reward(self, obs, acts, next_obs):
        return torch.zeros(obs.shape[0], device=self.device)

    def update(self, policy_obs, policy_acts, policy_next_obs, policy_dones):
        self.total_updates += 1
        if self._policy is None:
            return {"irl_loss": 0.0}

        batch_size = self.args.irl_batch_size
        n_expert = len(self.expert_demos["obs"])
        n_bc_steps = 20
        total_bc_loss = 0.0

        for _ in range(n_bc_steps):
            idx = torch.randint(0, n_expert, (batch_size,))
            expert_obs = self.expert_demos["obs"][idx]
            expert_acts = self.expert_demos["acts"][idx]

            _, log_prob, entropy, _ = self._policy.get_action_and_value(
                expert_obs, expert_acts
            )

            neglogp = -log_prob.mean()
            ent_bonus = -0.001 * entropy.mean()
            bc_loss = neglogp + ent_bonus

            self._policy_optimizer.zero_grad()
            bc_loss.backward()
            nn.utils.clip_grad_norm_(self._policy.parameters(), 0.5)
            self._policy_optimizer.step()
            total_bc_loss += bc_loss.item()

        return {"irl_loss": total_bc_loss / n_bc_steps}
```
