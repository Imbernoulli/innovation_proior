## Research question

Inverse reinforcement learning for continuous control. Given expert demonstrations — state-action-next-state transitions from a competent MuJoCo locomotion agent, with **no reward signal attached** — learn a reward function that explains the expert, then train a policy under that learned reward and measure it against the true environment reward. The design object is the **reward-learning module**: the architecture of $R(s,a,s')$ and the IRL training procedure that fits it against expert-vs-policy data and feeds rewards to the policy optimizer. Everything else — PPO, the rollout buffer, running reward normalization, evaluation — is fixed substrate the module plugs into.

## Prior art / Background / Baselines

Three standard ways to turn demonstrations into behavior, and the gap each leaves.

- **Behavioral cloning (Pomerleau, 1991).** Supervise $\pi(a\mid s)$ on expert $(s,a)$ pairs. Gap: it learns only under the expert's state distribution; deviations put the learner in states it never trained on, and errors compound over the horizon.
- **Maximum-causal-entropy inverse RL (Ziebart et al., 2008/2010).** Fit a cost under which the expert trajectory is optimal, choosing the most randomized trajectory distribution that matches the expert's feature expectations. Gap: the likelihood gradient requires the partition function over all trajectories, so each update needs a full RL solve in an inner loop and remains expensive.
- **Apprenticeship / feature-expectation matching (Abbeel & Ng, 2004).** Find a policy whose expected features match the expert's under a restricted class of costs. Gap: the recovered cost is only as expressive as the chosen class; a low-dimensional linear class usually cannot pin down the expert.

## Fixed substrate / Code framework

A custom PPO loop is frozen and must not be touched. It loads expert demos (`obs, acts, next_obs, dones` as GPU tensors), builds a `PolicyNetwork` (Tanh actor-critic: 256-256 critic, 256-256 actor mean, state-independent `actor_logstd`, Gaussian actions), and runs CleanRL-style PPO — GAE ($\gamma=0.99$, $\lambda=0.95$), clipped surrogate ($\epsilon=0.2$), value loss, Adam at $3\times10^{-4}$). During rollouts the loop **replaces the environment reward with the module's learned reward**: it calls `irl.compute_reward(obs, act, next_obs)` every transition and stores only that in the buffer. After each rollout it calls `irl.update(...)` `n_irl_updates_per_round` times on the freshest on-policy batch, then applies fixed running mean/std normalization to the buffer rewards before the PPO update. The loop also exposes the policy to the module via `set_policy(policy, optimizer)` if defined. A parameter-count check caps the reward net at $\approx1.05\times$ the largest baseline's size. Args (`irl_lr`, `irl_batch_size`, `n_irl_updates_per_round`, …) are fixed outside the editable region; any method that wants a larger effective batch or more discriminator updates must arrange it inside `update()`.

## Editable interface

Only one region is editable: the `RewardNetwork` and `IRLAlgorithm` classes in `custom_irl.py` (lines 231-357). The contract the fixed loop relies on: `RewardNetwork.forward(state, action, next_state) -> (batch,)`; and on the algorithm, `self.reward_net` set to a `RewardNetwork`, `compute_reward(obs, acts, next_obs) -> (batch,)` (called every rollout step), and `update(policy_obs, policy_acts, policy_next_obs, policy_dones) -> dict` (called to train the reward net on expert-vs-policy data). Optionally `set_policy(policy, optimizer)`.

The starting point is a placeholder scaffold: an MLP over $[s,a,s']$ trained against a **zero loss** that learns nothing. Every method on the ladder replaces exactly these two classes.

```python
# EDITABLE region of custom_irl.py (lines 231-357) — default fill (placeholder, learns nothing)
class RewardNetwork(nn.Module):
    """Reward network R(s, a, s') -> scalar. forward must stay (batch,)."""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        input_dim = obs_dim + action_dim + obs_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action, next_state):
        x = torch.cat([state, action, next_state], dim=-1)
        return self.net(x).squeeze(-1)


class IRLAlgorithm:
    """Inverse RL / reward learner. Trains reward_net; serves rewards to PPO."""

    def __init__(self, reward_net, expert_demos, obs_dim, action_dim, device, args):
        self.reward_net = reward_net
        self.expert_demos = expert_demos
        self.device = device
        self.args = args
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=args.irl_lr)
        self.total_updates = 0

    def compute_reward(self, obs, acts, next_obs):
        with torch.no_grad():
            return self.reward_net(obs, acts, next_obs)

    def update(self, policy_obs, policy_acts, policy_next_obs, policy_dones):
        self.total_updates += 1
        batch_size = self.args.irl_batch_size
        n_expert = len(self.expert_demos["obs"])
        expert_idx = torch.randint(0, n_expert, (batch_size,))
        expert_obs = self.expert_demos["obs"][expert_idx]
        expert_acts = self.expert_demos["acts"][expert_idx]
        expert_next_obs = self.expert_demos["next_obs"][expert_idx]
        n_policy = len(policy_obs)
        policy_idx = torch.randint(0, n_policy, (batch_size,))
        gen_obs = policy_obs[policy_idx]
        gen_acts = policy_acts[policy_idx]
        gen_next_obs = policy_next_obs[policy_idx]
        loss = torch.tensor(0.0, device=self.device)   # placeholder — learns nothing
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"irl_loss": loss.item()}
```

## Evaluation settings

Three Gymnasium MuJoCo locomotion environments — **HalfCheetah-v4**, **Hopper-v4**, and **Walker2d-v4** — each over three seeds {42, 123, 456}, using the bundled pre-generated expert demonstrations. Each run trains the reward module and the PPO policy under the learned reward for the fixed budget, then evaluates the policy under the **true environment reward**. Metric: mean episodic return over evaluation episodes, **higher is better**, reported per environment.
