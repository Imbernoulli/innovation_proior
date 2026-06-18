## Research question

Inverse reinforcement learning on continuous control: given a fixed pile of expert demonstrations —
state-action-next-state transitions rolled out by a competent MuJoCo locomotion agent, with **no reward
signal attached** — learn a reward function that explains the expert, then train a policy under that
learned reward and score the policy against the true (hidden) environment reward. The single thing being
designed is the **reward-learning module**: the `RewardNetwork` (the architecture of $R(s,a,s')$) and
the `IRLAlgorithm` (how that network is trained against expert-vs-policy data, and what reward it hands
to the policy optimizer). Everything else — the PPO policy learner, the rollout buffer, the running
reward normalization, the evaluation — is fixed substrate the module plugs into.

## Prior art before the first rung (imitation lineage)

The first rung reacts to the two standard ways to turn demonstrations into behavior, and to why each one
hurts on this exact setting (a fixed demo set, a policy that must act in the environment, scored on a
reward it never sees).

- **Behavioral cloning (Pomerleau, 1991).** Pool the expert's $(s,a)$ pairs and fit $\pi(a\mid s)$ by
  supervised learning — no reward, no environment interaction. It is the cheapest thing that can work,
  and it learns no reward at all. Gap: it fits single-timestep decisions under the *expert's* state
  distribution; the moment the learner's own small errors push it off that distribution, it is in
  states it never trained on, errors compound, and over a horizon $T$ the regret grows like
  $\varepsilon T^2$ rather than $\varepsilon T$. Covariate shift, structural, not fixable by more data.
- **Maximum-causal-entropy inverse RL (Ziebart et al., 2008/2010).** Don't fit per-step decisions; fit a
  *cost* under which the whole expert trajectory is optimal, picking the unique answer that is maximally
  random subject to matching the expert's return. Because the cost scores entire trajectories, wandering
  off the expert's path accumulates cost — no single-step covariate-shift trap. Gap: the maximum-
  likelihood gradient needs an expectation under the model's own trajectory distribution, i.e. the
  intractable partition function $Z$ over all trajectories; classic solvers estimate it by solving a
  full RL problem in an *inner loop* for every cost update — slow — and then hand back a cost that still
  needs RL run on it.
- **Apprenticeship / feature-expectation matching (Abbeel & Ng, 2004).** Match the expert's expected
  features by finding a policy that does at least as well as the expert across a small (usually linear)
  class of costs. Scales better than full IRL. Gap: it can only pin the expert down if a cost that truly
  explains the expert lives in that small class; for a low-dimensional linear class it usually does not,
  so the recovered policy need not match the expert.

## The fixed substrate

A custom PPO loop is frozen and must not be touched. It loads the expert demos
(`obs, acts, next_obs, dones` as GPU tensors), builds a `PolicyNetwork` (a Tanh actor-critic: a 256-256
critic, a 256-256 actor mean, and a state-independent `actor_logstd`, with Gaussian actions), and runs
the standard CleanRL-style PPO update — GAE ($\gamma=0.99$, $\lambda=0.95$), clipped surrogate
($\epsilon=0.2$), value loss, Adam at $3\times10^{-4}$. The crucial wiring: during each rollout the loop
**replaces the environment reward with the module's learned reward** — for every transition it calls
`irl.compute_reward(obs, act, next_obs)` and stores *that* in the buffer, never the true reward. After
each rollout it calls `irl.update(...)` `n_irl_updates_per_round` times on the freshest on-policy batch,
then applies a **fixed running mean/std normalization to the buffer rewards** before the PPO update. The
loop also passes the policy into the module via `set_policy(policy, optimizer)` if the module defines it
(so a method may read $\log\pi(a\mid s)$ or train the policy directly). A parameter-count check caps the
reward net at $\approx1.05\times$ the largest baseline's size. Args (`irl_lr`, `irl_batch_size`,
`n_irl_updates_per_round`, …) are **fixed outside the editable region** — a method that wants more
discriminator updates or a larger effective batch must bump them *inside* `update()`.

## The editable interface

Exactly one region is editable — the `RewardNetwork` and `IRLAlgorithm` classes in `custom_irl.py`
(lines 231-357). The contract the fixed loop relies on: `RewardNetwork.forward(state, action,
next_state) -> (batch,)`; and on the algorithm, `self.reward_net` set to a `RewardNetwork`,
`compute_reward(obs, acts, next_obs) -> (batch,)` (called every rollout step to reward the policy), and
`update(policy_obs, policy_acts, policy_next_obs, policy_dones) -> dict` (called to train the reward net
on expert-vs-policy data). Optionally `set_policy(policy, optimizer)`.

The starting point is the scaffold default: an MLP over $[s,a,s']$ trained against a **zero loss** — a
placeholder that learns nothing. Every method on the ladder replaces exactly these two classes.

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

Three Gymnasium MuJoCo locomotion environments — **HalfCheetah-v4**, **Hopper-v4**, and **Walker2d-v4**
— each over three seeds {42, 123, 456}, using the pre-generated expert demonstrations bundled with the
benchmark. Each run trains the reward module and the PPO policy under the learned reward for the fixed
budget, then evaluates the resulting policy under the **true environment reward**. Metric: mean episodic
return over evaluation episodes, **higher is better**, reported per environment. A strong method learns
a reward that generalizes across the state distribution the policy visits during training, without the
policy hacking artifacts of the learned reward.
