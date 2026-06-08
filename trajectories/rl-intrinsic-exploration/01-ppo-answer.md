# Step 1 — PPO (no intrinsic bonus), distilled

**Problem.** Sparse-reward Atari: the policy gradient is built from the environment reward, which is
zero on almost every transition, so there is nothing to ascend and undirected action noise cannot
carry the agent across hundreds of reward-free steps. The base learner must at least be reliable and
data-efficient; exploration is left to a later bonus.

**Key idea (the fixed base learner).** PPO recovers trust-region reliability with a first-order loss:
the clipped surrogate $L^{CLIP}=\hat{\mathbb{E}}_t[\min(r_t\hat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)]$
with $r_t=\pi_\theta/\pi_{old}$ and $\epsilon=0.2$ — a trust region as a flat spot in the loss, no
KL/Fisher/line-search — so one batch is safely reused for $K$ epochs. Advantages are truncated GAE
($\lambda\approx0.95$); the value/entropy terms are folded in; the network shares a torso with two
value heads (extrinsic + intrinsic), valued at $\gamma=0.999$ and $\texttt{int\_gamma}=0.99$.

**Step-1 edit.** This whole loop is the scaffold's fixed substrate. The only thing the baseline edits
is the intrinsic module — left at the **default, no bonus**: `compute_bonus` returns zeros and
`mix_advantages` keeps only the extrinsic stream. It is the floor by construction.

**What to watch.** The games should split on whether the first reward is reachable by random action
noise: traction where it is, near-nothing (and possibly negative on the deceptive game) where it is
not. That failure is what forces a manufactured reward at step 2.

```python
# EDITABLE region of custom_intrinsic_exploration.py — step 1: no bonus (PPO only)
class IntrinsicBonusModule(nn.Module):
    """Baseline: no intrinsic reward."""

    def __init__(self, action_dim: int, device: torch.device, args: Args):
        super().__init__()
        self.action_dim = action_dim
        self.device = device
        self.args = args

    def initialize(self, envs) -> None:
        return None

    def trainable_parameters(self):
        return []

    def update_batch_stats(self, batch_obs: torch.Tensor, batch_next_obs: torch.Tensor) -> None:
        return None

    def compute_bonus(self, obs, next_obs, actions) -> torch.Tensor:
        return torch.zeros(obs.shape[0], device=self.device)          # no bonus

    def normalize_rollout_rewards(self, rollout_intrinsic) -> torch.Tensor:
        return torch.zeros_like(rollout_intrinsic)

    def loss(self, batch_obs, batch_next_obs, batch_actions) -> torch.Tensor:
        return torch.zeros((), device=self.device)


def mix_advantages(ext_advantages, int_advantages, args: Args) -> torch.Tensor:
    return args.ext_coef * ext_advantages                            # intrinsic stream dropped
```
