Before I add any exploration machinery I should pin down what the bare loop does, because the failure of the no-bonus agent is what defines the problem and forces every later step. The scaffold already hands me a complete two-stream PPO — the policy/value network, the Atari preprocessing, both value heads, the optimizer — and leaves exactly one hole: the intrinsic bonus. The honest starting point is to leave that hole empty and see where pure PPO breaks.

So the "method" at step 1 is the degenerate one: the intrinsic bonus is identically zero, and the advantage the policy ascends is the extrinsic stream alone. Let me reason about what that must do on these games. PPO ascends $\mathbb{E}[\nabla\log\pi\cdot A]$ where the advantage $A$ is built from the environment reward. When reward is dense, exploration is free — the stochastic policy plus the entropy bonus jiggle the agent around, it bumps into rewards, and the advantage points somewhere useful. When reward is *sparse*, the advantage is almost always zero: there is nothing to ascend, because the agent has seen no reward to be advantaged over. The only exploration left is the undirected noise of the action distribution, and on a hard-exploration game that noise will not carry the agent across hundreds of reward-free steps to the first payoff. So vanilla PPO with no bonus is the weakest thing I can run *by construction* — it has no mechanism for directed exploration, and on the games where directed exploration is the whole problem it should mostly find nothing, and may even be dragged negative on the deceptive game where wrong moves carry penalties.

Concretely, the no-bonus baseline is the scaffold default verbatim: `compute_bonus` returns zeros, the module trains nothing, and `mix_advantages` drops the intrinsic stream so the agent optimizes only `ext_coef * ext_advantages`.

```python
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
        return torch.zeros(obs.shape[0], device=self.device)

    def normalize_rollout_rewards(self, rollout_intrinsic) -> torch.Tensor:
        return torch.zeros_like(rollout_intrinsic)

    def loss(self, batch_obs, batch_next_obs, batch_actions) -> torch.Tensor:
        return torch.zeros((), device=self.device)


def mix_advantages(ext_advantages, int_advantages, args: Args) -> torch.Tensor:
    return args.ext_coef * ext_advantages
```

This is the right place to begin: establish the no-bonus agent, read exactly where and how it fails, and let each subsequent design step be forced by the specific failure of the one before it. I expect the games to split — wherever the first reward is reachable by random action noise, PPO should get traction; wherever it is not, the agent should be at the mercy of luck, and on the deceptive game possibly worse than idle. Whatever the exact split, the next step is already pointed at: manufacture a reward signal where the environment gives none.
