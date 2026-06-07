# Step 1 answer — vanilla PPO (no bonus)

**Method.** The fixed PPO loop on the clipped extrinsic reward, with the intrinsic-bonus module
left empty. Exploration is entirely undirected: the stochastic policy plus the entropy term. No
intrinsic reward, no second value head, no novelty memory.

**Bonus module.** `compute_bonus` returns zero (no intrinsic signal); `loss` trains nothing;
`mix_advantages` returns the extrinsic advantage unmodified ($A = A_{\text{ext}}$).

**What it tests.** The floor: how far pure policy-gradient exploration gets on each game. It is the
weakest baseline by design — it has no mechanism for directed exploration, so on the games where
directed exploration *is* the task it should mostly find nothing.

Full derivation and the PPO clipped-surrogate / GAE details: `methods/ppo/`.

```python
# intrinsic-bonus module: the trivial (empty) one
class IntrinsicBonusModule(nn.Module):
    def __init__(self, action_dim, device, args):
        super().__init__()
    def initialize(self, envs): pass
    def trainable_parameters(self): return []
    def update_batch_stats(self, batch_obs, batch_next_obs): pass
    def compute_bonus(self, obs, next_obs, actions):
        return torch.zeros(obs.shape[0], device=obs.device)   # no intrinsic reward
    def normalize_rollout_rewards(self, rollout_intrinsic):
        return rollout_intrinsic
    def loss(self, batch_obs, batch_next_obs, batch_actions):
        return torch.tensor(0.0)

def mix_advantages(ext_advantages, int_advantages, args):
    return ext_advantages                                     # extrinsic only
```
