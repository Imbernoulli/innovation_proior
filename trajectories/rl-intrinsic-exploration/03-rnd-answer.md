# Step 3 answer — RND (Random Network Distillation)

**Method.** Make the intrinsic reward the distillation error against a *frozen, randomly-initialized*
target network — a deterministic, in-model-class prediction target whose error is driven only by lack
of nearby training data, i.e. global novelty.

**Bonus module.**
- *Target* $f$ — conv stack to a feature vector, **frozen** at random init.
- *Predictor* $\hat f$ — deeper trainable conv stack, trained to match the target on visited states.
- *Bonus* $i_t=\|\hat f(s_{t+1})-f(s_{t+1})\|^2$ — small on seen-like states, high on globally novel
  ones. No noisy-TV trap (target is deterministic and representable).
- *Normalization* — divide $i_t$ by a running std of intrinsic returns; whiten + clip the observations
  feeding target/predictor (a frozen random net can't adapt to input scale); warm the obs-norm stats
  with a random rollout in `initialize`.
- *Training loss* — predictor MSE toward the frozen target (on a fraction of the batch).
- *Mixing* — $A=c_E A_{\text{ext}}+c_I A_{\text{int}}$ (the full method also splits episodic vs
  non-episodic returns across two value heads; see the standalone trace).

**What it tests.** Whether a *global*, slowly-decaying novelty signal — far simpler than ICM's three
coupled networks — finally gives the agent a reason to push past the first mastered region and crack
the hardest game.

Full derivation (deterministic-target argument, randomized-prior uncertainty reading, two-stream
returns, observation whitening): `methods/rnd/`.

```python
class IntrinsicBonusModule(nn.Module):
    def compute_bonus(self, obs, next_obs, actions):
        n = self._normalize_obs(next_obs)
        return ((self.target(n) - self.predictor(n)).pow(2).sum(1) / 2).detach()

    def normalize_rollout_rewards(self, rollout_intrinsic):
        disc = np.stack([self.discounted_reward.update(r)
                         for r in rollout_intrinsic.cpu().numpy()], 0).reshape(-1)
        self.reward_rms.update_from_moments(float(disc.mean()), float(disc.var()), int(disc.size))
        return rollout_intrinsic / float(np.sqrt(self.reward_rms.var + 1e-8))

    def loss(self, batch_obs, batch_next_obs, batch_actions):
        n = self._normalize_obs(batch_next_obs)
        fwd = F.mse_loss(self.predictor(n), self.target(n).detach(), reduction="none").mean(-1)
        mask = (torch.rand(len(fwd), device=self.device) < self.args.update_proportion).float()
        return (fwd * mask).sum() / torch.clamp(mask.sum(), min=1.0)

def mix_advantages(ext_advantages, int_advantages, args):
    return args.ext_coef * ext_advantages + args.int_coef * int_advantages
```
