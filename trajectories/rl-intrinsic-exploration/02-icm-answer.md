# Step 2 answer — ICM (Intrinsic Curiosity Module)

**Method.** Manufacture an intrinsic reward from forward-prediction error computed in a *learned
controllable-state space*, so curiosity is blind to uncontrollable distractors.

**Bonus module.**
- *Encoder* $\phi(s)$ — conv stack to a feature vector, the controllable state.
- *Inverse model* — from $(\phi(s_t),\phi(s_{t+1}))$ predict $a_t$ (cross-entropy). This anchors
  $\phi$ to encode only action-relevant content and prevents collapse.
- *Forward model* — from $(\phi(s_t),a_t)$ predict $\phi(s_{t+1})$;
  $i_t=\tfrac12\|\hat\phi(s_{t+1})-\phi(s_{t+1})\|^2$ is the intrinsic reward.
- *Training loss* — inverse cross-entropy $+\,0.2\cdot$ forward MSE (inverse dominates so $\phi$ is
  shaped by action-relevance, not predictability). The module is trained only on its own losses; the
  bonus is detached from the policy gradient so it can't be gamed.
- *Mixing* — $A=c_E A_{\text{ext}}+c_I A_{\text{int}}$.

**What it tests.** Whether directed, distractor-robust curiosity rescues the games where undirected
PPO found nothing — and how far a *vanishing* prediction-error bonus carries on the hardest game.

Full derivation (noisy-TV, inverse-anchors-$\phi$, loss weighting, architecture): `methods/icm/`.

```python
class IntrinsicBonusModule(nn.Module):
    def compute_bonus(self, obs, next_obs, actions):
        f_t = self.encoder(self._normalize_obs(obs))
        f_tp1 = self.encoder(self._normalize_obs(next_obs))
        pred = self.forward_model(torch.cat([f_t, self._one_hot(actions)], 1))
        return 0.5 * (pred - f_tp1).pow(2).mean(1).detach()     # forward error in phi-space

    def loss(self, batch_obs, batch_next_obs, batch_actions):
        f_t = self.encoder(self._normalize_obs(batch_obs))
        f_tp1 = self.encoder(self._normalize_obs(batch_next_obs))
        pred = self.forward_model(torch.cat([f_t, self._one_hot(batch_actions)], 1))
        inv = F.cross_entropy(self.inverse_model(torch.cat([f_t, f_tp1], 1)), batch_actions.long())
        fwd = 0.5 * (pred - f_tp1.detach()).pow(2).mean()
        return inv + 0.2 * fwd

def mix_advantages(ext_advantages, int_advantages, args):
    return args.ext_coef * ext_advantages + args.int_coef * int_advantages
```
