**Problem (from step 2).** The Hilbert rep beat the floor on antmaze/cube but its value
`-||phi(s)-phi(g)||` is rigidly *symmetric* and not a universal two-argument approximator, so it
underfits exactly where reaching is asymmetric (manipulation: one-way transitions) and where the value
surface is non-metric. The fix is a *directed, universal* aggregator.

**Key idea.** Identify a goal by all its incoming temporal distances, `phi^vee(g) = (s |-> d*(s,g))` —
provably sufficient for an optimal greedy step and invariant to exogenous (latent-preserving) noise.
Finitize it with two embeddings and a **bilinear inner-product** value:

```
V(s, g) = psi(s)^T phi(g) / sqrt(d).
```

Different maps for state and goal make it *directed* (can encode asymmetric reaching); the inner
product is *universal* for continuous two-variable functions on compact domains; `/sqrt(d)` keeps the
score order-one across `rep_dim`. The goal code is the goal branch `psi(g)`.

**Why over Hilbert.** Drops the two metric constraints — symmetry and limited expressiveness — while
keeping the same offline recipe. No `sqrt(||.||)` singularity, so no floor needed; numerically smoother
at init.

**Same recipe as step 2.** Offline, the Bellman `max` is unsafe (OOD actions), so the bilinear value is
fit by expectile regression toward the harness's private twin `target_rep_critic` (in-sample max), and
the `rep_critic` is fit by TD to the bilinear value at the next state. This shapes `phi`/`psi` via
`compute_rep_loss`; the fixed GCIVL value does not backprop into the embeddings. The bilinear value is
kept separate from the downstream control value on purpose — the inner-product head is great for a
compact relational *representation* but too constrained for policy extraction, so GCIVL keeps its own
value/actor over `psi(g)`.

**Harness naming.** In the bilinear edit the state branch is `phi` and the goal branch is `psi`; the
value is `(phi(obs) * psi(goal)).sum(-1) / sqrt(rep_dim)`, and `encode_goal` returns `psi(goal)`
averaged over the 2-member ensemble.

**Hyperparameters.** `rep_dim=256`, `phi`/`psi` = 2-member ensembles of LayerNorm MLPs
`(512,512,512)->rep_dim`, `rep_expectile=0.7`, `discount=0.99`, twin `rep_critic` + EMA target
(`tau=0.005`). Downstream GCIVL unchanged.

**What to watch (vs Hilbert).** Largest gain on cube-single (asymmetric reaching — the rung's strongest
claim; if cube does not improve, the asymmetry diagnosis is wrong), a clear but smaller gain on
antmaze-large (universality), roughly tied on pointmaze-large (benign reaching). Bilinear average above
Hilbert average, the improvement traceable to dropping the symmetric-metric assumption.

```python
# EDITABLE region of dual-goal-representations/custom_train.py — step 3: bilinear (inner-product) dual rep
class GoalRepresentation(nn.Module):
    """Bilinear dual goal representation.

    Separate state (phi) and goal (psi) embeddings; V(s,g) = phi(s)^T psi(g) / sqrt(d).
    Goal code = psi(g) averaged across the ensemble. Trained by IQL expectile
    regression with a separate MLP critic.
    """

    obs_dim: int
    rep_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True
    rep_expectile: float = 0.7
    discount: float = 0.99

    def setup(self):
        mlp_module = ensemblize(MLP, 2)
        self.phi = mlp_module((*self.hidden_dims, self.rep_dim),
                              activate_final=False, layer_norm=self.layer_norm)
        self.psi = mlp_module((*self.hidden_dims, self.rep_dim),
                              activate_final=False, layer_norm=self.layer_norm)
        critic_module = ensemblize(MLP, 2)
        self.rep_critic = critic_module((*self.hidden_dims, 1),
                                        activate_final=False, layer_norm=self.layer_norm)
        self.target_rep_critic = critic_module((*self.hidden_dims, 1),
                                               activate_final=False, layer_norm=self.layer_norm)

    def _bilinear_value(self, observations, goals):
        phi_s = self.phi(observations)
        psi_g = self.psi(goals)
        v = (phi_s * psi_g / jnp.sqrt(self.rep_dim)).sum(axis=-1)
        return v

    def encode_goal(self, goals):
        psi_g = self.psi(goals)            # (2, batch, rep_dim)
        return psi_g.mean(axis=0)          # (batch, rep_dim)

    def compute_rep_loss(self, observations, goals, next_observations,
                         rewards, masks, actions=None):
        critic_input_obs = jnp.concatenate([observations, goals], axis=-1)
        if actions is not None:
            critic_input_obs = jnp.concatenate([critic_input_obs, actions], axis=-1)
        q1_t, q2_t = self.target_rep_critic(critic_input_obs)
        q1_t = q1_t.squeeze(-1)
        q2_t = q2_t.squeeze(-1)
        q_t = jax.lax.stop_gradient(jnp.minimum(q1_t, q2_t))

        v = self._bilinear_value(observations, goals)
        v_mean = v.mean(axis=0)

        adv = q_t - v_mean
        weight = jnp.where(adv >= 0, self.rep_expectile, (1 - self.rep_expectile))
        value_loss = (weight * (adv ** 2)).mean()

        next_v = self._bilinear_value(next_observations, goals)
        next_v_mean = next_v.mean(axis=0)
        td_target = jax.lax.stop_gradient(rewards + self.discount * masks * next_v_mean)

        critic_input_obs_cur = jnp.concatenate([observations, goals], axis=-1)
        if actions is not None:
            critic_input_obs_cur = jnp.concatenate([critic_input_obs_cur, actions], axis=-1)
        q1, q2 = self.rep_critic(critic_input_obs_cur)
        q1 = q1.squeeze(-1)
        q2 = q2.squeeze(-1)
        critic_loss = ((q1 - td_target) ** 2 + (q2 - td_target) ** 2).mean()

        total_loss = value_loss + critic_loss
        info = {
            'rep_value_loss': value_loss,
            'rep_critic_loss': critic_loss,
            'rep_v_mean': v_mean.mean(),
        }
        return total_loss, info

    def __call__(self, goals, observations=None, next_observations=None,
                 rewards=None, masks=None, actions=None, mode='encode'):
        if mode == 'rep_loss':
            return self.compute_rep_loss(
                observations, goals, next_observations, rewards, masks, actions)
        return self.encode_goal(goals)
```
