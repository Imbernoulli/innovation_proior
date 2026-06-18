**Problem (from step 1).** The identity floor handed the value and actor a coordinate system where
proximity does not track reachability and exogenous coordinates ride in as if relevant, leaving the
most success on the table exactly where raw geometry is furthest from reachability (cube, antmaze).
The fix is a learned `phi(g)` whose geometry *is* reachability.

**Key idea.** Goal-conditioned value and temporal distance are the same object: with reward `r = -1`
until the goal, `V*(s, g) = -d*(s, g)`. So learn a value and force its form to be a distance between a
single shared encoder's codes,

```
V(s, g) = -||phi(s) - phi(g)||,   Z = R^D, l2 (a Hilbert space).
```

Training `V` to satisfy the goal Bellman equations drives `||phi(s) - phi(g)|| -> d*(s, g)`: `phi`
becomes an approximate isometry of the MDP's temporal structure. One shared symmetric encoder; `V(s,s)
= 0` and `V <= 0` for free. The goal code is `phi(g)`.

**Why these choices.**
- *Hilbert (l2), not a bare metric:* the `l2` norm comes from an inner product (`l1`/`l∞` do not), so
  the space carries directions/projections — a `phi(g) - phi(s)` direction is an actionable control
  signal downstream.
- *Symmetric is an approximation:* `d*` is a quasimetric (asymmetric); not every metric embeds in
  `l2`; and a discount is used for TD stability though `d*` is undiscounted. The fixed point is the
  best *discounted symmetric Hilbert approximation* — fine for reversible navigation, lossy on
  irreversible manipulation.
- *Expectile + TD, not a max:* offline the Bellman `max` cannot query OOD actions. The harness exposes
  a private twin `rep_critic` with an EMA `target_rep_critic`; the Hilbert value is pulled by expectile
  regression toward the target critic (in-sample max), and the critic is fit by TD to the Hilbert
  value at the next state. This shapes `phi` without touching the downstream GCIVL value.
- *sqrt floor:* `sqrt(max(squared_dist, 1e-6))` bounds the square-root gradient near zero (no NaN at
  init / for `s` near `g`).

**Hyperparameters.** `rep_dim=256`, shared `phi` = 2-member ensemble of LayerNorm MLPs `(512,512,512)
-> rep_dim`, `rep_expectile=0.7`, `discount=0.99`, twin `rep_critic` + EMA target (`tau=0.005`
maintained by the loop). Downstream GCIVL unchanged.

**What to watch.** Should beat identity decisively on antmaze-large and cube-single, tie or barely
improve on pointmaze-large (raw position already decent). Its residual weakness — vs a directed,
non-symmetric aggregator — should appear on cube manipulation, where reachability is most asymmetric.
That gap motivates the inner-product value at step 3.

```python
# EDITABLE region of dual-goal-representations/custom_train.py — step 2: Hilbert (symmetric) dual rep
class GoalRepresentation(nn.Module):
    """Hilbert (symmetric) dual goal representation.

    Shared phi for states and goals; V(s,g) = -||phi(s) - phi(g)||.
    Goal code = phi(g) averaged across the ensemble. Trained by IQL
    expectile regression with a separate MLP critic.
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
        critic_module = ensemblize(MLP, 2)
        self.rep_critic = critic_module((*self.hidden_dims, 1),
                                        activate_final=False, layer_norm=self.layer_norm)
        self.target_rep_critic = critic_module((*self.hidden_dims, 1),
                                               activate_final=False, layer_norm=self.layer_norm)

    def _hilbert_value(self, observations, goals):
        phi_s = self.phi(observations)
        phi_g = self.phi(goals)
        squared_dist = jnp.square(phi_s - phi_g).sum(axis=-1)
        v = -jnp.sqrt(jnp.maximum(squared_dist, 1e-6))
        return v

    def encode_goal(self, goals):
        phi_g = self.phi(goals)            # (2, batch, rep_dim)
        return phi_g.mean(axis=0)          # (batch, rep_dim)

    def compute_rep_loss(self, observations, goals, next_observations,
                         rewards, masks, actions=None):
        critic_input_obs = jnp.concatenate([observations, goals], axis=-1)
        if actions is not None:
            critic_input_obs = jnp.concatenate([critic_input_obs, actions], axis=-1)
        q1_t, q2_t = self.target_rep_critic(critic_input_obs)
        q1_t = q1_t.squeeze(-1)
        q2_t = q2_t.squeeze(-1)
        q_t = jax.lax.stop_gradient(jnp.minimum(q1_t, q2_t))

        v = self._hilbert_value(observations, goals)
        v_mean = v.mean(axis=0)

        adv = q_t - v_mean
        weight = jnp.where(adv >= 0, self.rep_expectile, (1 - self.rep_expectile))
        value_loss = (weight * (adv ** 2)).mean()

        next_v = self._hilbert_value(next_observations, goals)
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
