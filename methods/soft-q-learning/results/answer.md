# Soft Q-Learning

## Problem

Learn a stochastic, *multimodal* policy over continuous states and actions — one that captures the full range of good behaviors, not a single deterministic mode — for better exploration in multimodal reward landscapes, better robustness, and reusable pretraining/finetuning. The standard expected-return objective has a deterministic optimum and standard parametric stochastic policies (Gaussian, multinomial) are structurally unimodal.

## Key idea

Optimize the **maximum-entropy objective**, augmenting reward with the policy's entropy at every visited state:

`π*_MaxEnt = argmax_π Σ_t E_{(s,a)~ρ_π}[ r(s,a) + α H(π(·|s)) ]`.

Its optimal policy is provably an **energy-based / Boltzmann policy** whose energy is a *soft* Q-function:

`π*(a|s) = exp( (Q*_soft(s,a) − V*_soft(s)) / α )`,

with the **soft value** the log-partition (a soft maximum over actions, → hard max as `α→0`):

`V*_soft(s) = α log ∫_A exp( Q*_soft(s,a')/α ) da'`,

obeying the **soft Bellman equation** `Q*_soft(s,a) = r + γ E_{s'~p}[ V*_soft(s') ]`. The backup operator `T Q = r + γ E[ α log∫exp(Q/α) ]` is a γ-contraction in sup-norm, so the fixed point is unique.

## Final algorithm

Two intractabilities — the value integral and sampling from the EBM policy — are resolved by:

1. **Importance-sampled soft value.** `V^θ_soft(s) = α log E_{a'~q}[ exp(Q^θ_soft(s,a')/α)/q(a') ]`. With a uniform proposal on `[-1,1]^d`: `V = α(logsumexp_j Q(s,a_j)/α − log N + d·log 2)`.
2. **Soft-Bellman-error regression** (the all-`(s,a)` fixed point turned into an SGD loss via `g₁=g₂ ∀x ⟺ E_q[(g₁−g₂)²]=0`):
   `J_Q(θ) = E_{s,a}[ ½ ( Q̂^{θ̄}_soft(s,a) − Q^θ_soft(s,a) )² ]`, `Q̂^{θ̄}_soft = r + γ E_{s'}[V^{θ̄}_soft(s')]`, with delayed target params `θ̄`.
3. **Amortized SVGD sampler/actor.** Train a stochastic net `a = f^φ(ξ;s)`, `ξ~N(0,I)`, to minimize `KL(π^φ ‖ exp((Q−V)/α))`. The Stein descent direction for a particle `a`:
   `Δf^φ = E_{a~π^φ}[ κ(a, f^φ) ∇_{a'}Q_soft(s,a')|_a + α ∇_{a'}κ(a', f^φ)|_a ]`,
   (first term = pull toward high-Q actions; second = repulsion that spreads particles across modes), backpropagated into `φ` via `∂J_π/∂φ ∝ E_ξ[ Δf^φ · ∂f^φ/∂φ ]`.

The sampler is the actor; the critic update is an actor-critic backbone. DDPG is recovered by dropping the `α∇κ` repulsion (keeping only the MAP particle) and using a hard critic.

**Defaults (canonical implementation):** ADAM, Q-lr `3e-4`, policy-lr `3e-4`; replay `1e6`; minibatch `128`; two hidden layers of `128`, ReLU; target hard-copy (`τ=1`) every `1000` steps; `16` SVGD kernel particles (kernel_update_ratio `0.5`), `16` value particles; RBF kernel `κ(a,a')=exp(−‖a−a'‖²/h)`, `h = median_sq / log K` (clamped); `reward_scale` is the temperature knob (e.g. 30 for swimmer/hopper/walker, 300 for ant); tanh squashing with `Σ log(1−a²)` change-of-variables correction.

## Code

```python
import numpy as np, tensorflow as tf
EPS = 1e-6

class StochasticNNPolicy:
    """Sampler/actor f^φ: (state, gaussian noise) → action."""
    def __init__(self, env_spec, hidden_layer_sizes, squash=True, name='policy'):
        self._action_dim = flat_dim(env_spec.action_space)
        self._observation_dim = flat_dim(env_spec.observation_space)
        self._layer_sizes = list(hidden_layer_sizes) + [self._action_dim]
        self._squash = squash
        self._name = name

    def actions_for(self, observations, n_action_samples=1, reuse=False):
        n_state_samples = tf.shape(observations)[0]
        if n_action_samples > 1:
            observations = observations[:, None, :]
            latent_shape = (n_state_samples, n_action_samples, self._action_dim)
        else:
            latent_shape = (n_state_samples, self._action_dim)
        latents = tf.random_normal(latent_shape)
        with tf.variable_scope(self._name, reuse=reuse):
            raw_actions = feedforward_net((observations, latents),
                layer_sizes=self._layer_sizes, activation_fn=tf.nn.relu,
                output_nonlinearity=None)
        return tf.tanh(raw_actions) if self._squash else raw_actions


def adaptive_isotropic_gaussian_kernel(xs, ys, h_min=1e-3):
    Kx, D = xs.get_shape().as_list()[-2:]
    Ky, D2 = ys.get_shape().as_list()[-2:]
    leading_shape = tf.shape(xs)[:-2]
    diff = tf.expand_dims(xs, -2) - tf.expand_dims(ys, -3)
    dist_sq = tf.reduce_sum(diff**2, axis=-1)
    input_shape = tf.concat((leading_shape, [Kx * Ky]), axis=0)
    values, _ = tf.nn.top_k(tf.reshape(dist_sq, input_shape), k=(Kx * Ky // 2 + 1), sorted=True)
    medians_sq = values[..., -1]
    h = tf.maximum(medians_sq / np.log(Kx), h_min)
    h = tf.stop_gradient(tf.expand_dims(tf.expand_dims(h, -1), -1))
    kappa = tf.exp(-dist_sq / h)
    kappa_grad = -2 * diff / tf.expand_dims(h, -1) * tf.expand_dims(kappa, -1)
    return {"output": kappa, "gradient": kappa_grad}


class SQL:
    def _create_td_update(self):
        target_actions = tf.random_uniform((1, self._value_n_particles, self._action_dim), -1, 1)
        q_value_targets = self.qf.output_for(self._next_observations_ph[:, None, :], target_actions)
        self._q_values = self.qf.output_for(self._observations_ph, self._actions_pl, reuse=True)
        next_value = tf.reduce_logsumexp(q_value_targets, axis=1)
        next_value -= tf.log(tf.cast(self._value_n_particles, tf.float32))   # − log N
        next_value += self._action_dim * np.log(2)                          # + d log 2 (uniform proposal)
        ys = tf.stop_gradient(self._reward_scale * self._rewards_pl
                              + (1 - self._terminals_pl) * self._discount * next_value)
        bellman_residual = 0.5 * tf.reduce_mean((ys - self._q_values)**2)
        if self._train_qf:
            self._training_ops.append(tf.train.AdamOptimizer(self._qf_lr).minimize(
                bellman_residual, var_list=self.qf.get_params_internal()))

    def _create_svgd_update(self):
        actions = self.policy.actions_for(self._observations_ph,
            n_action_samples=self._kernel_n_particles, reuse=True)
        n_updated = int(self._kernel_n_particles * self._kernel_update_ratio)
        n_fixed = self._kernel_n_particles - n_updated
        fixed_actions, updated_actions = tf.split(actions, [n_fixed, n_updated], axis=1)
        fixed_actions = tf.stop_gradient(fixed_actions)
        svgd_target_values = self.qf.output_for(self._observations_ph[:, None, :], fixed_actions, reuse=True)
        squash_correction = tf.reduce_sum(tf.log(1 - fixed_actions**2 + EPS), axis=-1)
        log_p = svgd_target_values + squash_correction
        grad_log_p = tf.stop_gradient(tf.expand_dims(tf.gradients(log_p, fixed_actions)[0], axis=2))
        kernel = self._kernel_fn(xs=fixed_actions, ys=updated_actions)
        kappa = tf.expand_dims(kernel["output"], dim=3)
        action_gradients = tf.reduce_mean(kappa * grad_log_p + kernel["gradient"], reduction_indices=1)
        gradients = tf.gradients(updated_actions, self.policy.get_params_internal(),
                                 grad_ys=action_gradients)
        surrogate_loss = tf.reduce_sum([tf.reduce_sum(w * tf.stop_gradient(g))
            for w, g in zip(self.policy.get_params_internal(), gradients)])
        if self._train_policy:
            self._training_ops.append(tf.train.AdamOptimizer(self._policy_lr).minimize(
                -surrogate_loss, var_list=self.policy.get_params_internal()))
```
