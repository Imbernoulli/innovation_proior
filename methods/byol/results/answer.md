# BYOL — Bootstrap Your Own Latent

## Problem it solves

Self-supervised image representation learning without negative pairs. Contrastive methods prevent the trivial "constant representation" collapse by discriminating a positive view against many negatives, which forces large batches, memory banks, or hard-negative mining, and makes them sensitive to the augmentation set. BYOL learns by *predicting* one view's representation from another's — no negatives — and still avoids collapse.

## Key idea

Two parameter sets share the encoder+projector architecture, with an extra predictor only on the online path:
- **Online** network θ: encoder f_θ → projector g_θ → predictor q_θ.
- **Target** network ξ: encoder f_ξ → projector g_ξ (no predictor — the architecture is asymmetric).

The online network is trained to predict the target network's projection of another augmented view. The target's weights are an exponential moving average (EMA) of the online weights, and the gradient never flows into the target (stop-gradient). Two ingredients make collapse-without-negatives work:
1. **The predictor q on the online branch only.** With an optimal predictor q*(z_θ) = E[z'_ξ | z_θ], the online update follows the gradient of the expected conditional variance E[Σ_i Var(z'_{ξ,i} | z_θ)]. Since conditioning on more information can only lower expected conditional variance, E Var(X|Y,Z) ≤ E Var(X|Y), discarding information can never reduce the profiled loss; a constant online projection is the *worst* point, so the collapsed equilibrium is unstable.
2. **The slow EMA target.** It keeps the prediction problem changing slowly so the predictor stays near-optimal (the condition under which the variance argument holds). The dynamics are not gradient descent on any joint loss over (θ, ξ): θ minimizes the loss, but ξ only trails θ — so the collapse direction (which would minimize the loss over ξ by making the target constant) is never taken.

This was seeded by the observation that training a network to predict a *fixed randomly initialized* target yields a representation far better than the random target itself — so prediction alone improves a representation, and iterating the bootstrap (target = slow average of the online net) keeps improving it.

## Algorithm

For each image x, sample two augmentations t ~ T, t' ~ T', giving views v = t(x), v' = t'(x).

1. Online: y_θ = f_θ(v), z_θ = g_θ(y_θ), prediction q_θ(z_θ).
2. Target: y'_ξ = f_ξ(v'), z'_ξ = g_ξ(y'_ξ)  (stop-gradient).
3. l2-normalize and compute
   L = ‖q̄_θ(z_θ) − z̄'_ξ‖² = 2 − 2·⟨q_θ(z_θ), z'_ξ⟩ / (‖q_θ(z_θ)‖·‖z'_ξ‖).
4. Symmetrize: swap views (v → target, v' → online) to get L̃; total loss L^BYOL = L + L̃.
5. Updates:
   - θ ← optimizer(θ, ∇_θ L^BYOL, η)   (online only)
   - ξ ← τ ξ + (1 − τ) θ               (EMA target)

τ is annealed from τ_base toward 1: τ = 1 − (1 − τ_base)·(cos(πk/K) + 1)/2.

At the end, keep only the encoder f_θ as the representation.

## Relation to InfoNCE

For one ordered anchor direction, write the contrastive score to maximize as

InfoNCE^{α,β} = (2/B)Σ_i S(v_i,v'_i) − β(2α/B)Σ_i log(Σ_{j≠i} exp(S(v_i,v_j)/α) + Σ_j exp(S(v_i,v'_j)/α)).

With β = 1, dividing by 2α gives the usual log-softmax InfoNCE score (so the training loss is its negative, up to constants). With β = 0, the negative log-denominator disappears and the score is just (2/B)Σ_i S. BYOL is this β = 0 column with S(v,v') = cosine(q_θ(g_θ(f_θ(v))), g_ξ(f_ξ(v'))), plus the swapped view direction; minimizing ‖q̄_θ(z_θ) − z̄'_ξ‖² = 2 − 2S is equivalent to maximizing that β = 0 score.

**Key hyperparameters (ResNet-50, batch 4096, 1000 epochs):** projector/predictor = Linear(4096) → BN → ReLU → Linear(256), output not batch-normed, depth 2; τ_base = 0.996; LARS optimizer, cosine LR decay, 10-epoch warmup, base LR 0.2 × batch/256; weight decay 1.5e-6 excluding biases and BN params from both adaptation and decay. The only batch-size dependence is BatchNorm in the encoder, so BYOL stays stable over a wide range of batch sizes.

## Working code (JAX / Haiku)

```python
from typing import NamedTuple

import jax
import jax.numpy as jnp
import haiku as hk
import optax


class TrainState(NamedTuple):
    online_params: hk.Params
    target_params: hk.Params
    online_state: hk.State
    target_state: hk.State
    opt_state: optax.OptState


class Encoder(hk.Module):
    """Encoder slot. Full ImageNet runs use a ResNet-50 torso here."""
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x, is_training):
        bn = dict(create_scale=True, create_offset=True, decay_rate=0.9)
        x = x.astype(jnp.float32)
        for channels in (64, 128, 256, 512):
            x = hk.Conv2D(channels, kernel_shape=3, stride=2,
                          padding='SAME', with_bias=False)(x)
            x = hk.BatchNorm(**bn)(x, is_training=is_training)
            x = jax.nn.relu(x)
        return jnp.mean(x, axis=(1, 2))


def network(inputs, is_training):
    """Encoder f -> projector g -> predictor q. The target reuses the same
    definition; its 'prediction' output is simply not used."""
    embedding = Encoder(name='encoder')(inputs, is_training)     # representation y (kept for downstream)
    projection = MLP(name='projector')(embedding, is_training)   # z = g(y)
    prediction = MLP(name='predictor')(projection, is_training)  # q(z); used only on the online branch
    return {'projection': projection, 'prediction': prediction}


class MLP(hk.Module):
    """Linear(4096) -> BatchNorm -> ReLU -> Linear(256). Output is NOT batch-normed."""
    def __init__(self, name):
        super().__init__(name=name)

    def __call__(self, x, is_training):
        x = hk.Linear(4096)(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(x, is_training)
        x = jax.nn.relu(x)
        return hk.Linear(256, with_bias=False)(x)


net = hk.without_apply_rng(hk.transform_with_state(network))


def regression_loss(x, y, eps=1e-12):
    """l2-normalized squared error == 2 - 2*cosine."""
    x = x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)
    y = y / jnp.maximum(jnp.linalg.norm(y, axis=-1, keepdims=True), eps)
    return 2 - 2 * jnp.sum(x * y, axis=-1)


def apply_two_views(params, state, view_1, view_2):
    out_1, state = net.apply(params, state, view_1, is_training=True)
    out_2, state = net.apply(params, state, view_2, is_training=True)
    return out_1, out_2, state


def loss_fn(online_params, target_params, online_state, target_state, view_1, view_2):
    o1, o2, new_online_state = apply_two_views(online_params, online_state, view_1, view_2)
    t1, t2, new_target_state = apply_two_views(target_params, target_state, view_1, view_2)
    # Gradient flows into the online network only; stop_gradient marks the target as fixed here.
    loss = regression_loss(o1['prediction'], jax.lax.stop_gradient(t2['projection']))
    loss += regression_loss(o2['prediction'], jax.lax.stop_gradient(t1['projection']))  # symmetrized
    return jnp.mean(loss), (new_online_state, new_target_state)


def target_ema(step, base_ema, total_steps):
    return 1 - (1 - base_ema) * (jnp.cos(jnp.pi * step / total_steps) + 1) / 2


def update_fn(state, step, view_1, view_2, optimizer, total_steps, base_ema=0.996):
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, (online_state, target_state)), grads = grad_fn(
        state.online_params, state.target_params,
        state.online_state, state.target_state, view_1, view_2)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.online_params)
    online_params = optax.apply_updates(state.online_params, updates)

    tau = target_ema(step, base_ema, total_steps)
    target_params = jax.tree_util.tree_map(
        lambda t, o: tau * t + (1 - tau) * o,  # xi <- tau*xi + (1-tau)*theta
        state.target_params, online_params)
    new_state = TrainState(
        online_params=online_params,
        target_params=target_params,
        online_state=online_state,
        target_state=target_state,
        opt_state=opt_state)
    return new_state, loss


def init(rng, dummy_input, optimizer):
    online_params, online_state = net.init(rng, dummy_input, is_training=True)
    opt_state = optimizer.init(online_params)
    return TrainState(
        online_params=online_params,
        target_params=online_params,
        online_state=online_state,
        target_state=online_state,
        opt_state=opt_state)


def keep_encoder(tree):
    return {name: value for name, value in tree.items() if name.startswith('encoder')}


def main(dataset, optimizer, total_steps, augment_fn):
    rng = jax.random.PRNGKey(1337)
    rng, rng_init = jax.random.split(rng)
    dummy = next(dataset)
    state = init(rng_init, dummy, optimizer)

    for step in range(total_steps):
        images = next(dataset)
        rng, r1, r2 = jax.random.split(rng, 3)
        view_1 = augment_fn(images, r1, view_id=1)   # asymmetric view distributions T, T'
        view_2 = augment_fn(images, r2, view_id=2)
        state, _ = update_fn(state, step, view_1, view_2, optimizer, total_steps)

    return keep_encoder(state.online_params), keep_encoder(state.online_state)
```
