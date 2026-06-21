We want to learn useful image representations from unlabeled data. The best methods at the time, such as SimCLR and MoCo, use contrastive learning: they push together two augmented views of the same image and push apart views of different images. The negatives are what prevent the model from collapsing to a constant representation, but they are expensive. They require very large batches, memory banks, or hard-negative mining, and the methods become fragile to the exact choice of augmentations. The core question is whether negatives are truly necessary, or whether we can learn equally good representations just by predicting one view from another.

A naive prediction objective does collapse: if both the predictor and target are free to move, they can agree on a constant vector and get zero loss. Simply using a slow-moving target network without any other change also collapses, as the unsupervised mean-teacher setup shows. So preventing collapse without negatives needs more than just a frozen or slow target. We need an asymmetry that makes the constant solution unstable rather than optimal.

The method I propose is BYOL, Bootstrap Your Own Latent. It learns by predicting the representation of one augmented view from another, using two networks with shared architecture but different roles and different update rules. The online network has an encoder, a projector, and an additional predictor head. The target network has only an encoder and a projector; it has no predictor, and its weights are never updated by gradients. Instead, the target is an exponential moving average of the online network. For each image we sample two augmentations, pass one view through the online network and the other through the target network, l2-normalize both outputs, and minimize the squared error between the online prediction and the target projection. The loss is symmetrized by swapping the views, and gradients flow only into the online parameters.

The asymmetry is the key anti-collapse mechanism. Because the predictor sits only on the online branch, the online network is not asked to equal the target; it is asked to predict the target through an extra learned map. In the idealized case where the predictor is optimal, the online network minimizes the expected conditional variance of the target projection given the online projection. Conditioning on more information can only reduce this expected variance, so discarding information never helps. A constant online projection therefore gives the worst possible loss, not the best, which makes the collapsed equilibrium unstable. The target side is forbidden from taking the collapse step because it never minimizes the loss; it only trails the online network through the EMA. The slow target also keeps the prediction problem stable and slowly evolving, which lets the predictor stay near optimal and makes the conditional-variance argument hold in practice.

The training pipeline is straightforward. The encoder is typically a ResNet-50; the representation kept for downstream tasks is the output of its global average pooling layer. The projector and predictor are two-layer MLPs that map to a 256-dimensional output, with BatchNorm in the hidden layer but not on the final output. The loss is computed after l2-normalizing the online prediction and the target projection, which makes the squared error equivalent to two minus twice the cosine similarity. We use a cosine EMA schedule that anneals the target momentum toward one over training, and optimize the online network with LARS using a cosine learning-rate schedule and a small weight decay. At the end we discard the projector, predictor, and target network, keeping only the online encoder.

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


class MLP(hk.Module):
    """Linear(4096) -> BatchNorm -> ReLU -> Linear(256). Output is NOT batch-normed."""

    def __init__(self, name):
        super().__init__(name=name)

    def __call__(self, x, is_training):
        x = hk.Linear(4096)(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            x, is_training)
        x = jax.nn.relu(x)
        return hk.Linear(256, with_bias=False)(x)


def network(inputs, is_training):
    """Encoder -> projector -> predictor. The target reuses the same definition;
    its 'prediction' output is simply not used."""
    embedding = Encoder(name='encoder')(inputs, is_training)
    projection = MLP(name='projector')(embedding, is_training)
    prediction = MLP(name='predictor')(projection, is_training)
    return {'projection': projection, 'prediction': prediction}


net = hk.without_apply_rng(hk.transform_with_state(network))


def regression_loss(x, y, eps=1e-12):
    """l2-normalized squared error == 2 - 2 * cosine."""
    x = x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)
    y = y / jnp.maximum(jnp.linalg.norm(y, axis=-1, keepdims=True), eps)
    return 2 - 2 * jnp.sum(x * y, axis=-1)


def apply_two_views(params, state, view_1, view_2):
    out_1, state = net.apply(params, state, view_1, is_training=True)
    out_2, state = net.apply(params, state, view_2, is_training=True)
    return out_1, out_2, state


def loss_fn(online_params, target_params, online_state, target_state,
            view_1, view_2):
    o1, o2, new_online_state = apply_two_views(
        online_params, online_state, view_1, view_2)
    t1, t2, new_target_state = apply_two_views(
        target_params, target_state, view_1, view_2)
    loss = regression_loss(
        o1['prediction'], jax.lax.stop_gradient(t2['projection']))
    loss += regression_loss(
        o2['prediction'], jax.lax.stop_gradient(t1['projection']))
    return jnp.mean(loss), (new_online_state, new_target_state)


def target_ema(step, base_ema, total_steps):
    return 1 - (1 - base_ema) * (jnp.cos(jnp.pi * step / total_steps) + 1) / 2


def update_fn(state, step, view_1, view_2, optimizer, total_steps,
              base_ema=0.996):
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, (online_state, target_state)), grads = grad_fn(
        state.online_params, state.target_params,
        state.online_state, state.target_state, view_1, view_2)
    updates, opt_state = optimizer.update(
        grads, state.opt_state, state.online_params)
    online_params = optax.apply_updates(state.online_params, updates)

    tau = target_ema(step, base_ema, total_steps)
    target_params = jax.tree_util.tree_map(
        lambda t, o: tau * t + (1 - tau) * o,
        state.target_params, online_params)

    return TrainState(
        online_params=online_params,
        target_params=target_params,
        online_state=online_state,
        target_state=target_state,
        opt_state=opt_state), loss


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
    return {name: value for name, value in tree.items()
            if name.startswith('encoder')}


def main(dataset, optimizer, total_steps, augment_fn):
    rng = jax.random.PRNGKey(1337)
    rng, rng_init = jax.random.split(rng)
    dummy = next(dataset)
    state = init(rng_init, dummy, optimizer)

    for step in range(total_steps):
        images = next(dataset)
        rng, r1, r2 = jax.random.split(rng, 3)
        view_1 = augment_fn(images, r1, view_id=1)
        view_2 = augment_fn(images, r2, view_id=2)
        state, _ = update_fn(
            state, step, view_1, view_2, optimizer, total_steps)

    return keep_encoder(state.online_params), keep_encoder(state.online_state)
```
