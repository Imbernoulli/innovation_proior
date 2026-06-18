# Performer (FAVOR+)

## Problem

Softmax attention computes

    A = exp(QK^T / sqrt(d)),   D = diag(A 1_L),   Att(Q,K,V) = D^{-1} A V.

The `L x L` matrix `A` costs `O(L^2 d)` time and `O(L^2 + Ld)` memory. The target is still dense softmax attention, not a sparse, local, or low-rank replacement.

## Method

If the score matrix can be approximated by feature-map inner products,

    A(i,j) ~= phi(q_i)^T phi(k_j),

then attention can be evaluated without forming `A`:

    Att_hat(Q,K,V) = D_hat^{-1} Q'((K')^T V),
    D_hat = diag(Q'((K')^T 1_L)).

This costs `O(Lmd)` time and `O(Lm + Ld + md)` memory for `m` features.

FAVOR+ chooses `phi` by positive orthogonal random features:

    SM(x,y) = exp(x^T y)
            = E_{omega ~ N(0,I)}[
                exp(omega^T x - ||x||^2/2)
                exp(omega^T y - ||y||^2/2)
              ],

so

    phi_+(u) = m^{-1/2} exp(-||u||^2/2)
               (exp(omega_1^T u), ..., exp(omega_m^T u)).

This estimates each softmax-kernel entry unbiasedly and keeps every estimated score nonnegative. The row-normalized output `D_hat^{-1} A_hat V` is a ratio estimator, so the exact unbiasedness claim belongs to the kernel/attention-matrix entries before row normalization.

For independent samples, the paper's MSE identities are

    MSE(SM_hat_trig) =
      (1/(2m)) exp(||x+y||^2) SM(x,y)^{-2}
      (1 - exp(-||x-y||^2))^2,

    MSE(SM_hat_+) =
      (1/m) exp(||x+y||^2) SM(x,y)^2
      (1 - exp(-||x+y||^2)),

    MSE(SM_hat_hyp+) =
      (1/2)(1 - exp(-||x+y||^2)) MSE(SM_hat_+).

Thus trigonometric features become high-variance as `SM(x,y) -> 0`, while positive features become low-variance in that regime.

Orthogonal sampling reduces MSE further. For `m <= d` within an orthogonal block,

    MSE(SM_hat_ort+) <= MSE(SM_hat_+)
      - 2(m-1)/(m(d+2))
        (SM(x,y) - exp(-(||x||^2 + ||y||^2)/2))^2.

Implementations with `m > d` stack independent `d x d` orthogonal blocks plus a partial final block. The uniform convergence theorem gives `m = Theta((d/delta^2) log(4 d^{3/4} R / delta))`, with query/key norms bounded by `R` and `delta` set by the desired entrywise error; the dependence is on `d`, `R`, and precision, not on sequence length `L`.

For causal attention, replace the global sum by a prefix sum:

    G_i = sum_{j <= i} phi(k_j) [v_j, 1]^T,
    output_i = phi(q_i)^T G_i,

which costs `O(Lmd)` and never forms the lower-triangular `L x L` mask.

## Canonical Implementation

The paper points to `google-research/google-research/performer/fast_attention`; the local reference checkout is under `methods/performer/code/google-research` at commit `4fde028f6017e16aefcbc2b6d3f77f70b9f6b421`. The following is a compact, faithful skeleton of the JAX implementation defaults and tensor operations:

```python
import math
import jax
from jax import random
import jax.numpy as jnp

SOFTMAX_DEFAULTS = dict(
    renormalize_attention=True,
    numerical_stabilizer=1e-6,
    nb_features=256,
    ortho_features=True,
    ortho_scaling=0,
    redraw_features=True,
)

GENERALIZED_DEFAULTS = dict(
    renormalize_attention=True,
    numerical_stabilizer=0.0,
    nb_features=256,
    features_type="deterministic",
    kernel_fn=jax.nn.relu,
    kernel_epsilon=1e-3,
    redraw_features=False,
)

def gaussian_orthogonal_random_matrix(key, nb_rows, nb_columns, scaling=0):
    blocks = []
    rng = key
    for _ in range(nb_rows // nb_columns):
        rng, block_key = random.split(rng)
        q, _ = jnp.linalg.qr(random.normal(block_key, (nb_columns, nb_columns)))
        blocks.append(q.T)
    remaining = nb_rows - len(blocks) * nb_columns
    if remaining:
        rng, block_key = random.split(rng)
        q, _ = jnp.linalg.qr(random.normal(block_key, (nb_columns, nb_columns)))
        blocks.append(q.T[:remaining])

    matrix = jnp.vstack(blocks)
    if scaling == 0:
        multiplier = jnp.linalg.norm(random.normal(key, (nb_rows, nb_columns)), axis=1)
    elif scaling == 1:
        multiplier = math.sqrt(float(nb_columns)) * jnp.ones((nb_rows,))
    else:
        raise ValueError("scaling must be 0 or 1")
    return jnp.diag(multiplier) @ matrix

def softmax_features(data, projection_matrix, is_query, attention_axes=(-2,), eps=1e-6):
    data_normalizer = data.shape[-1] ** -0.25
    ratio = projection_matrix.shape[0] ** -0.5
    data_dash = jnp.einsum("...d,md->...m", data_normalizer * data, projection_matrix)
    diag = jnp.sum(data ** 2, axis=-1, keepdims=True) * (data_normalizer ** 2) / 2.0
    if is_query:
        max_term = jnp.max(data_dash, axis=-1, keepdims=True)
    else:
        max_term = jnp.max(data_dash, axis=(-1,) + attention_axes, keepdims=True)
    return ratio * (jnp.exp(data_dash - diag - max_term) + eps)

def generalized_features(data, projection_matrix=None, kernel_fn=jax.nn.relu, eps=1e-3):
    if projection_matrix is None:
        return kernel_fn(data) + eps
    return kernel_fn(jnp.einsum("...d,md->...m", data, projection_matrix)) + eps

def noncausal_favor(q_prime, k_prime, value, eps=1e-6):
    z = jnp.einsum("...lm,...ld->...md", k_prime, value)
    w = jnp.einsum("...lm,...md->...ld", q_prime, z)
    normalizer = jnp.einsum("...lm,...m->...l", q_prime, k_prime.sum(axis=-2))
    normalizer = normalizer + 2 * eps * (jnp.abs(normalizer) <= eps)
    return w / normalizer[..., None]

def causal_favor(q_prime, k_prime, value, eps=1e-6):
    kv_prefix = jnp.cumsum(jnp.einsum("...lm,...ld->...lmd", k_prime, value), axis=-3)
    k_prefix = jnp.cumsum(k_prime, axis=-2)
    w = jnp.einsum("...lm,...lmd->...ld", q_prime, kv_prefix)
    normalizer = jnp.einsum("...lm,...lm->...l", q_prime, k_prefix)
    normalizer = normalizer + 2 * eps * (jnp.abs(normalizer) <= eps)
    return w / normalizer[..., None]
```

For softmax attention, use `softmax_features` for queries and keys with a redrawn orthogonal matrix when configured, then call `noncausal_favor` or `causal_favor`. For generalized attention, the reference default is deterministic `ReLU(data) + 1e-3`; iid/orthogonal projected generalized features are optional, not the default.
