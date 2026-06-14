FiLM conditions a neural network by letting a conditioning input produce a per-feature-map affine transform for intermediate activations:

```text
gamma_{i,c} = f_c(x_i)
beta_{i,c}  = h_c(x_i)
FiLM(F_{i,c} | gamma_{i,c}, beta_{i,c}) = gamma_{i,c} * F_{i,c} + beta_{i,c}
```

`f` and `h` may be neural networks, usually implemented as one shared generator that emits both vectors. For CNN feature maps, each `gamma_{i,c}` and `beta_{i,c}` is broadcast over all spatial positions, so the mechanism uses two scalars per modulated feature map and its parameter count is independent of image resolution.

## Why this form

Concatenating a conditioning vector `z` to features before a linear layer decomposes as:

```text
W [F; z] = W_F F + W_z z
```

The `W_z z` term is a conditioning-dependent additive bias on the layer output, so after the split the concatenation case is the `gamma = 1` affine corner with `beta(z) = W_z z`. Feature-wise gating is the `beta = 0` special case with `gamma` squashed into `(0, 1)`. The full affine keeps both shift and scale unrestricted: `beta` can move a downstream ReLU threshold, `gamma > 1` amplifies, and `gamma = 0` shuts off. With a following ReLU, `gamma * F + beta > 0` becomes `F > -beta / gamma` for `gamma > 0`, but `F < -beta / gamma` for `gamma < 0`, so a negative scale flips which side of the feature distribution remains active.

The identity initialization is:

```text
gamma = 1 + Delta_gamma
```

If a generator emitted zero-centered `gamma` directly, early activations could be multiplied toward zero, and the local derivative with respect to the incoming feature would also be near zero. Emitting `Delta_gamma` instead lets a near-zero generator start at `gamma` approximately `1` and `beta` approximately `0`.

## Relation to earlier mechanisms

- Batch and instance normalization already end with the same affine form, but with learned `(gamma, beta)` that do not depend on a separate conditioning input.
- Conditional Instance Normalization selects `(gamma_s, beta_s)` from a style table and applies that conditioned affine after instance normalization.
- Conditional Batch Normalization predicts `Delta_gamma, Delta_beta` from a language embedding and adds them to frozen pretrained BatchNorm scalars before the post-normalization affine.
- Adaptive Instance Normalization uses another input's channel statistics as the conditioned affine scale and shift after instance normalization.
- Additive-bias conditioning and concatenation are the `gamma = 1` reduction.
- Bounded channel gates are the `beta = 0`, restricted-`gamma` reduction.

## Canonical layer

```python
import torch.nn as nn


class FiLM(nn.Module):
    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas
```

## Canonical concat branch

```python
if self.condition_method == 'concat':
    cond_params = film[:, :, :2 * self.module_dim]
    cond_maps = cond_params.unsqueeze(3).unsqueeze(4).expand(
        cond_params.size() + x.size()[-2:]
    )
else:
    gammas, betas = torch.split(
        film[:, :, :2 * self.module_dim], self.module_dim, dim=-1
    )
```

## Concat-FiLM baseline

The MLS-Bench `concat-film` baseline uses the `gamma = 1` additive-bias corner. The projected class embedding is added to the timestep embedding, so conditioning rides the denoiser's existing residual-block conditioning path; the extra hidden-state conditioner is a no-op.

```python
import torch.nn as nn


def prepare_conditioning(time_emb, class_emb):
    return time_emb + class_emb


class ClassConditioner(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()

    def forward(self, h, class_emb):
        return h
```
