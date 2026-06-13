# Kaiming / He initialization, distilled

He (Kaiming) initialization is a one-shot, data-independent rule for the standard deviation of
a layer's initial Gaussian weights in a rectifier network. It draws each weight from
`N(0, 2/fan)`, i.e. std `√(2/fan)`, with biases zero, so that neither the forward responses nor
the backward gradients shrink or grow exponentially with depth. It is the variance-propagation
analysis of Glorot & Bengio (2010) redone for the rectifier nonlinearity, where the rectifier's
deletion of the negative half-line introduces a factor of `1/2` that the linear analysis misses
— putting it back is exactly the factor of `2` inside the square root.

## Problem it solves

Training a very deep convolutional rectifier network *from scratch*. A signal passing through
`L` layers is scaled by the product of `L` per-layer variance factors; if each layer's factor
is a constant `β`, the total is `β^L`, which diverges for `β > 1` and vanishes for `β < 1`. With
a fixed-std Gaussian (e.g. `σ = 0.01` for all layers) the factor is not one, so deep stacks
(more than ~8 conv layers) fail to converge and early-layer gradients can become tiny. In a
VGG-style model B, comparing the fixed `0.01` std with the fan-out-matched stds
`0.059/0.042/0.029/0.021` gives a conv10-to-conv2 gradient-std ratio of
`1/(5.9·4.2²·2.9²·2.1⁴) ≈ 1/(1.7×10⁴)`. The fix must be a rule, depending only on each
layer's shape, that makes the per-layer factor exactly one.

## Key idea

Track the variance of the responses (forward) and of the back-propagated gradients (backward)
across one layer, and choose `Var[w]` so each is preserved.

- **Forward.** For `y = Wx + b` with i.i.d. zero-mean symmetric weights, `b = 0`, and i.i.d.
  inputs independent of the weights, `Var[y_l] = n_l·Var[w_l]·E[x_l²]`, where `n_l = k²·c`
  (fan-in). The input is `x_l = max(0, y_{l-1})`, which is *not* zero-mean, so `E[x²] ≠ Var[x]`.
  Because `y_{l-1}` is zero-mean symmetric, the rectifier keeps half its (symmetric) mass:
  `E[x_l²] = ½·Var[y_{l-1}]`. Hence `Var[y_l] = ½·n_l·Var[w_l]·Var[y_{l-1}]`. Setting the
  per-layer factor to one, `½·n_l·Var[w_l] = 1`, gives `Var[w_l] = 2/n_l`, std `√(2/n_l)`.
- **Backward.** For `Δx_l = Ŵ_l·Δy_l` with `Δy_l = f'(y_l)·Δx_{l+1}`, the ReLU derivative is
  `0`/`1` with probability `½` each, so `E[f'²] = ½` and `Var[Δy_l] = ½·Var[Δx_{l+1}]`. Then
  `Var[Δx_l] = ½·n̂_l·Var[w_l]·Var[Δx_{l+1}]` with `n̂_l = k²·d` (fan-out). Setting
  `½·n̂_l·Var[w_l] = 1` gives std `√(2/n̂_l)`. The same `½`, from a different mechanism (the
  derivative, not the output).
- **One condition suffices.** Imposing the backward condition makes the forward product equal
  `∏ n_l/n̂_l = ∏ d_{l-1}/d_l = c_2/d_L` (telescoping), a non-diminishing constant — so it
  cannot cause the `β^L` blow-up/collapse. By symmetry, imposing the forward condition leaves
  the backward product at `d_L/c_2`. So you pick one direction; no compromise is needed.

The novelty versus the linear (Xavier) analysis is the factor of `2`: std `√(2/fan)` instead of
`√(1/fan)`. Per layer that is a factor `√2`; over `L` layers the linear-regime scale is short
by `2^{L/2}` (`≈2000×` at 22 layers, `≈32768×` at 30 layers), so the missing rectifier factor
becomes a depth-dependent multiplier rather than a harmless constant.

## Final rule

For a layer with fan `N` (fan-in `n = k²c` to preserve the forward pass, or fan-out `n̂ = k²d`
to preserve the backward pass) under a rectifier of negative slope `a`:

```
std(w) = sqrt(2 / ((1 + a^2) * N)),   b = 0,   w ~ N(0, std^2)
```

equivalently `std = gain / sqrt(N)` with the activation **gain**:

- `gain = sqrt(2)` for ReLU (`a = 0`)  →  `std = sqrt(2/N)`
- `gain = sqrt(2/(1+a^2))` for leaky/parametric ReLU of slope `a`
- `gain = 1` for a linear/symmetric unit (`a = 1`)  →  `std = sqrt(1/N)`, the Glorot value

The leaky/parametric form comes from `E[x²] = ½(1+a²)Var[y]` (positive half plus the
`a`-scaled negative half), giving `½(1+a²)·N·Var[w] = 1`. The `a = 1` corner reproduces the
linear-regime result exactly, so the linear case is the special case of this rule, not a
separate recipe.

Practical notes from the derivation: the first layer has no rectifier in front of it, so its
honest condition is `n_1·Var[w_1] = 1` (no factor `2`), but a single layer's factor does not
compound, so the same rule is used for uniformity. With an unnormalized input (e.g. range
`[−128, 128]`), the preserved variance reaches the logits undimmed and can overflow the
softmax; shrink the classifier's fully-connected layers below `√(2/n)` (or include a small
per-layer factor like `(1/128)^{1/L}`).

## Final form (code)

The canonical library primitive computes `std = gain / sqrt(fan)`:

```python
import math
import warnings
import torch
from torch import Tensor


def _calculate_fan_in_and_fan_out(tensor: Tensor):
    # conv weight [out, in, k, k]: receptive_field = k*k
    # linear weight [out, in]:     receptive_field = 1
    if tensor.dim() < 2:
        raise ValueError("fan in and fan out require at least 2 dimensions")
    num_output_fmaps = tensor.size(0)
    num_input_fmaps = tensor.size(1)
    receptive_field_size = 1
    if tensor.dim() > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size     # n   = k^2 * c
    fan_out = num_output_fmaps * receptive_field_size   # n_hat = k^2 * d
    return fan_in, fan_out


def _calculate_correct_fan(tensor: Tensor, mode: str) -> int:
    mode = mode.lower()
    if mode not in ("fan_in", "fan_out"):
        raise ValueError("mode must be 'fan_in' or 'fan_out'")
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    if nonlinearity in (
        "linear", "conv1d", "conv2d", "conv3d",
        "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
        "sigmoid",
    ):
        return 1.0
    if nonlinearity == "relu":
        return math.sqrt(2.0)                           # sqrt(2): the rectifier's lost factor
    if nonlinearity == "leaky_relu":
        slope = 0.01 if param is None else param
        return math.sqrt(2.0 / (1.0 + slope ** 2))      # sqrt(2/(1+a^2)) for slope a
    raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def kaiming_normal_(tensor: Tensor, a: float = 0,
                    mode: str = "fan_in", nonlinearity: str = "leaky_relu",
                    generator: torch.Generator | None = None) -> Tensor:
    """Fill `tensor` from N(0, std^2) with std = gain / sqrt(fan)."""
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op", stacklevel=2)
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)                          # ReLU: sqrt(2/fan); leaky: sqrt(2/((1+a^2)*fan))
    with torch.no_grad():
        return tensor.normal_(0, std, generator=generator)
```

Applied as a one-shot, shape-only initializer over a built rectifier CNN (convs preserve the
backward pass, linear layers the forward pass, biases zero, batch-norm affine at identity):

```python
import torch.nn as nn


def initialize_weights(model, config):
    """He/Kaiming init for a deep ReLU CNN: std = sqrt(2/fan), data-independent."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # fan_out = out_channels * k^2: preserves the backward-pass variance
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)            # b = 0: keeps y zero-mean symmetric
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)             # affine = identity at init
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # fan_in = in_features: preserves the forward-pass variance
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
```

## Relation to prior methods

- **Fixed-std Gaussian** (`σ` constant for all layers): the per-layer factor `N·σ²` is not one,
  so signals/gradients drift exponentially; deep nets stall. The rectifier-aware rule replaces
  the single `σ` with `√(2/fan)`, shape-dependent and factor-one.
- **Xavier / Glorot (2010)**: `std = √(1/fan)`-flavored, derived in the linear regime
  (`f' ≈ 1`). It is the `a = 1` (linear) corner of `√(2/((1+a²)fan))`; for a rectifier it is
  short by `√2` per layer, hence `2^{L/2}` over depth `L`.
- **Orthogonal / Saxe et al. (2013)**: norm-preserving via orthogonal matrices (dynamical
  isometry) in deep *linear* nets; exact in the linear case but needs a gain to offset the
  nonlinearity (which the rectifier analysis here supplies as `√2`) and a matrix factorization
  rather than a single i.i.d. draw.

The uniform variant `kaiming_uniform_` draws from `U[−bound, bound]` with `bound = √3·std` so
the variance matches `std²`; otherwise identical.
