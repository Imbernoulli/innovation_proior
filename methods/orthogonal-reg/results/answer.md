# Orthogonal Regularization, distilled

Orthogonal Regularization is a soft weight regularizer for convolutional filter banks. Reshape a
conv weight tensor to a matrix `W in R^{m x n}`, with `m = out_channels` rows and
`n = in_channels * kernel_elements` columns. The row-Gram form penalizes how far the filters are
from being mutually orthonormal:

```
L_total = L_task + lambda * sum_layers sum_ij |(W W^T - I_m)_ij|
```

A common smooth variant replaces the abs sum by a squared Frobenius norm:

```
L_total = L_task + lambda * sum_layers ||W W^T - I_m||_F^2.
```

The penalty is added to the ordinary task loss before backpropagation. It needs no SVD, QR
projection, architecture change, optimizer change, or separate manifold step.

## Problem it solves

Deep and recurrent networks repeatedly multiply activations and gradients by weight matrices.
If those products contract, signals vanish; if the dynamics align with expansive directions, they
can explode. Square orthogonal matrices keep every singular value at `1`, so they preserve
Euclidean norm exactly and compose without changing scale. A rectangular conv filter bank cannot
be fully orthogonal on both sides, but it can be semi-orthogonal on the feasible side: all
nonzero singular values on that side are `1`, with unavoidable null directions on the longer
side. Orthogonal initialization gives this only at step zero; a soft regularizer keeps applying
pressure throughout training.

## Key idea

For row filters, set `R = W W^T - I_m`.

- Off-diagonal entries are filter correlations: `R_ij = <filter_i, filter_j>` for `i != j`.
- Diagonal entries are norm errors: `R_ii = ||filter_i||^2 - 1`.

Driving `R` toward zero decorrelates filters and keeps their norms near one. This controls the
shape of the spectrum, unlike ordinary L2 weight decay, which pulls all singular values toward
zero.

## Which Gram

For `W in R^{m x n}`:

- `W^T W = I_n` asks for `n` orthonormal columns in `R^m`, possible only when `m >= n`.
- `W W^T = I_m` asks for `m` orthonormal rows in `R^n`, possible only when `n >= m`.

When `n = in_channels * kernel_elements >= out_channels = m`, as in many conv layers beyond the
first, `W W^T - I_m` is the attainable row-filter penalty. If a layer is tall (`m > n`) and
feasibility is the priority, use `W^T W - I_n` instead.

## Gradient

For the squared row-Gram variant, with `R = W W^T - I`:

```
L = ||W W^T - I||_F^2
  = sum_ij (sum_a W_ia W_ja - delta_ij)^2

dL/dW_pq
  = 2 sum_ij R_ij (delta_ip W_jq + delta_jp W_iq)
  = 2 sum_j R_pj W_jq + 2 sum_i R_ip W_iq
  = 4 (R W)_pq
```

so

```
grad_W L = 4 (W W^T - I) W.
```

The column-Gram mirror is `grad_W ||W^T W - I||_F^2 = 4 W (W^T W - I)`. For the abs form,
`sum_ij |R_ij|`, a valid subgradient is `(S + S^T)W`, where `S_ij in partial |R_ij|`; since `R`
is symmetric this is `2 sign(R) W` away from zero.

## PyTorch

```python
import torch
import torch.nn as nn


_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _orthogonal_residual(W, side="rows"):
    if side == "auto":
        side = "rows" if W.size(0) <= W.size(1) else "cols"

    if side == "rows":
        gram = W @ W.t()          # W W^T: row/filter Gram
    elif side == "cols":
        gram = W.t() @ W          # W^T W: column Gram
    else:
        raise ValueError("side must be 'rows', 'cols', or 'auto'")

    eye = torch.eye(gram.size(0), device=W.device, dtype=W.dtype)
    return gram - eye


def compute_regularization(model, inputs, outputs, targets, config=None):
    """Soft orthogonal regularization for convolutional filter banks.

    Defaults to the abs row-Gram form: lambda * sum |W W^T - I|.
    Set config["orthogonal_reg_norm"] to "fro2" for the squared-Frobenius variant.
    Set config["orthogonal_reg_side"] to "auto" to choose the feasible smaller Gram.
    """
    config = config or {}
    lam = config.get("orthogonal_reg_lambda", 1e-6)
    norm = config.get("orthogonal_reg_norm", "abs")
    side = config.get("orthogonal_reg_side", "rows")

    reg = outputs.new_zeros(())
    for module in model.modules():
        if isinstance(module, _CONV_TYPES):
            W = module.weight.reshape(module.weight.size(0), -1)
            R = _orthogonal_residual(W, side=side)
            if norm == "abs":
                reg = reg + R.abs().sum()
            elif norm in {"fro2", "squared", "l2"}:
                reg = reg + (R * R).sum()
            else:
                raise ValueError("orthogonal_reg_norm must be 'abs', 'fro2', 'squared', or 'l2'")

    return lam * reg
```

Use `orthogonal_reg_lambda` as a small coefficient, commonly around `1e-6` for the abs form or
`1e-4` for the squared variant, and tune it with the task loss scale.
