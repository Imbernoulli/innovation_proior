# Learned Step Size Quantization

## Problem

Train networks that run inference in low precision (2-, 3-, 4-bit weights and activations) while keeping full-precision accuracy. At so few levels, accuracy hinges on the per-layer **step size** s that places the quantization levels. LSQ makes each weight-layer and activation-layer step size a learnable parameter, trained jointly with the weights against the task loss.

## Quantizer

```
v̄ = round( clip(v/s, −Q_N, Q_P) ),   v̂ = v̄ · s.
```
Unsigned activations: Q_N = 0, Q_P = 2^b − 1.  Signed weights: Q_N = 2^{b−1}, Q_P = 2^{b−1} − 1.

## Step-size gradient

Using the straight-through estimator for the round (treat it as identity on the backward pass) and differentiating the divide/clip/multiply normally:

```
            ⎧ −v/s + round(v/s),   if −Q_N < v/s < Q_P
∂v̂/∂s  =    ⎨ −Q_N,                if v/s ≤ −Q_N
            ⎩  Q_P,                if v/s ≥ Q_P.
```

The interior branch equals round(v/s) − v/s, the negative signed residual between v/s and the integer level it rounds to. Its magnitude is largest near quantization transitions and zero when v/s sits exactly on a level. The data gradient is

```
∂v̂/∂v = 1 if −Q_N < v/s < Q_P, else 0.
```

## Step-size gradient scale

Good convergence wants (update magnitude)/(parameter magnitude) balanced across layers. Estimating the imbalance ratio R = (∇_s L / s) / (‖∇_w L‖ / ‖w‖): with ‖w‖/s ≈ √(N_W Q_P), and ∇_s L scaling like ‖∇_w L‖ under the heuristic that the per-weight loss gradients are uncorrelated zero-mean and ∂ŵ/∂s contributes a per-element constant factor, one gets R ≈ √(N_W Q_P). To cancel it, multiply the step-size gradient by

```
g = 1/√(N_W Q_P)   (weights),     g = 1/√(N_F Q_P)   (activations),
```

where N_W is the number of weights and N_F the number of features in the layer.

## Initialization and training

Per-layer step size initialized to s = 2⟨|v|⟩ / √Q_P (from initial weights or first activation batch). Keep fp32 stored weights; quantized weights and activations are used in forward/backward passes with STE. First and last matrix-multiplication layers stay 8-bit; quantized nets are initialized from a trained full-precision model and fine-tuned with momentum SGD, cross-entropy, and cosine LR decay.

## Code

```python
import torch
import torch.nn as nn

def detach(x):
    return x.detach()                     # identity forward; blocks gradient backward

def gradscale(x, g):
    return detach(x - g * x) + g * x      # forward: x; backward: gradient to x scaled by g

def roundpass(x):
    return detach(torch.round(x) - x) + x # forward: round; backward: straight-through (grad=1)

def nfeatures(v):
    return v[0].numel()

def nweights(v):
    return v.numel()

def qparams(bits, is_activation):
    if is_activation:
        return 0, 2 ** bits - 1, nfeatures
    return -2 ** (bits - 1), 2 ** (bits - 1) - 1, nweights

def quantize(v, s, bits, is_activation):
    qmin, qmax, count_fn = qparams(bits, is_activation)
    g = 1.0 / (count_fn(v) * qmax) ** 0.5 # 1/sqrt(N_features*Qp) or 1/sqrt(N_weights*Qp)
    s = gradscale(s, g)
    v_scaled = torch.clamp(v / s, qmin, qmax)
    v_bar = roundpass(v_scaled)
    return v_bar * s

class QuantLayer(nn.Module):
    def __init__(self, bits, is_activation):
        super().__init__()
        self.bits, self.is_activation = bits, is_activation
        self.s = nn.Parameter(torch.tensor(1.0))
        self.inited = False

    def init_step(self, v):
        _, qmax, _ = qparams(self.bits, self.is_activation)
        init = 2 * v.detach().abs().mean() / (qmax ** 0.5)
        with torch.no_grad():
            self.s.copy_(init)
        self.inited = True

    def forward(self, v):
        if not self.inited:
            self.init_step(v)
        return quantize(v, self.s, self.bits, self.is_activation)
```
