# LSQ: Learned Step Size Quantization

## Problem

Train networks that run inference in low precision (2-, 3-, 4-bit weights and activations) while keeping full-precision accuracy. At such few levels, accuracy hinges on the per-layer **step size** s that places the quantization levels. LSQ makes each weight-layer and activation-layer step size a learnable parameter, trained jointly with the weights against the task loss.

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

The interior branch equals −frac(v/s) (signed distance to the nearest level), so the gradient to s is largest when v is near a quantization transition — exactly where a small change in s flips the integer code — and zero when v sits on a level. The data gradient is

```
∂v̂/∂v = 1 if −Q_N < v/s < Q_P, else 0.
```

## Step-size gradient scale

Good convergence wants (update magnitude)/(parameter magnitude) balanced across layers. Estimating the imbalance ratio R = (∇_s L / s) / (‖∇_w L‖ / ‖w‖): with ‖w‖/s ≈ √(N_W Q_P), and ∇_s L of the same order as ‖∇_w L‖ (both ≈ √(N_W·E[(∂L/∂ŵ)²]) since ∂ŵ/∂s ≈ 1 and the per-weight loss gradients are treated as uncorrelated zero-mean), one gets R ≈ √(N_W Q_P). To cancel it, multiply the step-size gradient by

```
g = 1/√(N_W Q_P)   (weights),     g = 1/√(N_F Q_P)   (activations),
```

where N_W is the number of weights and N_F the number of features in the layer.

## Initialization and training

Per-layer step size initialized to s = 2⟨|v|⟩ / √Q_P (from initial weights or first activation batch). Keep fp32 master weights; quantize in forward/backward with STE. First and last layers stay 8-bit; quantized nets are initialized from a trained full-precision model and fine-tuned; momentum SGD, cross-entropy, cosine LR decay.

## Code

```python
import torch, torch.nn as nn

def detach(x):
    return x.detach()                     # identity forward; blocks gradient backward

def gradscale(x, g):
    return detach(x - g * x) + g * x      # forward: x; backward: gradient to x scaled by g

def roundpass(x):
    return detach(torch.round(x) - x) + x # forward: round; backward: straight-through (grad=1)

def quantize(v, s, Qn, Qp, n_elems):
    g = 1.0 / (n_elems * Qp) ** 0.5       # 1/sqrt(N*Qp) to balance step size vs weights
    s = gradscale(s, g)
    v = torch.clamp(v / s, Qn, Qp)
    v_bar = roundpass(v)
    return v_bar * s

class QuantLayer(nn.Module):
    def __init__(self, bits, is_activation):
        super().__init__()
        self.bits, self.is_act = bits, is_activation
        self.s = nn.Parameter(torch.tensor(1.0))
        self.inited = False
    def init_step(self, v):
        Qp = (2 ** self.bits - 1) if self.is_act else (2 ** (self.bits - 1) - 1)
        self.s.data = 2 * v.abs().mean() / (Qp ** 0.5)
        self.inited = True
    def forward(self, v):
        if not self.inited:
            self.init_step(v)
        if self.is_act:
            Qn, Qp, n = 0, 2 ** self.bits - 1, v[0].numel()
        else:
            Qn, Qp, n = -2 ** (self.bits - 1), 2 ** (self.bits - 1) - 1, v.numel()
        return quantize(v, self.s, Qn, Qp, n)
```
