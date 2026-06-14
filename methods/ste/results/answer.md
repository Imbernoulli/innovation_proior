# Straight-Through Estimator (STE), distilled

The straight-through estimator trains networks that contain hard, non-differentiable operations
(a binary threshold, `sign`, `round`/`clamp` quantization) by **using the true hard operation in
the forward pass and treating it as the identity in the backward pass**. The hard op's honest
Jacobian is zero almost everywhere, so the chain rule kills the gradient; STE replaces that zero
Jacobian with the identity (slope 1), passing the incoming gradient `dL/dh` straight back to the
operation's input. The forward still produces the genuine discrete value, so the network is trained
for the discrete behavior it will actually use.

## Problem it solves

You want a hard discrete decision inside a differentiable network — a 0/1 gating unit for
conditional computation, or low-precision weights/activations so matmuls become cheap integer ops —
and you must still train end to end by gradient descent. The discrete op (indicator / `sign` /
`round`) has derivative 0 almost everywhere (undefined at jumps), so plain back-propagation transmits
no signal through it. STE manufactures a usable backward signal.

## Key idea

Forward = hard, backward = identity. For a binary stochastic neuron `h = 1_{z < sigm(a)}`
(`z ~ U[0,1]`, so `P(h=1) = sigm(a)`), the STE gradient with respect to the pre-activation is

```
g_a = dL/dh          (pretend dh/da = 1, i.e. back-propagate through the threshold as identity)
```

A variant multiplies by the sigmoid derivative `sigm(a)(1 - sigm(a))`; the identity form deliberately
omits that factor because it vanishes in saturation (`|a|` large) — exactly where confidently
thresholded units live — and would re-introduce vanishing gradients there. Plain identity keeps the
gradient alive in saturation.

## Why it works (and its limits)

- **Right sign for one layer.** The threshold is monotone increasing in `a`, so the true effect of
  `a` on `h` is positive; identity's slope `+1` matches it. Descent direction in expectation.
- **Descent in expectation.** For one hard unit with `p=sigm(a)` and downstream loss values `L(1)` and
  `L(0)`, the expected loss is `F(a)=pL(1)+(1-p)L(0)`. The true gradient is
  `F'(a)=p(1-p)(L(1)-L(0))`; the identity STE sends back `g_ST=L(1)-L(0)`. Thus
  `F'(a)=p(1-p)g_ST`, equivalently `g_ST=F'(a)/(p(1-p))` for finite `a`. This is a positive
  coordinate-wise rescaling, so it preserves descent direction in the one-layer setting. (A poorly
  chosen proxy need not.)
- **Biased, low-variance, local.** Unlike the unbiased likelihood-ratio (REINFORCE) estimator
  `(h_i - sigm(a_i))·L` — which is high-variance, needs the loss broadcast to every unit, and needs
  per-unit baselines — STE is biased but essentially zero added variance and uses the structured local
  back-propagation signal. The bias has the right sign for a single layer.
- **Limit:** "right sign" is a single-layer guarantee; stacked across many hard units the identity
  proxy's sign error can compound. Most trustworthy for a single hard (e.g. gating) layer.

## Two equivalent implementations

```python
import torch

# Explicit custom autograd primitive: forward = hard op, backward = identity.
class _StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        return (a > 0).float()        # the true hard decision (binary threshold / sign / round ...)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output            # identity: dL/da = dL/dh

# The detach trick (no custom class): forward value = hard(a), backward gradient = 1.
def straight_through(a, hard):
    return a + (hard(a) - a).detach()
```

## Use on weights: quantization-aware training

The canonical QAT use is forcing weights (or activations) to a small discrete set. The structure:

- keep a **full-precision shadow weight** (needed to accumulate/average many tiny noisy SGD steps);
- **quantize it on the fly** each forward pass and use the quantized value in the matmul;
- **straight-through** carries the gradient back to the shadow weight;
- **accumulate** updates in the shadow; when the grid has a fixed real range, clip the shadow, while
  max-abs per-group QAT recomputes the scale from the shadow so the bounded integer code range tracks it;
- after training, a **no-grad** quantize-dequantize materializes the deployed integer model.

For `B`-bit per-group symmetric signed quantization (codes in `[-2^{B-1}, 2^{B-1}-1]`, max-abs scale
`s = max|w|/qmax` per (row, group)):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


CONFIG_OVERRIDES = {
    "learning_rate": 2e-5,
    "num_steps": 500,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "warmup_steps": 50,
    "weight_decay": 0.0,
}


def _qrange(num_bits):
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))
    return qmin, qmax


def fake_quantize_weight(weight, num_bits, group_size):
    """Differentiable fake-quant for the QAT forward: per-(row, group) symmetric round/clamp,
    with straight-through so the gradient reaches the full-precision shadow weight."""
    qmin, qmax = _qrange(num_bits)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    w = weight.float().reshape(out_features, -1, group_size)
    w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)   # floor avoids /0
    scale = w_max / qmax
    # round + clamp are non-differentiable; the detach below makes the backward an identity
    w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
    w_dq = w + (w_q - w).detach()                                 # straight-through
    return w_dq.reshape(out_features, in_features).to(weight.dtype)


def fake_quantize_activation(x, num_bits):
    return x                                                      # weight-only QAT: identity


def quantize_dequantize_weight(weight, num_bits, group_size):
    """Real (no-grad) quantize-dequantize used after training to materialize the integer model."""
    qmin, qmax = _qrange(num_bits)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    with torch.no_grad():
        w = weight.float().reshape(out_features, -1, group_size)
        w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = w_max / qmax
        w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
        return w_q.reshape(out_features, in_features).to(weight.dtype)


class QATWrapper(nn.Module):
    def __init__(self, linear, num_bits, group_size):
        super().__init__()
        self.linear = linear                                     # full-precision shadow weight
        self.num_bits = num_bits
        self.group_size = group_size

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        x = fake_quantize_activation(x, self.num_bits)
        w_q = fake_quantize_weight(self.linear.weight, self.num_bits, self.group_size)
        return F.linear(x, w_q, self.linear.bias)


def prepare_qat_model(model, num_bits, group_size):
    from transformers.pytorch_utils import Conv1D

    def _replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name, QATWrapper(child, num_bits=num_bits, group_size=group_size))
            elif isinstance(child, Conv1D):
                in_f, out_f = child.weight.shape
                lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
                                device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    lin.weight.copy_(child.weight.t().contiguous())
                    if child.bias is not None:
                        lin.bias.copy_(child.bias)
                setattr(parent, name, QATWrapper(lin, num_bits=num_bits, group_size=group_size))
            else:
                _replace(child)

    _replace(model)
    # keep the output projection at full precision
    for head_attr in ("lm_head", "embed_out"):
        head = getattr(model, head_attr, None)
        if isinstance(head, QATWrapper):
            setattr(model, head_attr, head.linear)
    return model
```

## Relation to other estimators

- **REINFORCE / likelihood ratio (Williams 1992)** specialized to a sigmoid-Bernoulli unit gives the
  **unbiased** estimator `(h_i - sigm(a_i))·L` (variance-optimal baseline = per-unit weighted loss).
  STE is the cheap, biased, low-variance alternative that keeps backprop's locality.
- **Noisy rectifier** (`h = max(0, a+z)`, logistic `z`: `P(h>0)=sigm(a)`, `E[h]=softplus(a)`) and
  **stochastic-times-smooth** (`h = b·√p`, `b~Binomial(√p)`, `p=sigm(a)`: `E[h]=p`,
  `E[f(h)]=f(p)+o(√p)`) both back-propagate by replacing the hard threshold with a smooth surrogate;
  STE instead keeps the forward exactly hard and only edits the backward Jacobian.
- **Perturbation methods** (finite difference `O(N²)`; SPSA `O(N)` but valid only for small
  perturbations) are too costly or ill-suited to all-or-none decisions; STE needs no extra forward
  evaluations.
