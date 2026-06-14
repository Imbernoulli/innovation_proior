# Round-to-nearest (RTN) post-training quantization, distilled

RTN post-training quantization (PTQ) converts trained floating-point weights to a low-bit
quantize-dequantize form in a single shot, with no fine-tuning, no calibration data, and no
per-layer optimization. Each weight is mapped onto a uniform integer grid by an affine correspondence
`r = S(q - Z)` and rounded to the nearest grid point. For weights it specializes to the
*symmetric, per-group* form: zero-point `Z = 0`, step size `S = max|w| / qmax` computed
separately for each contiguous group of weights, and round-to-nearest with clamping to the
signed integer range. It is the cheapest possible no-training quantization reference.

## Problem it solves

Run an already-trained network on integer-only / memory-bound hardware (mobile and edge CPUs,
DSPs), where integer arithmetic is fast and 32-bit weight bandwidth is the bottleneck. Map
weights (and optionally activations) to low-bit integers so that an integer-only layer
reproduces the real-valued layer outputs closely, with a conversion that needs no retraining
and no data.

## Key idea

- **Affine correspondence (forced).** For quantized arithmetic to approximate real arithmetic,
  code `q` and real value `r` must be affinely related. Write it `r = S(q - Z)`, `S > 0` real
  (scale / step size), `Z` integer (zero-point, the code for real 0). This form makes "real 0 is
  exactly representable" automatic — needed so zero-padding in conv/pool layers does not inject a
  *biased* error.
- **Round-to-nearest (forced for one-shot).** Picking a code is
  `q = clamp(round(r/S + Z), qmin, qmax)`, and dequantizing gives `r_hat = S(q - Z)`. RTN
  minimizes the per-element error `|ŵ - w| ≤ S/2` and, when values spread over many grid cells,
  has mean-zero residual (variance `S²/12`). Stochastic rounding is unbiased for every single
  value but pays extra variance (up to `S²/4`); its anti-bias benefit only matters under
  *repeated* rounding (training). A one-shot conversion rounds each weight once, so RTN is the
  right rule.
- **Symmetric weights (`Z = 0`).** Trained weights are ~zero-centered, so a symmetric grid is
  natural and `Z = 0` falls out, collapsing the map to `r = S·q`. This also kills the weight
  zero-point cross-terms in the integer matmul, giving the leanest kernel.
- **No-clip max scale.** With no calibration data, set `S = max|w| / qmax`, where
  `qmin = -2^(B-1)` and `qmax = 2^(B-1)-1`. The positive endpoint `qmax` and the symmetric
  usable negative code `-qmax` cover `±max|w|`; the extra two's-complement code `qmin` is only a
  clamp rail. A smaller `S` would trade rounding error for clipping error, a tradeoff that
  requires data to tune.
- **Per-group scaling.** Use one `S` per contiguous group (e.g. `group_size = 128`) instead of
  per tensor, so a single outlier weight, or a channel with a much wider range, only coarsens
  its own group rather than the whole matrix — the two known one-shot failure modes.
- **Why integers pay off.** `Σ_j r1·r2 = S1·S2·Σ_j (q1 - Z1)(q2 - Z2)`; the heavy `O(N³)`
  accumulation runs purely on integers, and the remaining multiplier `M = S1·S2/S3` is folded
  offline into a fixed-point multiplier and shift.

## Final algorithm (per weight matrix, B bits)

```
qmin, qmax = -2^{B-1}, 2^{B-1} - 1                  # signed symmetric range
for each group g of `group_size` weights:
    w_max = max(|w_g|)            (floored at 1e-12) # no-clip cover, div-by-0 guard
    S_g   = w_max / qmax                             # step size, Z = 0
    q_g   = clamp(round(w_g / S_g), qmin, qmax)      # round-to-nearest, clamp
    ŵ_g   = S_g * q_g                                # dequantize (or store q_g, S_g)
```

Run-time: zero training steps (`num_steps = 0`, `lr = 0`); apply the no-grad QDQ once to every
linear weight, then evaluate. Weights only — activations stay full precision. The output
projection (LM head) and embeddings/LayerNorm stay full precision: a bias-like or
accuracy-critical, cheap-to-keep tensor.

## Where it sits

This is the zero-training PTQ reference: no data, no calibration search, no gradient update. At
low bit-widths (4 -> 3 -> 2 bits, i.e. 16 -> 8 -> 4 codes) the rounding floor `S/2` grows until
at 2 bits it can reach `max|w|/2`, and nothing in a one-shot scheme repairs it. Letting weights
move to compensate requires data, gradients, and optimization; RTN PTQ keeps the conversion fixed
to the nearest-grid rule.

## Working code

Per-group symmetric RTN quantize-dequantize (differentiable straight-through forward + no-grad
eval), the linear wrapper, and the model-preparation swap that leaves the head full precision,
with training switched off.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG_OVERRIDES = {
    "learning_rate": 0.0,            # no fine-tune
    "num_steps": 0,                  # training loop is a no-op
    "batch_size": 2,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "warmup_steps": 0,
    "weight_decay": 0.0,
}


def _qrange(num_bits):
    qmax = (1 << (num_bits - 1)) - 1     # +2^{B-1} - 1
    qmin = -(1 << (num_bits - 1))        # -2^{B-1}
    return qmin, qmax


def fake_quantize_weight(weight, num_bits, group_size):
    """Differentiable per-group symmetric RTN quantize-dequantize (forward path)."""
    qmin, qmax = _qrange(num_bits)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    w = weight.float().reshape(out_features, -1, group_size)         # group the columns
    w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)      # per-group max|w|
    scale = w_max / qmax                                             # S = max|w| / qmax
    w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale    # RTN + clamp + dequant
    w_dq = w + (w_q - w).detach()                                    # straight-through gradient
    return w_dq.reshape(out_features, in_features).to(weight.dtype)


def fake_quantize_activation(x, num_bits):
    return x   # weight-only quantization


def quantize_dequantize_weight(weight, num_bits, group_size):
    """No-grad version, applied once after (zero-step) training for evaluation."""
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
        self.linear = linear
        self.num_bits = num_bits
        self.group_size = group_size

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        # PTQ-only: by eval time the real QDQ has already replaced linear.weight,
        # so run the plain linear on the already-quantized weight.
        return F.linear(x, self.linear.weight, self.linear.bias)


def prepare_qat_model(model, num_bits, group_size):
    def _replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name,
                        QATWrapper(child, num_bits=num_bits, group_size=group_size))
            else:
                _replace(child)

    _replace(model)
    for head_attr in ("lm_head", "embed_out"):     # keep the output projection full precision
        head = getattr(model, head_attr, None)
        if isinstance(head, QATWrapper):
            setattr(model, head_attr, head.linear)
    return model
```
