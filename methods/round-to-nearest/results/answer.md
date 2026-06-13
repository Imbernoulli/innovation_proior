# Round-To-Nearest (RTN), distilled

RTN is the simplest post-training quantization (PTQ) of a trained network's weights:
map each weight to the nearest point of a uniform integer grid whose step size is set
from the weights' own magnitude, with no retraining and no calibration data. It is
the data-free, zero-overhead baseline — the direct post-training application of
uniform affine integer quantization in its symmetric (zero zero-point) form, at
per-channel or per-group granularity.

## Problem it solves

Shrink and speed up inference by storing/computing a trained layer's weight matrix in
`B`-bit signed integers (`B = 4` or `3`) instead of fp16, while perturbing the
layer's output as little as possible, under two hard constraints: no gradient updates
to the weights (no retraining/fine-tuning), and a genuine fixed-precision integer
encoding decodable by cheap hardware arithmetic.

## Key idea

Represent a real weight `w` by an integer code `q` through a uniform **affine** map
```
w ≈ S · (q - Z),     q = clamp( round(w / S) + Z,  qmin,  qmax ),
```
with `S > 0` the step (scale) and `Z` the integer zero-point. Uniform spacing makes
decode a single multiply (no lookup table). For weights:

- **Symmetric (`Z = 0`).** Weights are centered, so a nonzero `Z` only marginally
  tightens the range while injecting a `Z·Σq` cross-term into the integer matmul. Set
  `Z = 0`; encode becomes `q = round(w / S)`, decode `w_hat = S·q`.
- **Scale from the max.** The signed `B`-bit container has
  `qmin = -2^{B-1}` and `qmax = 2^{B-1}-1`; for `B=4`, `qmin=-8, qmax=7`, and for
  `B=3`, `qmin=-4, qmax=3`. Choose the restricted symmetric scale
  `S = max(|w|) / qmax`, so natural rounded codes lie in `[-qmax, qmax]` and the
  most-negative container code is left unused.
- **Round to nearest.** Among grid points, the nearest minimizes the per-weight error
  `|S·q - w|` — optimal when each weight is treated in isolation (no data, no
  cross-weight model). Use a nearest-integer primitive rather than truncation or an
  upward-biased half-tie rule; the implementation below mirrors the tensor code's
  `torch.round` followed by clamp.
- **Granularity.** A single per-tensor `S` is hostage to the loudest channel: if one
  output channel's range is 100× another's, the narrow channel collapses to a few
  codes near zero. Give each output row its own scale (**per-channel**), or each
  contiguous group of `g` input columns its own scale (**per-group**, `g = 64/128`),
  adapting `S` to local magnitude at a cost of one fp scale per group.
- **No calibration.** Weight ranges are static (fixed once training ends), so all
  parameters come from the weights alone — no forward pass, no data. The calibration
  hook is accepted but ignored.

## Why bit-width and group size matter (no measurement needed)

Model rounding as additive noise `η ~ Unif(-S/2, S/2)`, so per-weight MSE `≈ S²/12`
with `S ∝ max(|w|)/(2^{B-1}-1)`. Dropping `B = 4 → 3` changes the half-width
`7 → 3`, so `S` more than doubles and the noise power rises by `≈ (7/3)² ≈ 5.4×`,
compounded over all layers — which is why INT3 is much harder than INT4. Smaller
groups take `max(|w|)` over fewer weights, tightening `S` to local magnitude, so
group 64 ≤ group 128 ≤ per-channel in error.

## Limitation (the gap stronger PTQ closes)

RTN minimizes the **per-weight** error in isolation, a loose proxy for the **layer
output** error `(W_hat - W)·x`. It ignores (1) which input directions `x` actually
uses — information a calibration pass carries — and (2) the coherent accumulation of
a row's rounding errors into the output, which sequential error-compensation could
cancel. RTN is the right answer *given* a refusal to look at data or model weight
interactions, and a deliberately simple floor for data-aware post-training methods.

## Final algorithm

```
for each linear weight matrix W (out_features x in_features):
    qmin = -2^{B-1};  qmax = 2^{B-1} - 1
    split W by output row, or by contiguous input-column groups within each row
    for each row/group:
        w_max = max(|W_group|), floored to 1e-12
        S     = w_max / qmax          # effective grid is [-qmax, qmax]
        Z     = 0
    broadcast each row/group S and Z back to W's original shape
    W_q   = clamp(round(W / S) + Z, qmin, qmax)
    W_hat = (W_q - Z) * S
return W_hat (same shape & dtype as W; no weights are trained)
```

## Working code

Faithful to the canonical symmetric `pseudo_quantize_tensor` (AWQ) and the
per-channel min/max quantizer (GPTQ): a `quantize`/`dequantize` primitive, a
parameter finder, and a per-layer quantizer that ignores the calibration stream.
The repeated scale tensors below are a computation view; a packed implementation
stores the compact per-row or per-group scales.

```python
import torch


def quantize_tensor(x, scale, zero_point, qmin, qmax):
    """Encode a float tensor to integer codes: nearest rounding, clamped to grid."""
    x_int = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    return x_int


def dequantize_tensor(x_int, scale, zero_point):
    """Decode integer codes back to approximate floats (inverse affine map)."""
    return (x_int - zero_point) * scale


def find_scale_zero(weight, num_bits=4, group_size=-1, symmetric=True):
    """Per-channel (group_size=-1) or per-group quantization parameters from weights."""
    qmin = -(1 << (num_bits - 1))          # B=4 -> -8 ; B=3 -> -4
    qmax = (1 << (num_bits - 1)) - 1       # B=4 ->  7 ; B=3 ->  3

    if group_size > 0:
        out_features, in_features = weight.shape
        assert in_features % group_size == 0
        w_groups = weight.reshape(out_features, -1, group_size)
        if symmetric:
            w_max = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            scale = w_max / qmax                       # largest weight at grid edge
            zero_point = torch.zeros_like(scale)       # Z = 0 for centered weights
        else:
            w_min = w_groups.amin(dim=-1, keepdim=True)
            w_max = w_groups.amax(dim=-1, keepdim=True)
            w_range = (w_max - w_min).clamp(min=1e-12)
            scale = w_range / (qmax - qmin)
            zero_point = torch.round(qmin - w_min / scale)
        scale = scale.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
        zero_point = zero_point.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
    else:
        if symmetric:
            w_max = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            scale = w_max / qmax
            zero_point = torch.zeros_like(scale)
        else:
            w_min = weight.amin(dim=1, keepdim=True)
            w_max = weight.amax(dim=1, keepdim=True)
            w_range = (w_max - w_min).clamp(min=1e-12)
            scale = w_range / (qmax - qmin)
            zero_point = torch.round(qmin - w_min / scale)

    return scale, zero_point, qmin, qmax


class LayerQuantizer:
    """RTN: symmetric per-channel/per-group round-to-nearest, computed from the
    weights alone; calibration data is accepted but unused (weight ranges are static)."""

    def __init__(self, layer, num_bits=4, group_size=-1):
        self.layer = layer
        self.num_bits = num_bits
        self.group_size = group_size
        self.out_features, self.in_features = layer.weight.shape
        self.dev = layer.weight.device
        self.nsamples = 0
        # allocated only for interface parity with calibration-using methods; unused
        self.H = torch.zeros(
            (self.in_features, self.in_features), device=self.dev, dtype=torch.float32
        )

    def add_batch(self, inp):
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        self.nsamples += inp.shape[0]          # count only; weights need no data

    def quantize(self):
        W = self.layer.weight.data.clone().float()
        scale, zero_point, qmin, qmax = find_scale_zero(
            W, num_bits=self.num_bits, group_size=self.group_size, symmetric=True
        )
        W_q = quantize_tensor(W, scale, zero_point, qmin, qmax)
        W_dq = dequantize_tensor(W_q, scale, zero_point)
        return W_dq.to(self.layer.weight.dtype)

    def free(self):
        del self.H
        self.H = None
```
