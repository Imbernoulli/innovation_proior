# Context

## Research question

To run deep networks cheaply at inference, we want convolution and fully-connected layers to operate on low-precision integers — 2, 3, or 4 bits for both weights and activations — so the heavy matrix multiplies become low-precision integer operations. The quantizer maps a continuous value v to a small set of integer levels by dividing by a **step size** s, clipping, and rounding. At 2–4 bits there are only a handful of levels, so the *placement* of those levels — equivalently, the value of s for each layer — strongly determines how much accuracy is lost. The question: how do we set the step size for every weight and activation layer so that a network quantized this aggressively retains its full-precision accuracy?

## Background

A standard uniform quantizer with step size s, and Q_N / Q_P negative/positive level counts, maps data v to

```
v̄ = round( clip(v/s, −Q_N, Q_P) ),   v̂ = v̄ × s,
```

where v̄ is the integer code and v̂ is the dequantized value at the same scale as v. For an encoding with b bits: unsigned (activations) Q_N = 0, Q_P = 2^b − 1; signed (weights) Q_N = 2^{b−1}, Q_P = 2^{b−1} − 1. At inference, integer weight and activation codes feed low-precision integer matmul units, and the output is rescaled by the relevant step size factors afterward; that scalar rescale can be folded into batch-norm-style operations.

Load-bearing prior concepts:
- **Straight-through estimator (Bengio et al., 2013).** Rounding/quantization is treated as identity on the backward pass so gradients can propagate to the quantized quantity. This is what makes any quantization-aware training possible.
- **Full-precision shadow weights (Courbariaux et al., 2015, BinaryConnect).** Keep fp32 weights, quantize them in the forward/backward pass, and apply accumulated gradients to the fp32 copy.
- **Fine-tuning from a pretrained fp model** (Sung et al.; Zhou et al., DoReFa; Mishra et al., Apprentice; McKinstry et al.) improves quantized accuracy over training from scratch.
- **Update/parameter-magnitude balance (You et al., 2017, LARS).** Good convergence occurs when, across layers, the ratio of average update magnitude to average parameter magnitude is roughly equal. If a parameter's updates are disproportionately large or small relative to its magnitude, training overshoots or stalls.

## Baselines

- **PACT (Choi et al., 2018).** Learns a clipping/domain-width parameter. Its forward approximation removes the round and algebraically cancels terms.
- **QIL (Jung et al., 2018).** Learns a transformation of the data applied *prior* to discretization.
- **DoReFa (Zhou et al., 2016).** Fixed (non-learned) quantization with STE, weights/activations to low bits.
- **Fixed-step quantization-aware training** generally: STE through a quantizer with a hand-set or statistically-derived step size.

## Evaluation settings

- **Dataset.** ImageNet. Images resized to 256×256, random 224×224 crop with horizontal mirroring for training; 224×224 center crop at test.
- **Architectures.** Pre-activation ResNet (e.g. ResNet-18/34/50), VGG-16 with batch-norm, SqueezeNext.
- **Bit-widths.** 2, 3, 4, or 8 bits for all matrix-multiplication layer weights and input activations, except the first and last layers, which stay 8-bit; other model parameters remain fp32.
- **Protocol.** Quantized nets initialized from a trained full-precision model, then fine-tuned in the quantized space. Momentum 0.9, softmax cross-entropy, cosine learning-rate decay without restarts; 8-bit nets trained 1 epoch, others 90 epochs; initial LR 0.1 (fp), 0.01 (2/3/4-bit), 0.001 (8-bit). Metric: top-1/top-5 accuracy vs. full precision across bit-widths and model sizes.

## Code framework

The primitives that already exist: an autodiff framework, conv/linear layers, SGD with momentum, the uniform quantizer above (clip + round + rescale), a straight-through gradient for the round, and a custom-gradient mechanism (a `detach`-style op that is identity forward and blocks gradient backward). What does *not* yet exist is how the per-layer step size is parameterized, initialized, and given a gradient — that is the empty slot.

```python
import torch, torch.nn as nn

def detach(x):
    # identity in forward; gradient blocked in backward (framework primitive)
    return x.detach()

def roundpass(x):
    # round in forward, straight-through (gradient = 1) in backward
    return detach(torch.round(x) - x) + x

def quantize(v, s, bits, is_activation):
    if is_activation:
        qmin, qmax = 0, 2 ** bits - 1
    else:
        qmin, qmax = -2 ** (bits - 1), 2 ** (bits - 1) - 1
    # TODO: how is the step size s scaled / given a gradient so it can be learned?
    v = v / s
    v = torch.clamp(v, qmin, qmax)
    v_bar = roundpass(v)
    v_hat = v_bar * s
    return v_hat

class QuantLayer(nn.Module):
    def __init__(self, bits, is_activation):
        super().__init__()
        # TODO: per-layer step size as a learnable parameter; how to initialize it?
        pass
    def forward(self, v):
        # TODO: quantize v with a learnable, initialized, gradient-scaled step size
        pass
```
