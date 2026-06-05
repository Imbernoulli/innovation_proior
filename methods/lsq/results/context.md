# Context

## Research question

To run deep networks cheaply at inference, we want convolution and fully-connected layers to operate on low-precision integers — 2, 3, or 4 bits for both weights and activations — so the heavy matrix multiplies become low-precision integer operations. The catch is the quantizer: it maps a continuous value v to a small set of integer levels by dividing by a **step size** s, clipping, and rounding. At 2–4 bits there are only a handful of levels, so the *placement* of those levels — equivalently, the value of s for each layer — strongly determines how much accuracy is lost. The question: how do we set the step size for every weight and activation layer so that a network quantized this aggressively retains its full-precision accuracy?

The deeper difficulty is that rounding has zero gradient almost everywhere, so the loss cannot directly "see" the step size through ordinary backprop. We need a way to make each layer's step size a *trainable parameter*, learned jointly with the weights against the task loss, with a sensible gradient flowing back through the quantizer.

## Background

A standard uniform affine quantizer with step size s, and Q_N / Q_P negative/positive level counts, maps data v to

```
v̄ = round( clip(v/s, −Q_N, Q_P) ),   v̂ = v̄ × s,
```

where v̄ is the integer code and v̂ is the dequantized value at the same scale as v. For an encoding with b bits: unsigned (activations) Q_N = 0, Q_P = 2^b − 1; signed (weights) Q_N = 2^{b−1}, Q_P = 2^{b−1} − 1. At inference, the integer codes feed low-precision integer matmul units, and the layer output is rescaled by s afterward (and can be folded into batch-norm).

Load-bearing prior concepts:
- **Straight-through estimator (Bengio et al., 2013).** Rounding/quantization is treated as identity on the backward pass so gradients can propagate to the quantized quantity. This is what makes any quantization-aware training possible.
- **Full-precision shadow weights (Courbariaux et al., 2015, BinaryConnect).** Keep fp32 weights, quantize them in the forward/backward pass, and apply accumulated gradients to the fp32 copy.
- **Fine-tuning from a pretrained fp model** (Sung et al.; Zhou et al., DoReFa; Mishra et al., Apprentice; McKinstry et al.) improves quantized accuracy over training from scratch.
- **Update/parameter-magnitude balance (You et al., 2017, LARS).** Good convergence occurs when, across layers, the ratio of average update magnitude to average parameter magnitude is roughly equal. If a parameter's updates are disproportionately large or small relative to its magnitude, training overshoots or stalls. This becomes the key diagnostic for setting a per-step-size gradient scale.

A diagnostic observation about prior step-size/clip learning methods, which sets up the problem: PACT learns a clipping value but, by removing the round from the forward equation and cancelling terms, ends up with **zero gradient to its clip parameter inside the quantized domain** — the gradient ignores how close a value sits to a quantization transition. QIL learns a transformation that happens entirely *before* discretization, so its gradient is sensitive only to distance from the clip points, not to the transitions between interior levels. In both, the relative proximity of v to the nearest transition point does *not* affect the gradient to the quantization parameters — which is intuitively the wrong behavior, since values near a transition are exactly the ones most likely to flip bins under a small change in s.

## Baselines

- **PACT (Choi et al., 2018).** Learns an activation clipping level α. Its forward equation drops the round and algebraically cancels terms, giving ∂v̂/∂(param) = 0 for v strictly inside the quantized range. Gap: no gradient signal from interior transitions; clip-only sensitivity.
- **QIL (Jung et al., 2018).** Learns an interval/transformation of the data applied *prior* to discretization, with a tunable nonlinearity. Gap: its gradient depends only on distance to the clip points, not on proximity to interior quantization transitions.
- **DoReFa (Zhou et al., 2016).** Fixed (non-learned) quantization with STE, weights/activations to low bits. Gap: no learned step size; level placement is not optimized against the loss.
- **Fixed-step quantization-aware training** generally: STE through a quantizer with a hand-set or statistically-derived step size. Gap: the step size is not adapted by the task loss, so level placement is suboptimal especially at very low bit-widths.

## Evaluation settings

- **Dataset.** ImageNet (ILSVRC-2012), ~1.2M training images, 1000 classes. Images resized to 256×256, random 224×224 crop with horizontal mirroring for training; 224×224 center crop at test.
- **Architectures.** Pre-activation ResNet (e.g. ResNet-18/34/50), VGG-16 with batch-norm, SqueezeNext.
- **Bit-widths.** 2, 3, 4, or 8 bits for all matmul layers, with the first and last layers kept at 8-bit (standard practice); all other tensors fp32.
- **Protocol.** Quantized nets initialized from a trained full-precision model, then fine-tuned in the quantized space. Momentum 0.9, softmax cross-entropy, cosine learning-rate decay without restarts; 8-bit nets trained 1 epoch, others 90 epochs; initial LR 0.1 (fp), 0.01 (2/3/4-bit), 0.001 (8-bit). Metric: top-1/top-5 accuracy vs. full precision, across bit-widths and model sizes; secondary axis is accuracy-vs-model-size and quantization error.

## Code framework

The primitives that already exist: an autodiff framework, conv/linear layers, SGD with momentum, the uniform affine quantizer above (clip + round + rescale), a straight-through gradient for the round, and a custom-gradient mechanism (a `detach`-style op that is identity forward and blocks/reshapes gradient backward). What does *not* yet exist is how the per-layer step size is parameterized, initialized, and given a gradient — that is the empty slot.

```python
import torch, torch.nn as nn

def detach(x):
    # identity in forward; gradient blocked in backward (framework primitive)
    return x.detach()

def roundpass(x):
    # round in forward, straight-through (gradient = 1) in backward
    return detach(torch.round(x) - x) + x

def quantize(v, s, Qn, Qp):
    # TODO: how is the step size s scaled / given a gradient so it can be LEARNED?
    v = v / s
    v = torch.clamp(v, Qn, Qp)
    v_bar = roundpass(v)
    v_hat = v_bar * s
    return v_hat

class QuantLayer(nn.Module):
    def __init__(self, bits, is_activation):
        super().__init__()
        # TODO: per-layer step size as a learnable parameter; how to initialize it?
        pass
    def forward(self, v):
        # TODO: compute Qn, Qp from bits; quantize v with a learnable step size
        pass
```
