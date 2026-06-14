# Context: running a trained neural network on integer-only hardware (circa 2011-2017)

## Research question

A neural network has been trained in 32-bit floating point and now has to run cheaply at
inference time on hardware where floating-point arithmetic is the bottleneck: mobile and edge
CPUs, DSPs, and microcontrollers, where integer multiply-accumulate units are far faster, far
smaller in silicon area, and far lower in power than floating-point units, and where reading
32-bit weights from memory costs 4× the bandwidth and cache footprint of 8-bit ones. The goal
is to represent the weights (and, where possible, the activations) of an *already-trained*
network as low-bit integers, so that an integer-only implementation of each layer reproduces
the real-valued layer outputs closely enough to keep end-to-end accuracy, while the conversion
itself is cheap — ideally requiring no retraining, no labelled data, and no per-layer
optimization. A solution has to specify three things at once and have them agree: (1) the
*correspondence* between an integer code and the real number it stands for, (2) the *rule* that
picks an integer code for each real weight, and (3) the *arithmetic* that lets a layer be
evaluated on the integer codes alone. The pain point is that at low bit-widths the integer grid
is coarse — only `2^B` distinct levels per tensor — so the rule in (2) has very little room,
and a careless choice of correspondence or rounding turns small per-weight errors into a
systematic shift of the layer output that compounds across depth.

## Background

By the early 2010s deep networks were winning on vision and speech (Krizhevsky et al. 2012;
the Vanhoucke et al. 2011 speech system), but the trained models were heavy: tens to hundreds
of megabytes of FP32 weights, and matrix multiplications dominated by floating-point MACs. Two
hardware facts framed the problem. First, on commodity CPUs a well-pipelined integer
multiply-add is no more expensive than an addition, and SIMD integer instructions (SSSE3/SSE4
on x86, NEON on ARM) can pack many 8-bit operations per instruction, so 8-bit integer GEMM can
beat optimized floating-point BLAS by a multiplicative factor. Second, weights in 8 bits
instead of 32 cut the model's memory footprint, bandwidth, and cache pressure by 3-4×, which on
a memory-bound mobile workload is itself most of the speedup.

The load-bearing concept is **fixed-point / linear quantization**: pick a real interval, lay a
uniform grid of `2^B` points across it, and represent each real value by the index of the
nearest grid point. The fineness of the grid is one number, the *step size* (or *scale*): the
real width covered divided by the number of intervals. Converting a real `r` to its code is a
division by the scale followed by a rounding to an integer; converting back multiplies by the
scale. Because the map is uniform, it is *affine* — an integer code and the real value it
denotes are related by a multiply and an add — and that affineness is what later lets integer
arithmetic stand in for real arithmetic at all.

Several diagnostic facts about quantizing trained nets were already established. Quantization
error is governed by the step size: rounding to the nearest grid point leaves a residual at
most half a step, so halving the step (one more bit) halves the worst-case error. The *rounding
rule* matters: deterministic truncation (always round down) leaves a residual with a non-zero
mean, i.e. a systematic bias, whereas round-to-nearest keeps the residual small and, when the
values are spread over many grid cells, centered near zero. Stochastic rounding — round up with
probability equal to the distance to the lower grid point — is provably *unbiased*,
`E[Round(x)] = x` (Gupta et al. 2015), and that property is what makes it necessary when a value
is rounded *over and over*, as a weight is during training: a biased rounding of many tiny
updates accumulates into a drift that stalls learning. Two failure modes of one-shot
quantization of a trained net were also on record: per-output-channel weight ranges in the same
layer can differ by more than 100×, so a single shared step size per layer gives the
small-range channels enormous relative error; and a few outlier weights can stretch a tensor's
range so that all the ordinary weights lose precision once the grid is fit to the outliers.
Finally, the domain fact that for convolution and pooling with padding it is very useful for the
real value 0 to be *exactly* representable by some integer code — otherwise the padded border
contributes a small but *biased* error to every output.

## Baselines

These are the prior approaches a new low-bit scheme would be measured against and would react
to.

**8-bit linear quantization for CPU inference (Vanhoucke, Senior & Mao 2011).** The first
demonstration that a trained deep net runs faster in 8-bit integers than in floating point on a
commodity CPU. Activations are quantized to unsigned 8-bit, intermediate-layer weights to
signed 8-bit, and biases kept as 32-bit integers; the input layer stays floating point. The
weight quantizer is the simplest possible: *scale each layer's weights by their maximum
magnitude and normalize them to the `[-128, 127]` range* — i.e. step size = (max absolute
weight in the layer) / 127, a symmetric grid centered at zero with no offset. Each layer's
integer matrix multiply accumulates into a 32-bit integer, which a fast approximate sigmoid maps
back to an 8-bit activation. The justification is empirical and architecture-specific: the
sigmoid keeps the weight dynamic range bounded, so signed 8-bit suffices, and quantization
errors propagate sub-linearly. **Gap:** the scheme is one global recipe tied to a particular
network and nonlinearity, with one step size per whole layer; it offers no general
code-to-real correspondence with an adjustable offset (so it cannot cleanly handle one-sided,
non-zero-centered activations such as ReLU outputs, nor guarantee an exact zero code), and it
neither analyzes nor addresses how accuracy falls as the bit-width drops below 8 or as one
layer's channels span very different ranges.

**Stochastic-rounding fixed-point training (Gupta et al. 2015).** Targets a different problem —
training in low precision — but supplies the rounding theory. In their fixed-point format
`⟨IL, FL⟩` the step size is `ε = 2^{-FL}` and the representable range is
`[-2^{IL-1}, 2^{IL-1} - 2^{-FL}]`. Round-to-nearest picks the closer fixed-point level, so its
error magnitude is at most `ε/2`; stochastic rounding rounds to the upper adjacent level with
probability proportional to the distance from the lower level, and is unbiased
(`E[Round(x)] = x`), while round-to-nearest is not unbiased for every individual value. Their
finding is that 16-bit fixed-point training works *only* with stochastic rounding, because
round-to-nearest of the many tiny gradient updates accumulates a bias that prevents convergence.
**Gap (and the boundary that matters here):** this anti-bias benefit is intrinsically about
rounding the *same quantity repeatedly*. It leaves open the separate one-shot problem: when a
fixed trained weight is rounded once and then held fixed, the objective may be nearest-value
error rather than repeated-update unbiasedness, and stochastic rounding pays for unbiasedness
with extra variance.

**Binary / ternary / bit-shift weight networks (BNN, XNOR-Net, TWN, and related).** Push the
weights to 1 or 2 bits, or to powers of two so multiplications become bit-shifts. Very small
models and, in custom hardware, very fast. **Gap:** on existing CPUs with pipelined
multiply-add, bit-shifts buy little over ordinary multiplies; 1-bit weights cause large
accuracy degradation; and these papers rarely report verified on-device timings, so the
promised speedups are unconfirmed on real hardware.

## Evaluation settings

The natural yardsticks already in use for quantized inference:

- **Tasks and models.** Image classification on ImageNet (ResNet, Inception, and the
  efficiency-oriented MobileNet family) and object detection on COCO, plus speech models;
  perplexity-style language-model evaluation for sequence models. The interesting test is an
  *already efficient* architecture (MobileNets), where there is little redundancy to hide
  quantization error, rather than an over-parameterized one (AlexNet/VGG) where compression is
  trivially easy.
- **Quantization configuration.** Bit-widths from 8 down to the low single digits (e.g. 8, 7,
  6, 5, 4, and below); weights and/or activations quantized; biases kept at high precision.
  Implementations expose a granularity knob for how much of a tensor shares one step size,
  because range variation inside a layer is already known to matter.
- **Protocol and metrics.** Quantize a trained FP model, run integer (or simulated-integer)
  inference, and compare task accuracy / perplexity against the FP baseline; the gap to FP is
  the headline number. On the systems side, measured on-device latency and memory on real CPUs
  (ARM big/LITTLE cores, x86), since the whole point is hardware-verified speed, not just a bit
  count.

## Code framework

The substrate is a standard PyTorch training / evaluation harness for a pretrained transformer
language model. What already exists: the model loaded in FP32, an optimizer, a training loop
that can run zero or more steps, and an evaluation that, after conversion, measures perplexity.
The conversion itself is exposed
through a small fixed interface that the harness calls: a function that maps a weight tensor to
its quantized-then-dequantized form for use inside the forward pass, a separate no-gradient
version applied once after any training to materialize the integer model for evaluation, a
module that wraps a linear layer, and a routine that walks the model swapping plain linear
layers for the wrapper with a place for output-projection handling. How a weight should actually
be mapped onto a low-bit integer grid — the correspondence, the rule that picks each code, and
the shape bookkeeping — is the empty slot.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Training hyperparameters. The harness reads these into its optimizer loop;
# the loop may run zero steps.
CONFIG_OVERRIDES = {
    # TODO: choose whether and how the optimizer loop should run.
}


def fake_quantize_weight(weight, num_bits, group_size):
    """Map a weight tensor onto a low-bit grid for use in the forward pass.
    Must let a gradient reach `weight` if the loop runs training steps."""
    # TODO: the weight->grid mapping we will design.
    pass


def fake_quantize_activation(x, num_bits):
    # TODO: decide whether and how activations are mapped.
    pass


def quantize_dequantize_weight(weight, num_bits, group_size):
    """No-gradient version of the same mapping, applied once after training
    to produce the weights that are actually evaluated."""
    with torch.no_grad():
        # TODO: the same weight->grid mapping, no gradient.
        pass


class QATWrapper(nn.Module):
    """Wraps an nn.Linear. forward() runs the (possibly quantized) linear."""

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
        # TODO: use the weight mapping (or call the plain linear).
        pass


def prepare_qat_model(model, num_bits, group_size):
    """Replace every nn.Linear with QATWrapper; handle the output projection."""
    def _replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name,
                        QATWrapper(child, num_bits=num_bits, group_size=group_size))
            else:
                _replace(child)
    _replace(model)
    # TODO: handle any output projection special case.
    return model
```

The harness supplies the trained model and the bit-width; the empty functions are where the
integer-grid mapping will live.
