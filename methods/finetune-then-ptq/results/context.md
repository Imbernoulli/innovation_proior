# Context: low-bit weight quantization of a trained transformer (circa the low-bit-LLM era)

## Research question

I have a transformer language model that was trained in 32-bit floating point, and I need to run
it with its linear-layer weights stored at very low bit-width — 4, 3, or even 2 bits per weight.
The point of going low-bit is memory: at 2-4 bits a weight matrix is 8-16x smaller than fp32, which
is what lets a multi-billion-parameter model fit and stream on memory-bound hardware. The hard
constraint is accuracy: after the weights are mapped onto a coarse integer grid, the model's
language-modeling loss must stay close to the full-precision model's. Concretely the model is scored
by perplexity (exponentiated mean next-token cross-entropy) on a held-out text split, and I have a
small amount of in-domain text available that I am allowed to compute on before fixing the final
low-bit model.

The precise problem: produce a set of low-bit integer weights whose *post-quantization* perplexity
is as low as possible. The difficulty is that the irreversible step — snapping each real weight to
a signed `B`-bit integer grid — is brutal at small `B`. At `B = 2`, `qmax = 1`, so the
max-magnitude scale makes the grid spacing equal to the largest absolute weight in the group; the
round-to-nearest error on a weight near the middle of that spacing can be as large as half that
largest weight. A method that simply rounds the pretrained weights has an error floor set by the grid's
coarseness and no way to repair what it loses. The question is what, if anything, I can do with the
weights and the in-domain text *before* that irreversible rounding to lower the final perplexity.

## Background

**Integer quantization as an affine map.** The accepted way to represent a real tensor by low-bit
integers is an affine correspondence between an integer code `q` and the real value `r` it stands
for, `r = S(q - Z)`, with `S` a positive real step size (the *scale*) and `Z` an integer
*zero-point* (the code that maps to real 0). This form is forced by two requirements that were
established before this problem: (1) for integer arithmetic on the codes to approximate real
arithmetic — so a layer can run on integer hardware — the map must be affine; (2) the real value 0
must be exactly representable (so that zero-padding in conv/pool layers injects no *biased* error),
which is automatic in the `r = S(q - Z)` parameterization since `q = Z` gives `r = 0` exactly. For a
weight tensor, which is roughly zero-centered, the natural choice is `Z = 0`, so the map collapses to
`r = S q`; a single scale per group of weights is then the only knob.

**The max-magnitude symmetric scale.** Vanhoucke, Senior & Mao (2011), accelerating neural-net
inference on x86 CPUs with 8-bit fixed point, set the scale by the largest magnitude in each weight
tensor: "weights are scaled by taking their maximum magnitude in each layer and normalizing them to
fall in the `[-128, 127]` range." This is the data-free, no-clip choice — `S = max|w| / qmax`, the
smallest step tied to the positive endpoint that covers every `|w| <= max|w|` without clipping; the
two's-complement signed range is `qmin = -2^(B-1)`, `qmax = 2^(B-1) - 1`, so the most negative code
is an extra code outside the positive-limited range when `Z = 0`. They observed that with the linear
nature of the operations and the dynamic-range compression of the nonlinearities, "quantization
errors tend to propagate sub-linearly and not cause numerical instability" — at 8 bits, on an
over-parameterized network, a one-shot conversion of the trained weights is often adequate.

**Rounding rules and where bias comes from.** When a real value `x` is snapped to the grid, the
choice of rounding rule controls the error's mean. Round-to-nearest minimizes each element's error
(`|error| <= S/2`) and, under the usual uniform-within-a-cell model, has zero-mean residual with
variance `S^2/12`. Truncation always rounds one way and so injects a systematic, mean-nonzero error.
Gupta, Agrawal, Gopalakrishnan & Narayanan (2015) introduced *stochastic rounding* for low-precision
*training*: between adjacent grid values `l` and `u = l + S`, round to `u` with probability
`(x - l) / S` and to `l` otherwise, which is unbiased in expectation (`E[round(x)] = x` exactly).
Their setting is the one that makes
unbiasedness matter: during fixed-point SGD the *same* weight is rounded over and over across
thousands of updates, and round-to-nearest's tiny per-step bias compounds into a drift that stalls
learning, while stochastic rounding lets small gradient updates survive. Crucially this benefit only
bites under *repeated* rounding of the same value; rounding once has no accumulation to fight.

**Diagnostic failure modes of one-shot quantization.** It is documented that simply rounding the
trained floating-point weights works well for large models with spare representational capacity but
drops accuracy sharply for small models. Two failure modes are named: (1) the ranges of weights can
differ by more than `100x` across different output channels of the same layer, so a single shared
scale forces the narrow-range channels onto a grid far too coarse for them, blowing up their relative
error; (2) a few outlier weight values inflate the group's `max|w|`, coarsening the scale for every
ordinary weight in the group. Both are the disease of one scale serving very different local scales;
grouping the scale (a separate `S` per contiguous block of columns) localizes the damage but does
not remove the underlying error floor at very low bit-width.

**What fine-tuning is known to recover.** Han, Mao & Dally (2015), in their compression pipeline,
showed that after quantizing weights into a small shared codebook, *retraining* the shared values —
letting gradients move the codebook entries — recovers much of the accuracy lost to quantization at
5-8 bits. The lesson that travels: weights are allowed to move to
compensate for quantization, and a short fine-tune is a cheap way to claw back lost loss. The
distinction is *what* is being fine-tuned and *with respect to what loss* — Deep Compression
fine-tunes the already-quantized centroids against the task loss, i.e. with the quantization in the
loop.

**The objective that actually matters.** A theme in the post-training-quantization literature is
that the right thing to minimize is not how close the quantized weights are to the originals
(weight-space MSE) but how close the *network's output / task loss* stays — what a layer computes,
not what its weights are. Two networks with very different weights can compute nearly the same
function; conversely, a weight set close in MSE can have noticeably worse loss. This reframing is
what makes the available in-domain text relevant before the final integer model is fixed.

## Baselines

**One-shot round-to-nearest of the pretrained weights (data-free PTQ).** Take the trained
floating-point weights as they are and apply per-group symmetric round-to-nearest: for each
contiguous group of columns, `S = max|w_group| / qmax` with `qmin = -2^(B-1)`, `qmax = 2^(B-1) - 1`,
then `w_hat = S * clamp(round(w / S), qmin, qmax)`. Core idea: the cheapest possible conversion —
no data, no training, one pass over the weights. The actual algorithm is exactly the affine map
above with `Z = 0`. **Gap:** the conversion is blind to the task; its error is frozen at the moment
of rounding, bounded by `S/2`, which at `B = 2` (`qmax = 1`) reaches `max|w_group| / 2`. On a small
or capacity-tight model the named failure modes (cross-channel range spread, outliers) bite, and
there is no mechanism to repair the rounding error after the fact.

**8-bit fixed-point inference of a trained network (Vanhoucke et al. 2011).** Quantize the trained
weights once to signed 8-bit by the max-magnitude scale and run integer GEMM. Core idea: high-bit,
data-free conversion of an offline-trained network for fast CPU inference. **Gap:** demonstrated at 8
bits on an over-parameterized network, where the grid is fine enough that one-shot conversion is
accurate; the approach has nothing to say about 2-4 bits, where the grid is coarse and the one-shot
error floor dominates, and it uses a single scale per layer rather than per group.

**Quantize-then-retrain the quantized representation (Han et al. 2015).** Cluster the weights into a
shared codebook, then fine-tune the codebook entries against the task loss with the codebook in the
forward pass. Core idea: let weights move to recover the accuracy lost to quantization, with the
quantization present during retraining. **Gap:** the retraining mixes ordinary task-loss adaptation
with adaptation to the quantized representation; it also uses a non-uniform codebook (k-means
centroids) rather than the uniform affine integer grid that integer-arithmetic inference and the
low-bit storage format require, and was shown at 5-8 bits, not 2-4.

## Evaluation settings

- **Backbone:** a pretrained decoder-only transformer language model (GPTNeoX-family, ~1.4B
  parameters, 24 layers, hidden 2048), loaded in fp32. Every `nn.Linear` inside the transformer
  blocks is a candidate for low-bit weight quantization; embeddings, LayerNorm, and the output
  projection (LM head) stay full precision.
- **Quantization format (fixed):** per-row, per-group of `group_size = 128` contiguous columns,
  symmetric, signed; bit-widths `B in {4, 3, 2}`.
- **Calibration / fine-tune data:** the WikiText-2 raw v1 train split, sampled as random 1024-token
  crops. This is the in-domain text available before the final model is fixed.
- **Metric:** perplexity on the WikiText-2 raw v1 test split — sliding non-overlapping 1024-token
  blocks, exponentiated mean next-token cross-entropy; lower is better. A secondary number is the
  degradation versus the full-precision model measured before any quantization.
- **Protocol:** all seeds and hyperparameters deterministic given a seed; the optimizer is AdamW with
  a cosine learning-rate schedule and linear warmup; gradients are clipped by norm; gradient
  checkpointing keeps the 1.4B model and optimizer in memory.

## Code framework

The harness already exists. It loads the fp32 model, wraps each transformer linear in a module that
the training loop can update, runs a fixed text-modeling optimization loop (AdamW, cosine LR with
warmup, gradient accumulation, grad-norm clipping), exposes a weight hook used by the wrapper's
forward pass, exposes a materialization hook used before the low-bit evaluation pass, and finally
measures test perplexity. What is not settled is the editable region: what the weight hook returns,
what the materialization hook computes, and how the wrapper connects those hooks to the existing
linear primitive.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Training hyperparameters; the fixed training loop reads this dict.
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
    """Symmetric signed integer range for `num_bits`-bit quantization."""
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))
    return qmin, qmax


def fake_quantize_weight(weight, num_bits, group_size):
    # TODO: define the training-time weight hook.
    pass


def fake_quantize_activation(x, num_bits):
    return x  # weight-only: activations stay full precision


def quantize_dequantize_weight(weight, num_bits, group_size):
    # TODO: define the low-bit materialization hook used by the evaluator.
    pass


class QATWrapper(nn.Module):
    """Wraps an nn.Linear so the training loop can update its weight.
    The forward pass is the slot to design."""

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
        # TODO: connect the editable weight hook to the existing linear primitive.
        pass


def prepare_qat_model(model, num_bits, group_size):
    """Replace every nn.Linear with QATWrapper in-place; keep the output
    projection (LM head) at full precision."""
    from transformers.pytorch_utils import Conv1D

    def _replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name,
                        QATWrapper(child, num_bits=num_bits, group_size=group_size))
            elif isinstance(child, Conv1D):
                in_f, out_f = child.weight.shape
                lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
                                device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    lin.weight.copy_(child.weight.t().contiguous())
                    if child.bias is not None:
                        lin.bias.copy_(child.bias)
                setattr(parent, name,
                        QATWrapper(lin, num_bits=num_bits, group_size=group_size))
            else:
                _replace(child)

    _replace(model)
    for head_attr in ("lm_head", "embed_out"):
        head = getattr(model, head_attr, None)
        if isinstance(head, QATWrapper):
            setattr(model, head_attr, head.linear)
    return model
```

The fixed driver supplies the data, optimizer, scheduler, replacement walk, and evaluation metric.
The three `# TODO` slots are the only open hooks.
