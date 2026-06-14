## Research question

A large language model is, computationally, almost entirely matrix multiplications inside its
linear projections — the QKV and output projections of attention, the two MLP projections, the
LM head. Each of those stores a dense full-precision (FP16/BF16) weight matrix and, at
inference time, must stream every one of those weights from DRAM and run a floating-point
multiply-accumulate against the activations. For a multi-hundred-million-to-billion parameter
model this makes weight memory and the energy of floating-point multiplies the dominant cost,
and the DRAM→on-chip bandwidth for the weights the dominant latency bottleneck.

The precise goal is to **train, from scratch, a GPT-style language model whose linear-layer
weights live in a tiny discrete set during every forward pass** — on the order of one to two
bits per weight — rather than in floating point, while keeping the validation loss and
downstream language ability as close as possible to a full-precision model of the same size and
token budget. "During every forward pass" is the crux: the forward computation that the loss
sees must already be the low-bit computation, with no separate high-precision forward used for
training and low-precision only for deployment. A solution has to answer four coupled
questions at once: which small discrete set the weights take; how to set the per-matrix scale
that relates that discrete set back to the real weight magnitudes; how to get gradients through
a forward map that is piecewise-constant in the weights (so its true derivative is zero almost
everywhere); and how to keep activations cheap enough that the matmul is genuinely low-bit,
without letting activation quantization destroy the signal. Doing this well below the
comfortable 4-bit regime — at 2 bits or fewer — is the open problem.

## Background

**Why full precision is the baseline, and where its cost sits.** Standard networks store and
compute weights in FP32 or, in modern training, BF16/FP16. The bulk of an LLM's inference cost
is the linear projections: with hidden size `d`, each token costs O(d²) floating-point
multiply-adds per projection, and the weights themselves are O(d²) values that must be held in
memory and moved on chip. Floating-point multiplies are markedly more energy-expensive than
integer adds (the energy-per-operation literature, Horowitz-style models used by Rastegari et
al. 2016 and later quantization work, puts an FP multiply at one to two orders of magnitude
above an integer add at the same process node). So two levers exist: shrink the *bits per
weight* (memory and bandwidth) and replace *floating-point multiplies* with integer additions
(energy and speed). Both point at the same move — make the forward weights small integers.

**Post-training quantization (PTQ) and where it breaks.** The cheapest way to low-bit is to
train in float and quantize afterward. Round-to-nearest absmax, and the smarter
error-correcting variants, are well understood. The empirical fact that motivates everything
below is that PTQ degrades gracefully down to about 4-bit weights and then **falls off a
cliff**: a model trained for a full-precision weight landscape simply has no nearby 2-bit or
1-bit configuration that preserves its behavior, because the training never had any pressure to
make the weights robust to coarse rounding. Pushing PTQ to 2 bits and below produces large,
sometimes catastrophic, loss increases, which is why one would consider paying the cost of
changing the training procedure rather than quantizing a finished model.

**Quantization-aware training (QAT) and its central difficulty.** The alternative is to make
the model experience the quantization during training, so the optimizer can steer the weights
into a configuration whose quantized form is good. The difficulty is optimization, not
representation: a quantizer is a step function of its input, constant between thresholds, so its
derivative is zero almost everywhere and undefined at the thresholds. Ordinary
back-propagation through it delivers zero gradient to the weights, and learning stops. Some
surrogate gradient is required.

**The moment view of a weight matrix.** A useful pre-method frame: for a weight matrix `W` with
`n·m` entries, the mean absolute value `(1/nm)Σ|W_ij|` summarizes its typical magnitude in a
single, cheap, outlier-robust scalar (a sample of `E|W|`), whereas the maximum absolute value
is set by a single extreme entry. The two behave very differently as the scale that maps a
real-valued matrix onto a fixed discrete grid.

**Diagnostic observations that the design must respect.** Two empirical phenomena, established
on existing systems, frame the choices. First, **weights are much easier to quantize than
activations**: across the PTQ literature, weight-only schemes survive far lower bit-widths than
weight-and-activation schemes (e.g. weight-only 2-bit remains usable while 4-bit *activation*
quantization already collapses many models), because activations carry per-token outliers with
large dynamic range. Second, in low-bit training **a small update to a high-precision quantity
frequently produces no change at all in its discrete image** — most SGD steps are smaller than
the gap between adjacent discrete levels — so the discrete weights move in rare, abrupt flips
rather than smoothly, which slows early convergence.

## Baselines

The prior art a low-bit-training method is measured against and reacts to.

**Straight-through estimator (Bengio, Léonard & Courbariaux 2013).** For a stochastic or hard
threshold unit whose output is a non-differentiable function of its input, they study gradient
estimators that let such a unit be trained by ordinary back-propagation. The estimator that
works best in practice is the "straight-through" one: in the backward pass, treat the hard
nonlinearity as if it were the identity, i.e. copy the gradient of the unit's output straight
to its input. This converts an un-trainable hard decision into a trainable one at the cost of a
biased gradient. **Gap it leaves:** it is stated for stochastic binary *neurons / activations*;
it does not, on its own, say how to carry a high-precision learnable parameter behind a hard
weight quantizer, nor how to keep that parameter from drifting arbitrarily far once the
quantizer saturates.

**BinaryConnect (Courbariaux, Bengio & David 2015).** Constrain the weights used in the forward
and backward propagations to `{-1,+1}` via the sign function, so that multiply-accumulates
become additions. The mechanism that makes this trainable: keep a **latent full-precision
weight** `w`, binarize it on the fly each forward pass (`w_b = sign(w)`), and apply the SGD
update to the latent `w` while ignoring the binarization in the update. Their argument is
explicit — SGD needs high resolution in the *accumulator* (it averages many small noisy
gradient contributions), but tolerates noise in the *propagated* weight, and discretization is
just such a noise (even a useful regularizer). The backward pass passes gradient through a clip
of the latent weight. **Gap:** weights are forced to exactly `±1` with no per-matrix magnitude,
so a layer whose real weights are all small or all large cannot match its own scale; on
large-scale problems this rigidity costs accuracy.

**Binarized Neural Networks (Hubara, Courbariaux, Soudry, El-Yaniv & Bengio 2016).** Binarize
both weights and activations to `{-1,+1}`, turning the matmul into XNOR-and-popcount. Their
refinement of the straight-through estimator is the load-bearing one: use the **saturating**
version, `g_r = g_q · 1_{|r|≤1}` — pass the gradient through where the pre-quantization value
`r` lies in `[-1,1]`, and **cancel it where `|r|>1`**. The htanh `clip(r,-1,1)` is the implied
forward surrogate. They report that *not* cancelling the gradient outside `[-1,1]` significantly
worsens results: without it, a latent weight already past the grid keeps receiving a push in
the same direction and runs off to infinity. **Gap:** still a single binary level with, at
most, a fixed scale; no notion of how to set the magnitude optimally, and only a `±1` grid.

**XNOR-Net (Rastegari, Ordonez, Redmon & Farhadi 2016).** Approximate a real weight filter `W`
by `α·B` with `B∈{-1,+1}ⁿ` and a positive scalar `α`, chosen to minimize the reconstruction
error. Solving `min_{α,B} ||W − αB||²`: expanding gives `α²n − 2α WᵀB + c`, so the optimal sign
pattern is `B* = sign(W)`, and setting the derivative in `α` to zero gives the closed form
`α* = WᵀB*/n = (1/n)Σ|W_i| = (1/n)||W||₁` — **the optimal per-tensor scale is the mean absolute
weight**. Training keeps full-precision weights, binarizes in the forward/backward, and updates
the real weights; gradient through the scaled sign uses the saturating STE. **Gap:** the grid
is still binary `{-1,+1}` (the scale is optimal *for that grid*); the analysis is done for
convolutional vision models, and the question of how it behaves at LLM scale, and how to add
more than two levels, is open.

**LLM.int8() absmax activation quantization (Dettmers, Lewis, Belkada & Zettlemoyer 2022).**
For the *activation* side: symmetric **absmax** 8-bit quantization scales a tensor into the
signed int8 range by `s = 127 / max(|x|)`, then `x_q = round(clip(x·s, -127, 127))`, with no
zero-point so the symmetric `[-127,127]` range is used directly. Because the scale is set by the
largest magnitude, no value is ever clipped. **Gap:** this is post-training and aimed at 8-bit
inference of a finished model; it does not address training a model whose *weights* are far
below 8 bits, and absmax alone does not preserve the output variance once both operands are
quantized.

**Where the binarized-Transformer line stalls.** Prior binarization work centers on
convolutional vision nets, and the few binarized-Transformer studies target machine
translation (encoder–decoder) or BERT (bidirectional encoder) — architectures and scales
unlike a large unidirectional decoder LLM, and not pushed to the model sizes where LLM behavior
emerges. The prior methods give a binary `{-1,+1}` weight with an L2-optimal scale and a
saturating-STE training recipe, but stop at one bit on convolutional/encoder models; carrying
the train-from-scratch recipe to decoder-LLM pretraining while preserving the cost advantage is
the open ground.

## Evaluation settings

The natural yardsticks for very-low-bit decoder-LLM training.

- **Architecture / scale.** A GPT-2-style decoder-only Transformer — here GPT-2 Medium: 24
  layers, 16 heads, hidden size 1024 (~355M parameters) — with linear projections in attention,
  the MLP, and the LM head.
- **Data / tokenization.** A large web-text pretraining corpus (FineWeb 10B sample), GPT-2 BPE
  tokenizer, on the order of 7B training tokens (≈ Chinchilla-optimal `D ≈ 20N` for this size).
- **Training protocol.** Fixed iteration budget (~13.5k iterations), micro-batch 64, gradient
  accumulation 8, two-GPU data-parallel; identical pipeline across the weight-quantization
  variants so only the quantizer changes.
- **Metrics.** Validation cross-entropy / loss on a held-out shard (primary, lower better);
  word-level perplexity on WikiText-2 and LAMBADA (lower better); zero-shot downstream accuracy
  on ARC-Easy, HellaSwag, PIQA, WinoGrande (higher better).
- **Bit-width accounting.** Bits-per-weight is `log2(#levels)`: a binary `{-1,+1}` grid is 1
  bit, a ternary `{-1,0,+1}` grid is `log2 3 ≈ 1.58` bits, a 4-level grid is exactly 2 bits.

## Code framework

The PyTorch substrate is a drop-in projection module that can replace `nn.Linear` wherever a
Transformer uses a dense projection, so that the training loop, optimizer, data pipeline, and
loss are reused unchanged. The existing pieces are only the module shell and its contract: it
owns a full-precision latent `weight` (and optional `bias`) as `nn.Parameter`s; its `forward(x)`
maps `(..., in_features) → (..., out_features)`; and the forward path should be the same in
training and eval. The empty helper slots choose a forward representation for the latent weight
and for the activation, return the scales needed to reconstruct the operands, and decide how
gradients pass through any hard decisions.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_quant(weight):
    # TODO: map the full-precision latent weight to its low-bit forward form.
    # Return (quantized_weight, scale) with quantized_weight * scale ~= weight.
    # Must be differentiable end-to-end (the forward is piecewise-constant in `weight`).
    pass


def activation_quant(x):
    # TODO: map the activation to its low-bit forward form.
    # Return (quantized_x, scale) with quantized_x * scale ~= x.
    pass


class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear with a full-precision latent weight
    and empty forward-representation slots."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))  # latent FP weight
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # TODO: combine the two forward representations, apply their returned
        #       scales, and add bias.
        pass


# existing training loop the module plugs into (unchanged)
def train(model, loss_fn, data_loader, optimizer):
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        logits = model(inputs)            # every projection inside uses the drop-in module
        loss = loss_fn(logits, targets)
        loss.backward()                   # gradients must reach each latent weight
        optimizer.step()
```

The two `weight_quant` / `activation_quant` bodies and the `forward` path are the empty slots.
