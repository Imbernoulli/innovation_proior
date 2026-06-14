# Context: low-bit linear layers for large language models (circa 2022-2023)

## Research question

A large autoregressive Transformer spends almost all of its memory and almost all of its
arithmetic in one operation: the dense linear projections inside attention (the Q/K/V and
output projections) and the feed-forward network. At inference time those projections are
matrix multiplications against FP16 weight matrices, and as the model grows past a few billion
parameters the cost of *moving those weights* from DRAM into the compute units — the memory
bandwidth — becomes the binding constraint on latency, while the multiply-accumulate energy of
the FP16 matmul dominates the power budget. Halving the bits per weight roughly halves both the
bytes moved and the arithmetic energy, so the appeal of low precision is direct and large.

The precise goal is a linear layer whose forward weights are drawn from a *tiny discrete set*
— in the extreme, just two values — used on **every** forward pass during both training and
inference, not merely at deployment. Such a layer has to (1) replace a standard floating-point
`Linear` with no change to the surrounding architecture; (2) remain trainable from scratch by
backpropagation even though the operation that produces the discrete weights has zero gradient
almost everywhere; (3) keep the forward signal at a healthy scale so that stacking dozens of
such layers does not blow up or collapse the activations; (4) survive being trained at the
multi-billion-parameter scale where LLMs actually live, ideally following the same predictable
loss-vs-size scaling law as a full-precision model; and (5) interact gracefully with the other
quantities in the layer — chiefly the activations, which are themselves expensive to keep in
FP16. Each existing line of work below achieves part of this; none delivers a 1-bit linear
layer that trains a decoder-only LLM at scale. Closing that gap is the problem.

## Background

By this time the dominant recipe for shrinking a trained network's footprint is
**quantization**, and it comes in two flavors. *Post-training quantization* (PTQ) takes an
already-trained FP16 model and rounds its weights (and sometimes activations) to a low-bit grid
after the fact; it is attractive because it touches neither the training pipeline nor the data,
but the model was never optimized for the rounded representation, so accuracy falls — gently at
8 bits, sharply as the grid coarsens. *Quantization-aware training* (QAT) instead simulates the
rounding inside the training forward pass ("fake quantization"), so the network learns weights
that tolerate the grid; it recovers more accuracy but the optimization gets harder as precision
drops, and whether it obeys the neural-scaling law of language models is unknown. Both share a
defining feature: the parameters being optimized are still floating-point numbers, and the
discreteness is a post-hoc or simulated overlay.

A separate, older line constrains weights to be *natively* discrete during the forward
computation. The load-bearing facts it established:

- **A binary weight and a per-tensor scale are the l2-optimal rank-one summary of a real
  weight.** If you must approximate a real weight vector `W` by `α·B` with `B ∈ {−1,+1}ⁿ` and a
  positive scalar `α`, the best `B` in squared error is the elementwise sign of `W`, and the
  best `α` is the mean absolute value of `W`. This is a clean least-squares fact, derived in the
  binary-CNN literature (Rastegari et al. 2016), and it gives both the *direction* (sign) and
  the *magnitude* (a single scale) of the binarization for free.

- **A non-differentiable threshold can still be trained, by passing the gradient straight
  through it.** Bengio, Léonard & Courville (2013) studied gradient estimators for hard,
  stochastic threshold units and showed that treating the threshold as the identity in the
  backward pass yields a usable — in the canonical stochastic-neuron case, provably unbiased —
  estimator of the gradient. This "straight-through estimator" (STE) is what makes a `sign`
  nonlinearity trainable at all: its true derivative is zero almost everywhere, which would kill
  learning, so the backward pass substitutes the identity.

- **Discreteness in the forward pass must be paired with a high-precision shadow of the
  weight.** Courbariaux, Bengio & David (2015) trained networks whose forward and backward
  passes used binary weights `sign(w)`, but accumulated the SGD updates into a *real-valued*
  latent weight, on the argument that "only the expected value of the weight needs high
  precision": stochastic gradient descent makes an enormous number of tiny, noisy steps whose
  signal only emerges after averaging, and a binary variable cannot accumulate them — a small
  step usually fails to flip the sign and is simply lost. They also observed that binarization
  acts as a *regularizer* (weight noise whose expectation is the clean value, in the spirit of
  DropConnect), and clipped the latent weights to a bounded range so they could not drift
  arbitrarily far without affecting the binary forward weight.

- **Centering the weights before binarizing increases the capacity of the binary code.** Work
  on binarized Transformers (Liu et al. 2022) noted that if the real weights have a nonzero
  mean, the sign operation wastes representational power, and that subtracting the per-tensor
  mean before taking the sign measurably improves the binarized layer; they pair this with the
  same absmean scale inherited from the binary-CNN least-squares derivation.

On the activation side, two empirical facts about large language models are by now well
documented and are the reason activations cannot be treated as casually as weights:

- **Activations are intrinsically harder to quantize than weights.** Weight distributions in
  trained Transformers are fairly flat and uniform, easy to round; activation distributions are
  not. (Xiao et al. 2023.)

- **LLM activations carry large-magnitude outlier features that persist in fixed channels.**
  Beyond roughly the 6.7B scale, a small fraction (on the order of 0.1%) of feature dimensions
  take on values tens to a hundred times larger than the rest, and they recur in the same
  channels across tokens (Dettmers et al. 2022; Xiao et al. 2023). Under a single per-tensor
  scale these outliers dominate the dynamic range and squeeze every other value into very few
  effective levels. The standard symmetric scheme for handling them is **absmax** quantization:
  scale by the inverse of the absolute maximum of the tensor and round into the integer range.

There is also a standard piece of forward-stability theory worth stating, because the
binarized layer will have to answer to it. For a linear layer `y = Wx` with the elements of `W`
and `x` treated as independent and identically distributed, the output variance is
`Var(y) = n·Var(wx) = n·E[w²]E[x²]`. Under Kaiming/Xavier initialization a full-precision layer
is arranged so that `Var(y)` sits at order 1, which is what keeps a deep stack numerically
stable. Any change to how `w` is represented changes `E[w²]` and therefore this variance, so a
binarized layer that wants to stack tens of times deep has to account for it. Layer
normalization (Ba et al. 2016) and its group/sublayer placements (Wu & He 2018; the SubLN
placement of Wang et al. 2022) are the available tools for restoring a target variance.

Finally, the **neural scaling law** (Kaplan et al. 2020; Hoffmann et al. 2022) is the backdrop:
full-precision language-model loss falls as a power law `L(N) = aN^b + c` in parameter count,
which is what makes large models worth building. Whether a natively-discrete-weight model obeys
the same law is an open and decisive question — if it does not, low-bit training does not scale.

## Baselines

These are the prior methods a 1-bit LLM linear layer would be measured against and reacts to.

**Post-training absmax / LLM.int8-style quantization (Dettmers et al. 2022).** Round an
FP16-trained model's weights and activations to int8 with a per-tensor (or per-vector) absmax
scale, using the half-width `Q_b = 2^{b−1}` with a signed int8 storage range `[-128, 127]` at
8 bits. At 8 bits this is nearly lossless on LLMs once the outlier features are given special
treatment. **Gap:** it is a deployment-time overlay on a float model; pushed to very low
bit-width — and especially to a single bit — the rounding error swamps the signal and accuracy
collapses, because the model was never trained to live on the coarse grid.

**SmoothQuant (Xiao et al. 2023).** A PTQ method that migrates the activation outlier
difficulty into the weights by an offline per-channel rescaling `Y = (X·diag(s)⁻¹)(diag(s)·W)`,
so that both factors become easy to quantize at W8A8. **Gap:** it is still post-training and
still float-trained; it eases activation quantization but does not contemplate weights at 1 bit,
and at W4A4/W1A8 the activation difficulty re-emerges.

**Weight-only PTQ with error correction — GPTQ (Frantar et al. 2023), QuIP (Chee et al.
2023).** Quantize weights to 4 or even 2 bits while leaving activations in FP16, using
second-order/error-feedback corrections to choose the rounding so as to minimize the layer
output error. **Gap:** activations stay full precision (so the activation-side cost is not
addressed), and these methods are tuned for 2-4 bits; 1-bit weight-only quantization of an
FP16-trained LLM is outside their reach.

**Binarized convolutional networks — BinaryConnect (Courbariaux et al. 2015), XNOR-Net
(Rastegari et al. 2016).** Train from scratch with `sign`-binarized weights (XNOR-Net adds the
absmean scale and, in its full form, binary activations), latent FP weights for the update, and
STE for the gradient. **Gap:** demonstrated on small image classifiers and convolutional
architectures; the binarization recipe is in hand but it has never been carried to a
decoder-only Transformer, nor to the multi-billion-parameter regime, and its forward-variance
behaviour inside a deep residual language model is unexamined.

**Binarized Transformers for BERT / machine translation — BiT (Liu et al. 2022), binarized
NMT (Zhang et al. 2023).** Bring binarization to Transformers, contributing weight centering
before binarization, a learnable "elastic" scale/threshold for activations, and architectural
tweaks for stability under distillation. **Gap:** these target bidirectional encoders (BERT) or
encoder-decoder translation models, both of which are trained at modest scale and often rely on
knowledge distillation from a full-precision teacher; the unidirectional decoder LLM, trained
from scratch and scaled to billions of parameters, is a different regime that none of them
enters.

## Evaluation settings

The natural yardsticks already in use for language models:

- **Autoregressive language-model pretraining** on a large English corpus (e.g. Pile, Common
  Crawl, RealNews, CC-Stories), tokenized with a subword tokenizer, models spanning roughly
  125M to 30B parameters trained from scratch. The primary metric is validation cross-entropy /
  **perplexity** on held-out text (lower is better).
- **Scaling-law fitting**: train a series of sizes, fit `L(N) = aN^b + c` on the smaller models,
  and check whether the law predicts the loss of the larger held-out sizes — the test of
  whether an architectural change preserves predictable scaling.
- **Zero-shot and few-shot downstream accuracy** on commonsense / completion benchmarks —
  HellaSwag, WinoGrande, Winograd, StoryCloze (and, in the surrounding literature, ARC-Easy,
  PIQA, LAMBADA) — higher is better.
- **Efficiency accounting**: arithmetic-operation energy estimated from per-op ADD/MUL energy
  tables at given process nodes (e.g. 7nm, 45nm), and memory footprint, as functions of model
  size — the inference-cost axis a low-bit method is meant to win on.
- Protocol: identical data and training configuration for the low-bit model and its FP16
  Transformer baseline, so differences are attributable to the layer, not the recipe.

## Code framework

The new layer is a drop-in replacement for the standard floating-point linear projection used
throughout a Transformer; the surrounding harness — the data pipeline, the embeddings, the
attention/FFN block structure, the optimizer, the loss, the training loop — already exists and
is unchanged. What is *not* settled is the internals of the projection: how the stored
high-precision parameter is turned into the value actually used in the forward matmul, how the
input to the matmul is prepared, and how a gradient is obtained through whatever
non-differentiable operations that involves. Those are exactly the empty slots. The substrate is
the generic machinery that already exists: a parameter tensor, the elementwise tensor ops, and
`F.linear`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_quant(weight):
    """Turn the stored high-precision weight into the value used in the forward matmul,
    and report any scale needed to undo it afterwards.
    Returns (forward_weight, scale). Must admit a gradient back to `weight`."""
    # TODO: the weight representation we will design.
    pass


def activation_quant(x):
    """Prepare the layer input for the forward matmul, and report any scale needed to undo it.
    Returns (forward_x, scale). Must admit a gradient back to `x`."""
    # TODO: the activation representation we will design.
    pass


class LowBitLinear(nn.Module):
    """Drop-in replacement for nn.Linear. Stores a high-precision weight as a Parameter;
    the forward pass uses whatever representation weight_quant / activation_quant produce.
    The same path is taken during training and eval (no separate fake-quant branch)."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # TODO: build the forward weight and the forward input from self.weight and x,
        #       run the matmul, and undo the scales. This body is what we will design.
        pass


# existing Transformer training loop the layer plugs into, unchanged
def train(model, loss_fn, data_loader, optimizer):
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        logits = model(inputs)                 # forward through the existing Transformer
        loss = loss_fn(logits, targets)        # existing next-token cross-entropy
        loss.backward()                        # backprop must reach self.weight through the layer
        optimizer.step()                       # high-precision optimizer update on the latent weight
```

The optimizer keeps and updates the high-precision `weight`; the forward pass uses the
representation the two `quant` functions and `forward` produce. Those three bodies are the
slots the new layer fills.
