# Context: native low-bit linear layers for large language models (circa 2023-2024)

## Research question

The cost of serving a large language model is dominated by its linear projections. The bulk
of the compute in a Transformer is matrix multiplication, and in FP16/BF16 that means
floating-point multiply-and-accumulate; on top of the arithmetic, just moving the weight
matrices from DRAM into the on-chip accelerator's SRAM is a memory-bandwidth bottleneck
that grows with model size, and on a power-limited chip the energy of those FP multiplies is
the hard ceiling. The standard lever for this is quantization — represent weights (and
sometimes activations) in fewer bits — but the prevailing recipe quantizes a model that was
*trained in floating point*.

The question is how to design a linear layer whose forward weights are *natively* drawn from
a tiny discrete set on **every** forward pass — during training and inference alike, with no
separate float forward path — so that the projection's matmul becomes (almost) integer
add/subtract and the weight memory collapses by an order of magnitude, while maintaining the
scaling behavior expected from a decoder-only language model.

## Background

By this time, neural language models are known to scale predictably: loss follows a power
law in model size / compute (Kaplan et al. 2020; Hoffmann et al. 2022, "Chinchilla"),
which fixes the natural way to compare any architectural change — hold tokens fixed, sweep
model size, and check whether the loss-vs-size curve still follows a power law and where it
sits relative to the full-precision curve. The de-facto open backbone is LLaMA (Touvron et
al. 2023): pre-normalized decoder blocks using RMSNorm (Zhang & Sennrich 2019), the SwiGLU
gated feed-forward (Shazeer 2020), rotary position embeddings (Su et al.), and no bias
terms. These choices are load-bearing only in that any new layer wants to slot into that
ecosystem (HuggingFace, vLLM, llama.cpp) with minimal friction.

Relevant prior findings:

- **Sub-4-bit post-training quantization.** Post-training quantization (PTQ) is
  attractive because it touches nothing in the training pipeline, but it can only move
  weights to grid points *near* the trained float values. Empirically, PTQ holds up at
  8 and 4 bits but degrades sharply at 2 bits and collapses at 1 bit.

- **Quantization-aware training (QAT).** Training the model to account for reduced precision
  from the start gives better accuracy than PTQ and supports continued training/fine-tuning.

- **The discrete forward operator.** Mapping a real weight onto a discrete set with sign /
  round / clip is piecewise-constant, so its derivative is zero almost everywhere — backprop
  through it transmits nothing. The standard workaround is the *straight-through estimator*
  (STE; Bengio, Léonard & Courville 2013, formalizing Hinton's lecture-15b idea): in the
  backward pass, treat the hard threshold as if it were the identity. It is a biased
  estimator, but for a single layer it has the right sign, and it is what makes training
  through a non-differentiable quantizer possible at all.

- **Real-valued latent weights.** The practice inherited from the binary-network literature
  (Courbariaux et al. 2015) is to keep a *real-valued latent weight* that accumulates the
  updates and is discretized on the fly in each forward pass; the latent copy is discarded
  at inference.

- **Activation outlier features.** Beyond roughly the 6.7B scale, a small fraction (~0.1%)
  of activation feature dimensions develop systematically large magnitudes that appear in
  every layer (Dettmers et al. 2022). Per-row / per-token scaling of activations addresses
  this at finer granularity than per whole tensor.

- **Output variance after quantization.** With standard initialization the output of a
  full-precision projection has variance on the order of 1. Sub-LayerNorm (Wang et al. 2022,
  "Magneto") — an extra normalization placed *inside* each sublayer, before the projections
  — is a stabilizer for large-scale Transformers.

## Baselines

These are the prior methods a native low-bit linear layer would be measured against.

**Absmax / vector-wise 8-bit quantization (Dettmers et al., "LLM.int8()", 2022).** Quantize
a tensor to the signed `b`-bit range by scaling with `Qb / ||x||_∞`, where `Qb = 2^{b-1}`
(`Qb = 128` for 8 bits), then round and clip to the implementable int8 range `[-128, 127]`;
dequantize by the inverse scale. For 8-bit (W8A8) this is nearly lossless. The authors
handle outlier dimensions by peeling them into a separate FP16 matmul.

**Weight-only PTQ: GPTQ and QuIP (Frantar et al. 2023; Chee et al. 2023).** Keep activations
in FP16 and quantize only the weights, using second-order/error-feedback machinery (GPTQ
greedily quantizes columns while updating the rest to compensate; QuIP adds incoherence
processing with guarantees) to push weights to 4 or even 2 bits. These are the strongest
sub-8-bit baselines, with activations left in float.

**Sign-based binary weight networks with an absmean scale (Rastegari et al., "XNOR-Net",
2016).** For a real weight tensor `W`, approximate it as `W ≈ α·B` with `B ∈ {-1, +1}` and a
positive scalar `α`, chosen to minimize the L2 error `||W − α·B||²`. Expanding,
`J(B, α) = α²·BᵀB − 2α·WᵀB + WᵀW = α²·n − 2α·WᵀB + c`; the maximizer of `WᵀB` over
`B ∈ {±1}ⁿ` is `B* = sign(W)`, and setting `∂J/∂α = 0` gives `α* = Wᵀsign(W)/n = (1/n)Σ|Wᵢ| =
mean(|W|)`. So the L2-optimal binary approximation is the sign of the weights scaled by their
mean absolute value, and a matmul against a `{-1,+1}` tensor needs no multiplies — only
additions and subtractions, with one scalar multiply by `α` at the end. This line was
developed and validated on convolutional vision networks.

**Centralized binarization for Transformers (Liu et al., "BiT", 2022).** Before taking the
sign, subtract the mean of the real weights so the binarized set is zero-centered:
`W_B = (||W_R||₁/n)·sign(W_R − W̄_R)`; that work also explores a learnable "elastic"
quantizer that fits a scale and threshold per layer to the data. Centralizing is shown to
increase the information-carrying capacity of the binary weights. Demonstrated on BERT-style
encoders and machine translation.

**Binary/low-bit Transformers for translation and BERT (e.g. binarized NMT; binary-BERT
lines).** Binarization had been carried into Transformers, but for encoder (BERT) or
encoder-decoder (translation) architectures, at modest scale.

## Evaluation settings

The natural yardsticks, all pre-existing:

- **Pretraining corpus and protocol.** Autoregressive language-model pretraining on a
  large English corpus (e.g. the Pile / Common Crawl / RealNews / CC-Stories mixtures, or
  the open RedPajama reproduction of the LLaMA mixture), with a fixed token budget so that
  model sizes can be compared at equal data. Optimizer is Adam; cosine or polynomial LR
  decay with a short warmup; sequence length ~2048.
- **Scaling-law sweep.** Train a series of models across sizes (≈125M up through several
  billion parameters) at a fixed token count, and plot loss against parameter count to fit
  `L(N) = a·N^b + c` and check the power law, comparing against a same-recipe full-precision
  Transformer.
- **Language-model perplexity** on held-out validation sets — WikiText-2 (Merity et al.
  2016) and C4 (Raffel et al. 2019) — lower is better.
- **Zero-shot (and few-shot) downstream accuracy** via the `lm-evaluation-harness` pipeline
  on commonsense / QA tasks: ARC-Easy and ARC-Challenge (Clark et al.), HellaSwag (Zellers
  et al. 2019), WinoGrande (Sakaguchi et al. 2020), PIQA (Bisk et al. 2019), OpenBookQA
  (Mihaylov et al. 2018), BoolQ (Clark et al. 2019); higher is better.
- **Cost metrics for deployment.** Inference memory footprint (GB), decoding
  latency per output token (ms), throughput (tokens/s, by raising batch size to the memory
  limit), and an arithmetic-operations energy estimate from a published per-operation energy
  model (Horowitz 2014; PokeBNN) at a given process node.

## Code framework

A native low-bit linear layer is a drop-in replacement for `nn.Linear` inside an otherwise
ordinary LLaMA-style decoder. What already exists is the surrounding harness: a Transformer
block that calls linear projections for Q/K/V, the attention output, and the two/three
feed-forward matrices; the pre-existing normalization, the optimizer (Adam) keeping
high-precision states, the loss, and the training loop. The straight-through-estimator idiom
already exists as a general tool. What is *not* settled — and is exactly the slot to fill —
is the per-layer forward: how the stored real-valued parameter is turned into the tensor
that actually multiplies the activations, how the activations themselves are handled, and how
the gradient is routed back to the latent parameter.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ste(x_real, x_discrete):
    """Straight-through estimator: forward returns the discrete value, backward
    passes the gradient straight to x_real (treats the discretizer as identity)."""
    return x_real + (x_discrete - x_real).detach()


def weight_map(weight):
    """Turn a stored real-valued weight tensor into the tensor that will actually
    be multiplied in the forward pass (plus whatever scalar(s) are needed to keep
    it a faithful stand-in for the real weights).

    The discretizer and its scale are exactly what we have to design."""
    # TODO: the forward weight tensor we will define here.
    pass


def activation_map(x):
    """Turn an activation tensor into the tensor that will actually be multiplied,
    plus whatever scalar(s) dequantize the result."""
    # TODO: the forward activation tensor we will define here.
    pass


class LowBitLinear(nn.Module):
    """Drop-in replacement for nn.Linear whose forward weights come from a small
    discrete set on every pass. A high-precision latent weight is stored and
    accumulates updates; it is mapped on the fly in forward()."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))  # latent FP weight
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # normalization to keep the output variance ~1 after the mapping
        # x = normalize(x)                 # an existing LayerNorm-family op
        # w_fwd = ste(self.weight, weight_map(self.weight))
        # x_fwd = ste(x, activation_map(x))
        # out = F.linear(x_fwd, w_fwd)     # the (mostly) low-precision matmul
        # out = dequantize(out)            # undo the scales
        # if self.bias is not None: out = out + self.bias
        # TODO: assemble the forward pass from the pieces above.
        pass
```

The two `# TODO` maps and the assembled `forward` are the unsettled slots; everything else
— the latent parameter, the STE idiom, the normalization family, the `F.linear` matmul, the
optimizer and loop — is already on hand.
