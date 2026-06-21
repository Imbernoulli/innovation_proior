# Context: Memory-Efficient Finetuning of Very Large Language Models

## Research question

Finetuning a large language model on a downstream task or instruction dataset is the most effective
way to improve it, but the memory cost scales with model size. A 65B-parameter model in 16-bit
weights alone is ~130 GB — before any optimizer state, gradients, or activations — so full
finetuning requires a multi-GPU server. Parameter-efficient finetuning (LoRA) trains a small adapter
while freezing the base model, but still holds the frozen base weights in memory at 16-bit, so the
base model remains the dominant cost. The question is how to finetune a model as large as 65B
parameters on a single consumer/data-center GPU (24–48 GB) while preserving task performance.

## Background

**Block-wise k-bit quantization.** Quantization maps a high-precision tensor to fewer bits. The
standard absmax scheme rescales an FP32 tensor into the target integer range by its absolute
maximum: X^Int8 = round( (127 / absmax(X^FP32)) · X^FP32 ) = round(c · X), with c the *quantization
constant* (scale); dequantization divides by c. A single large-magnitude *outlier* wastes the
range — most values then crowd into a few bins. The fix is **block-wise** quantization: flatten the
tensor, slice it into contiguous blocks of size B, and quantize each block independently with its
own constant c_i (n = (b·h)/B constants). Small blocks limit the damage any one outlier can do, at
the cost of storing many constants.

**Quantile quantization** (Dettmers et al. 2022). An information-theoretically optimal code assigns
an *equal number of input values to each quantization bin* — i.e. the bin edges are the *quantiles*
of the input distribution. For an arbitrary tensor this needs an empirical CDF, which is expensive,
and fast quantile-approximation algorithms have large errors precisely on the outliers (the
important values). A relevant empirical fact: pretrained neural-network weights are approximately
zero-centered Gaussian with some standard deviation σ.

**LoRA** (Hu et al. 2021). Freeze the base weight W and add a trainable low-rank update. For a
projection Y = XW with W ∈ R^{h×o}, LoRA computes Y = XW + s·X L₁ L₂, where L₁ ∈ R^{h×r},
L₂ ∈ R^{r×o}, r ≪ min(h,o), and s a scaling scalar. Only L₁,L₂ train; gradients flow *through* the
frozen W to reach them. This makes the trainable parameter count tiny (commonly ~0.2% of the model).

**Where finetuning memory actually goes (a diagnostic).** A measured breakdown for a 7B LLaMA on
FLAN v2, batch size 1, LoRA at 0.2% of weights: the LoRA *parameters* take only 26 MB, but the LoRA
*input/activation gradients* take 567 MB. With gradient checkpointing (Chen et al. 2016 — recompute
activations in the backward pass instead of storing them) the activation gradients drop to ~18 MB
per sequence — now *smaller* than the LoRA weights. The frozen 4-bit base model would consume
~5,048 MB. Gradient checkpointing is standard practice in this setting. The LoRA parameter count is
a small part of this budget.

**Gradient-checkpointing memory spikes.** Checkpointing recomputes activations during the backward
pass; for a long-sequence mini-batch the transient activation memory spikes during these
recomputations.

**The precision/precedent on 8-bit.** 8-bit quantization is known to be essentially lossless for
these tensors (Dettmers et al. 2022).

## Baselines

- **Full 16-bit finetuning** (BF16). Update all weights; the accuracy target. Default
  hyperparameters are often undertuned, so a fair baseline needs an lr/batch sweep.
- **16-bit LoRA** (Hu et al. 2021). Freeze base at 16-bit, train low-rank adapters. Standard
  practice applies LoRA only to the query/value attention projections.
- **8-bit / Int8 quantized base** + adapters. Halves base memory vs. 16-bit; lossless in practice.
- **4-bit Float (FP4) / 4-bit Integer (Int4) quantized base.** Smallest storage footprint; uses an
  evenly-spaced 4-bit grid.

## Evaluation settings

- **Post-quantization quality of data types** (no finetuning): zero-shot accuracy (Winogrande,
  HellaSwag, PiQA, Arc-Easy, Arc-Challenge) and language-modeling perplexity for OPT, BLOOM, Pythia,
  LLaMA across 125M–65B, comparing regular 4-bit integer/float grids with a Gaussian-aware 4-bit
  candidate and optional quantization-metadata compression (setup of Dettmers et al. 2022).
- **Recovering quality via 4-bit adapter finetuning**: RoBERTa and T5 (125M–3B) on GLUE and
  Super-NaturalInstructions, comparing 16-/8-/4-bit adapters to full 16-bit finetuning; and LLaMA
  7B–65B finetuned on Alpaca and FLAN v2, evaluated on MMLU 5-shot accuracy, comparing 4-bit
  storage candidates to BF16/Int8.
- **Instruction/chatbot finetuning**: LLaMA 7B–65B on instruction datasets (e.g. Alpaca, FLAN v2,
  OASST1), evaluated on MMLU and on chatbot benchmarks (Vicuna) via human and GPT-4 judgments.
- **Training protocol**: Adam (β₂=0.999), constant learning-rate schedule, max grad norm 0.3,
  group-by-length batching, gradient checkpointing throughout; LoRA rank, α, dropout, and layer
  coverage searched.

## Code framework

The primitives that already exist: a pretrained Transformer with linear projections, block-wise
absmax quantize/dequantize, the LoRA factorized update, gradient checkpointing, and an Adam-family
optimizer.

```python
import torch
import torch.nn as nn

def quantize_blockwise(W_fp, blocksize):
    """Absmax block-wise quantization (already standard)."""
    blocks = W_fp.flatten().view(-1, blocksize)
    c = blocks.abs().max(dim=1, keepdim=True).values          # per-block scale
    codes = map_to_grid(blocks / c)                            # TODO: which grid? (data type)
    return codes, c

def dequantize_blockwise(codes, c, blocksize, out_shape):
    # TODO: invert; reconstruct compute-precision weights
    pass

def make_grid(k):
    # TODO: what are the 2^k code points for a 4-bit data type?
    pass

def store_quant_constants(c):
    # TODO: the per-block constants are themselves sizeable metadata.
    pass

class QuantLinear(nn.Module):
    """A frozen linear layer whose weight is stored quantized, plus a LoRA adapter."""
    def __init__(self, W_fp, r, alpha, blocksize):
        super().__init__()
        self.codes, self.c = quantize_blockwise(W_fp, blocksize)   # frozen, low-bit storage
        self.h, self.o = W_fp.shape
        # TODO: low-rank trainable adapter (which factor starts at zero?)
        self.L1 = None
        self.L2 = None
        self.scaling = alpha / r

    def forward(self, x):
        W = dequantize_blockwise(self.codes, self.c, ...).to(x.dtype)  # to compute dtype
        # TODO: base projection (frozen, dequantized) + scaled low-rank update
        pass

def build_optimizer(params):
    # TODO: optimizer states can spike past GPU memory during checkpointed long-sequence
    #       backward passes — how to survive transient OOM without crashing?
    pass
```
