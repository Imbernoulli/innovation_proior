# Research question

The largest publicly available generative Transformers — GPT-3-scale models with 175 billion parameters — store their weights in FP16, which means roughly 326 GB of memory just to hold the parameters. That exceeds the capacity of any single accelerator, so even *inference* (not training) forces a multi-GPU deployment with all the cost and complexity that entails. The question is whether we can shrink the weights enough to fit such a model on far fewer GPUs — ideally a single one — without measurably hurting its language-modeling quality.

Concretely: can we take an already-trained model and quantize its weights down to 3 or 4 bits *in one shot*, with no retraining and no fine-tuning, using only a tiny amount of calibration data and a few hours of compute, while keeping perplexity essentially unchanged? Retraining-based compression is out of the question at this scale — it would cost weeks of GPU time. The constraint that makes this hard is that the only post-training methods known to scale to hundreds of billions of parameters are the crudest ones (round each weight to the nearest grid point), and those fall apart below 8 bits.

# Background

**The layer-wise reconstruction view of post-training quantization.** State-of-the-art post-training quantization (PTQ) does not try to preserve the weights themselves; it preserves each linear layer's *output*. For a layer with weights $\mathbf{W}$ and a set of $m$ calibration inputs collected as columns of $\mathbf{X}$, the goal is to find quantized weights $\widehat{\mathbf{W}}$ minimizing

$$\arg\min_{\widehat{\mathbf{W}}}\ \lVert \mathbf{W}\mathbf{X} - \widehat{\mathbf{W}}\mathbf{X}\rVert_2^2 .$$

The grid (the set of representable quantized values) is fixed in advance; individual weights are then free to take any grid value. This objective decomposes over the rows of $\mathbf{W}$: each output channel is an independent least-squares-with-rounding problem sharing the same input statistics.

**Second-order weight selection (the OBS/OBD lineage).** The classical Optimal Brain Damage / Optimal Brain Surgeon framework asks, for a quadratic loss, which single parameter can be removed (or here, rounded) with least increase in loss, and how to update the remaining parameters to compensate. For the layer-wise quadratic above, the Hessian with respect to a row of weights is $\mathbf{H} = 2\mathbf{X}\mathbf{X}^\top$. OBS gives a closed form: rounding weight $q$ to its grid value costs $(\mathrm{quant}(w_q)-w_q)^2 / [\mathbf{H}^{-1}]_{qq}$, and the optimal compensating update to the still-free weights is proportional to the $q$-th column of $\mathbf{H}^{-1}$. After fixing a weight, the row/column for $q$ is deleted from $\mathbf{H}$, and the inverse can be updated in place by one step of Gaussian elimination rather than re-inverted.

**Optimal Brain Quantization (OBQ).** OBQ applies this iteratively to quantization: within each row, repeatedly pick the *next* weight to quantize as the one with smallest such error, round it, push the compensating update onto the remaining free weights, shrink $\mathbf{H}^{-1}$, repeat until the whole row is quantized. It produces strong PTQ accuracy on vision models up to ~100M parameters. The cost is the problem: the per-weight inverse update over a $d_{\text{row}}\times d_{\text{col}}$ matrix gives runtime $O(d_{\text{row}}\cdot d_{\text{col}}^3)$ — cubic in the column count — because each row chooses its own quantization order and therefore maintains its own evolving $\mathbf{H}^{-1}$. At billions of parameters this is hopeless.

**Diagnostic facts about the regime.** Two empirical observations frame the design. First, greedy ordering (always quantize the lowest-error weight next) barely beats arbitrary fixed ordering on large, heavily over-parameterized layers — the advantage of being clever about order shrinks as layers grow. Second, naive round-to-nearest (RTN), the only thing that currently scales, holds up at 8-bit but collapses at 3-bit, where perplexity blows up by orders of magnitude on the largest models. So accuracy at low bit-width is achievable in principle (the accurate-but-slow methods prove it on small models) but not yet at scale.

# Baselines

**Round-to-nearest (RTN).** Each weight is independently mapped to the closest value on a uniform asymmetric per-row min–max grid. Runtime scales trivially, which is why every large-model quantization effort (ZeroQuant, LLM.int8(), nuQmm) ultimately falls back to it for the actual rounding. Limitation: it ignores interactions between weights entirely; at 3–4 bits the accumulated output error destroys accuracy on large models.

**AdaRound.** Learns a per-weight rounding direction (up or down) by annealing a continuous relaxation with a penalty pulling each soft rounding toward a hard grid point. Strong accuracy, but it is an SGD-based optimization per layer — far too slow for billion-parameter models.

**BitSplit / AdaQuant / BRECQ.** BitSplit builds quantized values bit by bit against the squared residual error; AdaQuant directly optimizes quantized weights with straight-through gradients; BRECQ adds Fisher-information weighting and jointly optimizes the layers inside a residual block. All deliver good accuracy on sub-100M-parameter models and all rely on gradient-based inner loops that do not scale.

**OBQ (the direct ancestor).** The most accurate of the one-shot, gradient-free methods (see Background). Its greedy per-row ordering forces a separate inverse-Hessian trajectory per row and a cubic runtime; it tops out around 100M parameters in reasonable time.

**ZeroQuant / nuQmm.** Large-model efforts that pick a finer granularity (group-wise / vector-wise scaling) but still round to nearest. ZeroQuant adds layer-wise distillation, but the largest model it distills is 1.3B, already taking hours.

# Evaluation settings

- **Models.** The OPT family (125M → 175B) and the BLOOM family (560M → 176B); smaller sanity checks on ResNet-18/50 (standard vision PTQ benchmarks) and BERT-base.
- **Calibration data.** A small generic set — 128 segments of 2048 tokens sampled from the C4 web-crawl corpus — with no task-specific data, so results stay zero-shot.
- **Metric.** Perplexity, known to be an especially stringent and quantization-sensitive measure, computed on WikiText-2, Penn Treebank, and C4 by concatenating each validation set, splitting into non-overlapping windows of the model's full 2048-token context, and exponentiating the average next-token negative log-likelihood. Plus zero-shot accuracy on LAMBADA, ARC (Easy/Challenge), and PIQA via the standard LM-evaluation-harness preprocessing.
- **Bit-widths.** 4, 3, and the extreme regime (2-bit and ternary, with grouping).
- **Protocol.** Quantize on a single A100-80GB; load one Transformer block at a time, accumulate that block's layer Hessians from calibration activations, quantize the block, then re-run the calibration inputs through the now-quantized block to produce inputs for the next block.
- **Systems.** PyTorch with HuggingFace model integrations; timing under a HuggingFace-accelerate-style layer-sharded setup where inter-GPU communication is <5% of runtime.

# Code framework

The primitives that already exist: PyTorch linear layers, a calibration loop that streams a few batches through the model to collect per-layer input statistics, and a per-row uniform-grid quantizer. The per-layer compression routine is the empty slot.

```python
import torch

def quantize_to_grid(w, scale, zero, maxq):
    # round-to-nearest onto a fixed uniform integer grid, then map back to float
    q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
    return scale * (q - zero)

class Quantizer:
    # fits a per-row asymmetric min-max grid from observed weight ranges
    def find_params(self, w): ...
    def quantize(self, w): ...

class LayerCompressor:
    """Accumulates this layer's input second-moment statistics over calibration
    batches, then produces quantized weights minimizing the layer's output error."""
    def __init__(self, layer):
        self.layer = layer
        self.W = layer.weight.data.clone().float()
        self.d_row, self.d_col = self.W.shape
        self.H = torch.zeros((self.d_col, self.d_col))  # input second moment
        self.nsamples = 0

    def add_batch(self, inp):
        # TODO: accumulate the (scaled) input second moment that defines the
        #       layer-wise reconstruction Hessian
        pass

    def compress(self, quantizer, blocksize=128, percdamp=0.01):
        # TODO: the contribution — turn self.H and self.W into quantized weights
        #       that minimize ||WX - WhatX||^2, scalably, in one shot
        pass

# driver: load one transformer block, register hooks that feed add_batch,
# run calibration data, then call compress on each linear layer, then
# re-run the calibration inputs through the quantized block for the next one.
```
