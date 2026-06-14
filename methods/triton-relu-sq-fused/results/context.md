# Context: making the Transformer feed-forward block cheaper to train (circa 2019-2021)

## Research question

A decoder-only Transformer language model spends most of its compute in two places: the
attention block and the position-wise feed-forward block (FFN). The FFN is the larger of the
two in raw FLOPs at the scales people actually train: for a hidden width `d` it projects each
token up to a wider intermediate width (conventionally `4d`), applies a pointwise
nonlinearity, and projects back down. Concretely, per token it is two dense matmuls of shape
`(d → 4d)` and `(4d → d)` with an elementwise activation sandwiched between them. As model
and data scale, the cost of pretraining these models has grown to the point where shaving a
constant factor off the training cost is worth a great deal.

The precise goal is to reduce the *training cost to reach a target quality* of an
auto-regressive Transformer language model, by improving the FFN block, in two senses that
trade against each other and must be considered together. (1) **Sample efficiency** — a
modification to the block that makes the model reach a lower validation loss in fewer
optimizer steps is a win even if each step is slightly slower, because total compute is steps
times step-time. (2) **Step throughput** — the same mathematical block executed with less
wasted memory traffic finishes each step faster. The pain is that the obvious dimensions to
tune — the activation function and the way the block is executed on the GPU — are usually
treated as fixed: ReLU or GELU, run as a stack of generic library ops. A solution would have
to (a) find an activation that improves quality-per-step without adding parameters or
fragile numerics, and (b) execute the block so that the parts of it that are pure memory
traffic stop dominating, while keeping the math and the gradients exactly correct so it drops
into an existing training loop with no other change.

## Background

**The position-wise FFN and its activation.** In the original Transformer the FFN is
`FFN(x, W1, W2, b1, b2) = max(0, x W1 + b1) W2 + b2`, i.e. an up-projection, a ReLU, and a
down-projection. In the T5 line the biases are dropped, so the working form is
`FFN_ReLU(x, W1, W2) = max(x W1, 0) W2`. The activation is applied elementwise to the wide
`4d` intermediate. Two families of replacements were on the table. The first is *pointwise*
activations that keep the two-matrix structure but swap the nonlinearity: GELU,
`GELU(x) = x Φ(x)`, with the common tanh approximation
`0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))`, and Swish, `Swish_β(x) = x σ(β x)`. The second is
*gated* activations. Dauphin et al. (2017) introduced the Gated Linear Unit, the
component-wise product of two linear projections of the input, one passed through a
nonlinearity: `GLU(x, W, V) = σ(x W) ⊙ (x V)`. Shazeer (2020) tested GLU variants inside the
FFN and reported quality gains: with `⊙` the elementwise product,

```
FFN_GLU(x, W, V, W2)    = (σ(x W) ⊙ (x V)) W2
FFN_ReGLU(x, W, V, W2)  = (max(0, x W) ⊙ (x V)) W2
FFN_GEGLU(x, W, V, W2)  = (GELU(x W) ⊙ (x V)) W2
FFN_SwiGLU(x, W, V, W2) = (Swish_1(x W) ⊙ (x V)) W2
```

These gated forms have *three* weight matrices (`W`, `V`, `W2`) rather than two, so to hold
parameter and FLOP count fixed against the two-matrix FFN one shrinks the intermediate width
`d_ff` by a factor of `2/3`. The reported observation is that the gated variants, GEGLU and
SwiGLU especially, reach lower held-out perplexity than the plain ReLU/GELU FFN at matched
parameters and compute — a per-step quality gain attributed loosely to the multiplicative
interaction `(x W) ⊙ (x V)` the gate introduces.

**Rectified polynomials as activations.** Krotov & Hopfield (2016), working on associative
memory rather than language models, studied polynomial and rectified-polynomial energy terms
`F(s) = s^p` for `s ≥ 0` and `0` for `s < 0`. In their notation `p = 2` recovers the classical
Hopfield energy, and larger `p` makes each stored pattern's contribution sharper, so more
patterns can be packed before they interfere — the random-memory capacity scales like
`K_max ≈ α_p N^{p-1}`. The feed-forward-network correspondence in the same paper is one degree
lower: the `p = 2` energy case corresponds to a ReLU activation, while the next case, `p = 3`,
corresponds to a rectified parabola. Their stated open question is whether the positive branch
of a rectified unit should grow linearly, sub-linearly, or faster than linearly, and whether
higher rectified polynomials could have better computational properties than ReLU. As of this
point that question is open for Transformers: rectified polynomials above the ReLU case are
studied in principle but not in common use as a Transformer activation. The pointwise
activations above all answer the shape question with "asymptotically linear" (ReLU, GELU,
Swish all approach a line as `x -> infinity`).

**The GPU memory hierarchy and where the FFN wastes time.** A modern GPU has a steep memory
hierarchy: a large but slow off-chip DRAM (HBM), a small fast on-chip SRAM / shared memory
per streaming multiprocessor, and registers. The roofline model says a kernel is either
*compute-bound* (limited by arithmetic throughput) or *memory-bound* (limited by HBM
bandwidth), depending on its arithmetic intensity — FLOPs performed per byte moved. A dense
matmul of reasonably large matrices is compute-bound: it does `O(MNK)` FLOPs for `O(MN+NK+MK)`
bytes, high intensity. An elementwise op is the opposite extreme: applying a function to an
`(M, N)` tensor reads `MN` elements and writes `MN` elements while doing `O(MN)` cheap flops,
intensity near the floor, so it is purely bandwidth-bound — its runtime is essentially the
time to stream the tensor through HBM twice. The diagnostic accounting is explicit: executing
a chain like up-projection → activation → down-projection as three separate library kernels
forces the wide `(tokens × 4d)` intermediate to be *written* to HBM by the matmul, *read back*
by the activation, *written* again, and *read* a third time by the second matmul. The
activation step alone costs a full read-plus-write of the largest tensor in the block, and
that traffic is pure overhead — it does no useful arithmetic beyond the pointwise function.

**The cost of relying on vendor kernel libraries.** High-performance matmul on GPUs lives in
proprietary vendor libraries (cuBLAS, cuDNN). They are excellent at a dense matmul in
isolation but expose a fixed menu of operations; they cannot be customized to staple a
deep-learning-specific elementwise step onto a matmul. So the standard practice — call the
library matmul, then call a separate activation kernel — is locked into paying the round-trip
through HBM for the intermediate, precisely because the activation cannot be expressed inside
the library's matmul. Tile-level GPU programming systems (Tillet et al. 2019) emerged to break
this: they let one write a matmul as a blocked accumulation over statically-shaped tiles —
each program instance owns an output tile, loads tiles of the operands into SRAM, accumulates
`acc += dot(a, b)` over the contraction dimension in a register-resident fp32 accumulator, and
writes the result — *with the body of the epilogue under the programmer's control*. The
canonical such matmul keeps the accumulator in fp32 for accuracy and down-casts on store, and
reorders the output tiles into groups to raise L2 reuse. The relevant capability is that
arbitrary code can run on the accumulator after the contraction loop and before the store.

## Baselines

**Plain ReLU / GELU FFN (Vaswani et al. 2017; T5 line; GELU per Hendrycks & Gimpel 2016,
used in BERT/GPT).** Two matmuls with a pointwise activation between: `max(x W1, 0) W2` or
`GELU(x W1) W2`. Cheap (two matrices) and the default everywhere. Core idea: a wide nonlinear
expansion-then-contraction per token. **Limitation:** the activation is asymptotically linear
for large positive input and contains no interaction between coordinates of the
pre-activation — each output unit is a fixed univariate function of one pre-activation unit.
At matched parameters the gated variants below reach lower perplexity, which says this purely
pointwise, asymptotically-linear family is leaving quality on the table.

**GLU-variant FFNs — ReGLU, GEGLU, SwiGLU (Dauphin et al. 2017; Shazeer 2020).** Replace the
first linear-plus-activation by a gate: `(act(x W) ⊙ (x V)) W2`, where the second linear
projection `x V` multiplies the activated first projection elementwise. Core idea: the
multiplicative term lets the block represent products of two learned linear features, an
input-dependent gating that a single pointwise activation cannot. Reported to improve
held-out perplexity over ReLU/GELU at matched parameters. **Limitation:** it pays for the
gate with a *third* weight matrix `V`; to keep parameters and FLOPs fixed it must shrink the
intermediate width by `2/3`, so the gain comes bundled with extra implementation surface (a
third projection, the `2/3` bookkeeping) and a narrower nonlinear bottleneck. The
multiplicative interaction is grafted on by adding a matrix rather than coming for free from
the nonlinearity itself.

**Vendor-library execution of the block (cuBLAS/cuDNN + framework elementwise ops).** Run the
FFN as: library matmul to the wide intermediate, framework activation kernel, library matmul
back down; in mixed precision the matmuls accumulate in fp32 and store in bf16/fp16.
**Limitation:** every operator boundary is an HBM round-trip. The wide intermediate `(tokens ×
4d)` is materialized to HBM after the first matmul, re-read by the activation, re-written, and
re-read by the second matmul; the activation, being bandwidth-bound, is dominated by that
traffic rather than by its trivial arithmetic. The library cannot absorb the activation, so
this overhead is structural, not a tuning problem.

**Tile-level custom matmul with a fixed copy-back epilogue (Tillet et al. 2019; the canonical
tiled-matmul recipe).** A hand-written blocked matmul: tile the output, stream `BLOCK_K`
slabs, accumulate `acc += dot(a, b)` in an fp32 register accumulator, down-cast, store, with
grouped tile ordering for L2 reuse. Core idea: match vendor matmul throughput while keeping
the kernel body editable. **Limitation as it sits:** on its own it is *just a matmul* — its
epilogue is a plain down-cast-and-store. It demonstrates the *mechanism* by which extra work
could be folded into the accumulator before the store, but a bare matmul leaves the activation
to a separate downstream kernel, so by itself it does not remove the FFN's activation
round-trip. What to put in that epilogue, and which activation makes the whole block both
faster and better, is not settled by the matmul recipe.

## Evaluation settings

The natural yardsticks, which existed before any of this:

- **Auto-regressive (decoder-only) language modeling.** Search-scale: One Billion Words
  Benchmark (LM1B), sequence length 64, ~35M-parameter models, fixed training-compute budget.
  Transfer scale: C4 and PG19 in the T5 codebase, sequence length 512, ~110M parameters; and
  a GPT-3-XL-style setup, sequence length 1024, ~1.9B parameters, for one-shot downstream
  evaluation. Metric: validation perplexity / cross-entropy loss (lower better) at a fixed
  training-compute budget, plus per-step / wall-clock training time as the throughput axis.
- **Standard pretraining recipe:** Adafactor optimizer, ~10K warmup steps, base learning rate
  ~0.01, reciprocal-square-root decay, regularization disabled, default Transformer
  hyperparameters carried over unchanged. Codebases: Tensor2Tensor, T5, Lingvo.
- **Activation/architecture comparison set:** vanilla Transformer (ReLU), Transformer+GELU,
  and a strengthened Transformer++ (RMS normalization, Swish, SwiGLU gate) as the
  quality baselines a new FFN would be measured against at matched parameters and compute.
- **Kernel-throughput yardstick:** for the GPU-execution axis, achieved matmul throughput
  (TFLOPS) versus the vendor library on the relevant matrix shapes, and end-to-end training
  step time, on the target accelerators (TPU/GPU; the concrete kernel target here is a
  GPT-2-scale decoder trained on a web-text corpus, measured by validation loss and elapsed
  training time).

## Code framework

The block being modified is the FFN inside a nanoGPT-style GPT-2 decoder. Everything around
it already exists: the embedding, the causal self-attention, the layer norm, the residual
adds, the AdamW/Adafactor optimizer, the autocast mixed-precision context, the DDP training
loop. The FFN class is wired so that the entire forward pass of the block — the up-projection,
the activation, and the down-projection — goes through one replaceable function, with the
signature held fixed so anything dropped in must produce identical-shaped outputs and play
correctly with autograd. Dropout and the residual add stay outside it. The single empty slot
is *how the up-projection, the nonlinearity, and the down-projection are computed* — what the
nonlinearity is, and how the three steps are executed on the device.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F


def fused_mlp_forward(x, w_fc, w_proj):
    """The FFN core: up-project x, apply a nonlinearity, down-project.

    Fixed contract (must be preserved):
      x:      (B*T, n_embd)            flattened token activations
      w_fc:   (4*n_embd, n_embd)       up-projection weight
      w_proj: (n_embd, 4*n_embd)       down-projection weight
      returns (B*T, n_embd)

    Default substrate: the up-projection, the nonlinearity, and the
    down-projection, computed however we choose. The nonlinearity, and the way
    these three steps are executed on the GPU, are exactly what is to be designed.
    """
    # TODO: compute the up-projection, the nonlinearity, and the down-projection.
    pass


class MLP(nn.Module):
    """Position-wise feed-forward block. Owns the two projection weights;
    delegates the up/activation/down computation to fused_mlp_forward.
    Dropout and the residual add live outside this call."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        out = fused_mlp_forward(x.view(-1, C), self.c_fc.weight, self.c_proj.weight)
        return self.dropout(out.view(B, T, C))


# existing transformer block the FFN plugs into (pre-norm residual)
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1, self.attn = LayerNorm(config.n_embd), CausalSelfAttention(config)
        self.ln_2, self.mlp = LayerNorm(config.n_embd), MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))     # FFN on the residual stream
        return x
```

The training loop draws a batch, runs the stacked blocks, computes the cross-entropy loss,
backpropagates, and steps the optimizer; the only thing left undetermined is the body of
`fused_mlp_forward` and the gradients it implies.
