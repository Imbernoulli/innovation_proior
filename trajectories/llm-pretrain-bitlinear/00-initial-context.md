## Research question

Pretrain GPT-2 Medium so that the weights actually used in every linear projection come from a
**tiny discrete set** — one bit `{−1,+1}`, ternary `{−1,0,+1}`, or a few-bit grid — during both
training and inference, rather than full-precision floats. A float *latent* weight is kept only as the
optimizer's accumulator; the forward matmul sees the discrete value on every pass. The single thing
being designed is the **quantizer that maps the latent weight to its discrete forward value** (and,
optionally, the activation quantizer that prepares the matmul's other operand). Everything else — the
model, the data, the optimizer, the loop — is fixed. The goal is to minimize held-out validation loss
and preserve downstream language ability while the forward weights stay discrete.

## Prior art before the first rung (low-bit / quantization lineage)

The first rung — binarized weights via the sign function — is itself the resolution of a line of work
on making a network compute with discrete weights. These are the methods the ladder reacts to.

- **Post-training quantization (PTQ; absmax / LLM.int8(), Dettmers et al. 2022).** Round an already
  FP-trained model onto a coarse grid after the fact, scaling by the per-tensor absolute maximum. At
  8 bits this is nearly lossless once activation outliers are handled; pushed toward 1–2 bits the
  rounding error swamps the signal and accuracy detonates. Gap: the model never *learned* the grid, so
  there is no reason its float optimum has a good low-bit neighbor.
- **Quantization-aware training (QAT).** Simulate the rounding inside the training forward pass
  ("fake quant") so the network learns weights that tolerate the grid. Recovers more accuracy than
  PTQ, but the parameter being optimized is still a floating-point number; the discreteness is a
  simulated overlay, not the thing the forward pass natively is. Gap: float weights at heart.
- **Straight-through estimator (STE; Bengio, Léonard & Courville 2013).** The sign/round step has
  zero derivative almost everywhere, which would kill learning. Treat the threshold as the identity on
  the backward pass; in the canonical stochastic-neuron case this is a provably unbiased gradient
  estimator. This is what makes a discrete forward op trainable at all. Gap: it is a *device*, not a
  weight representation — it still needs a quantizer and a scale.
- **Latent high-precision weight (BinaryConnect, Courbariaux, Bengio & David 2015).** A binary
  variable cannot accumulate SGD's tiny noisy steps (a small step rarely flips a sign). Keep a float
  latent weight the optimizer updates and binarize it on the fly each forward pass; "only the expected
  value of the weight needs precision." Discard the latent weight at inference. Gap: demonstrated on
  small image classifiers; a bare sign throws away magnitude.
- **L2-optimal binary scale (XNOR-Net, Rastegari et al. 2016).** Approximating a real weight tensor by
  `α·sign(W)` and minimizing `‖W − α·sign(W)‖²` gives the optimal direction `sign(W)` and the optimal
  scalar `α = mean(|W|)` — the absmean. Both the discrete direction and the per-tensor scale fall out
  of one least-squares problem. Gap: a single bit per weight; no off-state and no magnitude resolution.

## The fixed substrate

The training harness is frozen and must not be touched. **Model:** GPT-2 Medium — 24 layers, 16 heads,
`d=1024`, ~355M parameters — built from the scaffold's `GPT`, in which **every** linear projection
(attention Q/K/V and output, both MLP matrices, and the tied `lm_head`) is a `BitLinear`. The
embeddings, the `LayerNorm`s, and the residuals stay full precision; the token embedding is tied to
`lm_head.weight`. **Data:** FineWeb 10B (`HuggingFaceFW/fineweb`, `sample-10BT`), GPT-2 tokenizer,
~7.1B training tokens (Chinchilla-optimal `D≈20N`). **Training:** 13,535 iterations, micro-batch 64,
gradient accumulation 8, 2-GPU DDP, AdamW (`β=(0.9,0.95)`, weight decay `0.1`, grad-clip `1.0`), a
cosine learning-rate schedule with 4% linear warmup peaking at `6e-4`, `bfloat16` autocast, and
`torch.compile` on the whole model. One structural fact is load-bearing for everything below: **each
transformer block already applies a `LayerNorm` immediately before its attention and MLP projections**,
so the input reaching every `BitLinear` is already normalized — the quantizer does not need to add its
own pre-quant normalization to keep the forward variance sane.

## The editable interface

Exactly one region is editable — the `BitLinear` module of `nanoGPT/custom_pretrain.py`: the two
free functions `weight_quant(weight)` and `activation_quant(x)`, the `BitLinear.__init__`/`forward`,
and a `CONFIG_OVERRIDES` dict (allowed keys: `learning_rate`, `weight_decay`, `warmup_iters`,
`min_lr`, `grad_clip`). The contract: `__init__(self, in_features, out_features, bias=True)` keeps
`self.weight` a `Parameter`; `forward(self, x)` maps `(…, in_features) → (…, out_features)`;
quantization runs in *every* forward pass (no train/eval branch); `weight_quant` returns
`(quantized_weight, scale)` with `quantized_weight * scale ≈ weight` (same convention for
`activation_quant`); helper `autograd.Function`s and learned parameters may be added; the module must
remain `torch.compile`-compatible (no `@torch.compiler.disable`). Every method on the ladder is a fill
of this same contract.

The starting point is the scaffold default below: a **pass-through** — `weight_quant` returns the float
weight unchanged (so the model is just FP GPT-2). Each method replaces these definitions and nothing
else.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 38-115) -- default fill (pass-through)
def weight_quant(weight):
    """Map the latent weight to its forward value; return (forward_weight, scale).
    Default: pass-through (no quantization)."""
    scale = weight.detach().abs().mean()
    return weight, scale


def activation_quant(x):
    """Prepare the layer input for the matmul; return (forward_x, scale).
    Default: pass-through."""
    scale = x.detach().abs().max().clamp(min=1e-12)
    return x, scale


class BitLinear(nn.Module):
    """Linear layer with native low-bit weights for training and inference.
    self.weight is the float LATENT weight the optimizer updates; the forward pass uses
    whatever weight_quant / activation_quant produce. Same path in train and eval."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        w_q, w_scale = weight_quant(self.weight)
        x_q, x_scale = activation_quant(x)
        out = F.linear(x_q, w_q, None)        # default pass-through: scales unused below
        if self.bias is not None:
            out = out + self.bias
        return out
```

## Evaluation settings

Single seed `42` (the leaderboard's fixed seed). Metrics, with the model checkpoint frozen after the
13,535-iteration run: **validation loss** — cross-entropy on a held-out FineWeb shard (lower is
better, primary); **perplexity** on WikiText-2 and LAMBADA (lower is better); **downstream zero-shot
accuracy** on ARC-Easy and HellaSwag (higher is better; PIQA and WinoGrande are also measured but
hidden). Wall-clock (`elapsed`) is logged but not optimized. Identical data, tokenizer, optimizer, and
loop for every fill, so any difference is attributable to the quantizer alone.
