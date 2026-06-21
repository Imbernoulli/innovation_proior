## Research question

Pretrain GPT-2 Medium so that the weights used in every linear projection come from a **tiny discrete set** — one bit {−1,+1}, ternary {−1,0,+1}, or a few-bit grid — during both training and inference, rather than full-precision floats. A float *latent* weight is kept only as the optimizer's accumulator; the forward matmul sees the discrete value on every pass. The design target is the **quantizer that maps the latent weight to its discrete forward value** (and, optionally, the activation quantizer that prepares the matmul's other operand). The model, the data, the optimizer, and the loop are fixed. The objective is to minimize held-out validation loss and preserve downstream language ability while the forward weights stay discrete.

## Prior art / Background / Baselines

- **Post-training quantization (PTQ; absmax / LLM.int8()).** Quantize an already-trained FP model onto a coarse grid by scaling with the per-tensor absolute maximum. At 8 bits it is nearly lossless; at 1–2 bits the rounding error dominates and accuracy collapses.
- **Quantization-aware training (QAT).** Simulate rounding in the training forward pass ("fake quant") so the network learns weights that tolerate the grid. Recovers more accuracy than PTQ, but the optimized parameter is still floating-point; discreteness is a simulated overlay, not the native forward path.
- **Straight-through estimator (STE).** The sign/round step has zero derivative almost everywhere, which would block gradient flow. STE treats the threshold as the identity on the backward pass, making a discrete forward op trainable.
- **BinaryConnect.** Keep a float latent weight for the optimizer to update, binarize it on the fly for each forward pass, and discard the latent weight at inference.
- **XNOR-Net.** Approximate a weight tensor by `α·sign(W)`, with the L2-optimal direction `sign(W)` and per-tensor scale `α = mean(|W|)`.

## Fixed substrate / Code framework

The training harness is frozen and must not be touched. **Model:** GPT-2 Medium — 24 layers, 16 heads, `d=1024`, ~355M parameters — built from the scaffold's `GPT`, in which **every** linear projection (attention Q/K/V and output, both MLP matrices, and the tied `lm_head`) is a `BitLinear`. The embeddings, the `LayerNorm`s, and the residuals stay full precision; the token embedding is tied to `lm_head.weight`. **Data:** FineWeb 10B (`HuggingFaceFW/fineweb`, `sample-10BT`), GPT-2 tokenizer, ~7.1B training tokens (Chinchilla-optimal `D≈20N`). **Training:** 13,535 iterations, micro-batch 64, gradient accumulation 8, 2-GPU DDP, AdamW (`β=(0.9,0.95)`, weight decay `0.1`, grad-clip `1.0`), a cosine learning-rate schedule with 4% linear warmup peaking at `6e-4`, `bfloat16` autocast, and `torch.compile` on the whole model. Each transformer block applies a `LayerNorm` immediately before its attention and MLP projections, so the input reaching every `BitLinear` is already normalized; the quantizer does not need its own pre-quant normalization to keep the forward variance sane.

## Editable interface

Exactly one region is editable — the `BitLinear` module of `nanoGPT/custom_pretrain.py`: the two free functions `weight_quant(weight)` and `activation_quant(x)`, the `BitLinear.__init__`/`forward`, and a `CONFIG_OVERRIDES` dict (allowed keys: `learning_rate`, `weight_decay`, `warmup_iters`, `min_lr`, `grad_clip`). The contract: `__init__(self, in_features, out_features, bias=True)` keeps `self.weight` a `Parameter`; `forward(self, x)` maps `(…, in_features) → (…, out_features)`; quantization runs in every forward pass (no train/eval branch); `weight_quant` returns `(quantized_weight, scale)` with `quantized_weight * scale ≈ weight` (same convention for `activation_quant`); helper `autograd.Function`s and learned parameters may be added; the module must remain `torch.compile`-compatible (no `@torch.compiler.disable`). The contract bounds every edit.

The starting point is the scaffold default below: a **pass-through** — `weight_quant` returns the float weight unchanged (so the model is just FP GPT-2). Each fill replaces these definitions and nothing else.

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

Single seed `42` (the leaderboard's fixed seed). Metrics, with the model checkpoint frozen after the 13,535-iteration run: **validation loss** — cross-entropy on a held-out FineWeb shard (lower is better, primary); **perplexity** on WikiText-2 and LAMBADA (lower is better); **downstream zero-shot accuracy** on ARC-Easy and HellaSwag (higher is better; PIQA and WinoGrande are also measured but hidden). Wall-clock (`elapsed`) is logged but not optimized. Identical data, tokenizer, optimizer, and loop for every fill, so any difference is attributable to the quantizer alone.
