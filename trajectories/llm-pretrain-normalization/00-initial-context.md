## Research question

In a GPT-style decoder, the only thing being designed here is the **per-block normalization and the way
attention and MLP are wired into the residual stream**. Everything else — tokenizer, data, optimizer,
schedule, evaluation — is frozen. The default block is the standard Pre-LN sandwich: a `LayerNorm` with
learned scale *and* bias in front of each sublayer, `x = x + Attn(LN(x))` then `x = x + MLP(LN(x))`. The
question is whether a different normalization rule (what statistic, with which affine parameters) or a
different placement / wiring of those two sublayers lowers the validation cross-entropy of a 355M GPT-2
Medium pretrained on ~7B tokens. The single edit surface is the `LayerNorm` class and the `Block` class;
the win or loss is read off one number, FineWeb validation loss, with downstream perplexity and accuracy
as secondary witnesses.

## Prior art before the first rung

The default block is itself the resolution of a line of work on where to put normalization in a residual
transformer; the first rung reacts to exactly this lineage.

- **Post-LN, the original transformer (Vaswani et al. 2017).** Normalize *after* the residual add:
  `x = LN(x + Attn(x))`. It trains well shallow, but in deep stacks the expected gradient at the output
  layer is well-conditioned while the gradient at early layers is not — the norm sits on the main path, so
  the identity shortcut is broken and signal must pass *through* a normalization at every block. Deep
  Post-LN needs careful warmup and is prone to divergence. Gap: unstable / warmup-sensitive at depth.
- **Pre-LN (Xiong et al. 2020, arXiv:2002.04745).** Move the norm *inside* the branch:
  `x = x + Attn(LN(x))`. Now the residual is a clean identity path and the gradient is bounded at
  initialization independent of depth, so training is robust without warmup gymnastics. This is what the
  default block uses and what every rung here keeps as the substrate. Its one documented cost: with both
  trained successfully, Pre-LN often lands at a *slightly higher* final loss than Post-LN, because the
  un-normalized residual stream lets activation variance grow monotonically with depth. Gap: stable but
  leaves a little final loss on the table.
- **LayerNorm itself (Ba et al. 2016).** Within one token and one layer, subtract the mean and divide by
  the standard deviation of the feature vector, then apply a per-channel gain and bias:
  `LN(a) = (a − μ)/σ · γ + β`. Two reductions (mean, then variance) plus a subtraction, every block,
  every token. The default fill carries the **bias** `β`. Gap: pays for a mean-subtraction whose
  contribution to the stabilization is not obviously load-bearing, and carries an affine bias whose value
  is unclear in a pre-norm transformer.
- **Parallel attention + MLP (GPT-J, Wang & Komatsuzaki 2021; PaLM, Chowdhery et al. 2022,
  arXiv:2204.02311).** Compute both sublayers from the *same* normalized input and sum them into the
  residual in one step, `x = x + Attn(LN(x)) + MLP(LN(x))`, instead of sequentially. Halves the number of
  norms per block and shortens the sequential depth; reported ~15% faster at large scale with no quality
  loss at 62B, but a measurable quality loss at small scale. Gap: trades a little quality for speed, and
  the trade worsens as the model shrinks.

## The fixed substrate

A nanoGPT pretraining loop is frozen and must not be touched. The model is **GPT-2 Medium**: 24 layers,
16 heads, `n_embd=1024`, `bias=False` at the config level, weight-tied embeddings, learned absolute
position embeddings, GELU MLP with 4× expansion, Flash attention, ~355M parameters. Training is fixed at
12,030 iterations, micro-batch 96, gradient accumulation 6 (split across a 2-GPU DDP run), AdamW
(`β=(0.9, 0.95)`, weight decay 0.1, decoupled so the norm gains never decay), a cosine schedule with 4%
linear warmup and `min_lr = lr/10`, grad-clip 1.0, bf16 autocast, `torch.compile`. The data is FineWeb
`sample-10BT` tokenized with the GPT-2 BPE (~7.1B training tokens). Initialization is the nanoGPT recipe:
`N(0, 0.02)` on linears and embeddings, and the residual-projection (`c_proj`) weights scaled by
`1/√(2·n_layer)` so the residual stream variance does not explode with depth.

The loop also fixes the contract every rung must honor: the model is built from a `GPTConfig`, each
`Block` receives that config, the stack runs `for block in self.transformer.h: x = block(x)` followed by a
final `ln_f` norm before the tied LM head, and the optimizer groups parameters by dimensionality — tensors
with `dim ≥ 2` decay, tensors with `dim < 2` (which includes every norm `weight` and `bias`) do not. Any
new normalization parameter therefore inherits no-weight-decay automatically as long as it stays 1-D.

## The editable interface

Exactly two regions of `nanoGPT/custom_pretrain.py` are editable, plus a small whitelisted config slot:

1. **The `LayerNorm` class** (the normalization rule and its affine parameters).
2. **The `Block` class** (how the two sublayers are normalized, placed, and combined into the residual).
3. **`CONFIG_OVERRIDES`** — a dict allowing only `learning_rate`, `weight_decay`, `warmup_iters`,
   `min_lr`, `grad_clip`. No other hyperparameter, and nothing in the data / tokenizer / optimizer
   construction / evaluation, may change.

The contract: whatever `LayerNorm(ndim, bias)` is, the rest of the model constructs it with those two
positional arguments (`bias=config.bias`, which is `False` here), so a replacement must accept the same
signature even if it ignores `bias`. Whatever `Block(config)` is, it must expose `forward(x) -> x` of the
same shape and must keep the residual stream intact for the next block and for `ln_f`. The naming inside a
block is free (the rest of the model never reaches into a block's submodules). Each rung replaces exactly
these definitions and nothing else.

The starting point is the scaffold **default fill**: standard `LayerNorm` with bias, in a sequential
Pre-LN block.

```python
# EDITABLE regions of nanoGPT/custom_pretrain.py — default fill

# ── Normalization (lines 22–31) ──────────────────────────────────────────────
class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# ── Transformer Block (lines 88–100) ─────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ── Whitelisted hyperparameter overrides (lines 245–247) ─────────────────────
# Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
CONFIG_OVERRIDES = {}
```

## Evaluation settings

One seed (42). The primary metric is **FineWeb validation cross-entropy** (`val_loss`, lower is better),
estimated over 200 batches at the end of training. Secondary witnesses, all reported on the same
checkpoint: **WikiText-2 perplexity** and **LAMBADA perplexity** (lower is better, computed as
`exp(mean chunk loss)` over non-overlapping `block_size` windows), and zero-shot downstream accuracy via
the lm-evaluation-harness on **ARC-Easy** and **HellaSwag** (higher is better), with PIQA and WinoGrande
tracked but held out. Wall-clock (`elapsed`) is logged but is not the objective. The comparison anchor is
the default fill above — standard `LayerNorm` with bias in a sequential Pre-LN block.
