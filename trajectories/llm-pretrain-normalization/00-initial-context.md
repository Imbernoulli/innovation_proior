## Research question

In a GPT-style decoder, the only design surface is the **per-block normalization and the way attention and MLP are wired into the residual stream**. Everything else—tokenizer, data, optimizer, schedule, evaluation—is frozen. The default block is the standard Pre-LN sandwich: a `LayerNorm` with learned scale and bias in front of each sublayer, `x = x + Attn(LN(x))` then `x = x + MLP(LN(x))`. The question is whether a different normalization rule or a different placement/wiring of the two sublayers lowers the FineWeb validation cross-entropy of a 355M GPT-2 Medium pretrained on ~7B tokens. The edit surface is exactly the `LayerNorm` class and the `Block` class; the outcome is read from one number, FineWeb validation loss, with downstream perplexity and accuracy as secondary witnesses.

## Prior art / Background / Baselines

- **Post-LN transformer.** It places `LayerNorm` after the residual add, `x = LN(x + Attn(x))`, so the branch operates on the unnormalized stream while the norm sits on the main path. Deep stacks have ill-conditioned early-layer gradients and require long, careful warmup to train without divergence.

- **Pre-LN transformer.** It moves the norm inside the branch, `x = x + Attn(LN(x))`, leaving the residual stream as a clean identity path and making initial gradients bounded independent of depth. When both setups train to convergence, Pre-LN often reaches a slightly higher final loss than Post-LN.

- **LayerNorm.** It subtracts the token-wise mean, divides by the standard deviation, and applies a learned gain and bias at every block. The per-token normalization therefore requires two reductions and an affine transform, and not every component is known to be load-bearing in a pre-norm block.

- **Parallel attention + MLP block.** It computes both sublayers from the same normalized input and adds them in one step, `x = x + Attn(LN(x)) + MLP(LN(x))`, halving the norms per block and shortening sequential depth. At small scale it gives a measurable quality loss in return for the speedup.

## Fixed substrate / Code framework

The nanoGPT pretraining loop is frozen. The model is **GPT-2 Medium**: 24 layers, 16 heads, `n_embd=1024`, `bias=False` at the config level, weight-tied embeddings, learned absolute position embeddings, GELU MLP with 4× expansion, Flash attention, ~355M parameters. Training is fixed at 12,030 iterations, micro-batch 96, gradient accumulation 6 over a 2-GPU DDP run, AdamW (`β=(0.9, 0.95)`, weight decay 0.1, decoupled so norm gains do not decay), cosine schedule with 4% linear warmup and `min_lr = lr/10`, grad-clip 1.0, bf16 autocast, `torch.compile`. The data is FineWeb `sample-10BT` tokenized with GPT-2 BPE (~7.1B training tokens). Initialization follows the nanoGPT recipe: `N(0, 0.02)` on linears and embeddings, and `c_proj` weights scaled by `1/√(2·n_layer)`.

The loop enforces a contract: the model is built from `GPTConfig`, each `Block` receives that config, the stack runs `for block in self.transformer.h: x = block(x)` followed by a final `ln_f` before the tied LM head, and the optimizer groups parameters by dimensionality—tensors with `dim ≥ 2` decay, tensors with `dim < 2` (including all norm `weight` and `bias`) do not. Any new normalization parameter therefore inherits no-weight-decay automatically as long as it stays 1-D.

## Editable interface

Exactly two regions of `nanoGPT/custom_pretrain.py` are editable, plus a small whitelisted config slot:

1. **The `LayerNorm` class** — the normalization rule and its affine parameters.
2. **The `Block` class** — how the two sublayers are normalized, placed, and combined into the residual.
3. **`CONFIG_OVERRIDES`** — a dict allowing only `learning_rate`, `weight_decay`, `warmup_iters`, `min_lr`, `grad_clip`. No other hyperparameter, and nothing in data/tokenizer/optimizer/evaluation, may change.

The contract: `LayerNorm(ndim, bias)` must accept the same two positional arguments (`bias=config.bias`, which is `False` here) even if it ignores `bias`. `Block(config)` must expose `forward(x) -> x` of the same shape and keep the residual stream intact for the next block and for `ln_f`. Internal naming inside a block is free. Each rung replaces exactly these definitions and nothing else.

The starting point is the scaffold **default fill**: standard `LayerNorm` with bias, in a sequential Pre-LN block.

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

One seed (42). The primary metric is **FineWeb validation cross-entropy** (`val_loss`, lower is better), estimated over 200 batches at the end of training. Secondary witnesses on the same checkpoint: **WikiText-2 perplexity** and **LAMBADA perplexity** (lower is better, computed as `exp(mean chunk loss)` over non-overlapping `block_size` windows), and zero-shot downstream accuracy via lm-evaluation-harness on **ARC-Easy** and **HellaSwag** (higher is better), with PIQA and WinoGrande tracked but held out. Wall-clock (`elapsed`) is logged but is not the objective. The comparison anchor is the default fill above—standard `LayerNorm` with bias in a sequential Pre-LN block.
