## Research question

In a GPT-style language model, every transformer block alternates two sublayers: multi-head self-attention, which mixes information *across* positions, and a position-wise feed-forward network (the MLP), which transforms each token's hidden vector *on its own*. The MLP is where most of the model's parameters and a large share of its compute live, yet it has the simplest possible form — one linear up-projection to a wide hidden dimension, one pointwise nonlinearity, one linear down-projection. The single thing being designed here is **the MLP sublayer**: can a drop-in replacement for the default `Linear(d,4d) → GELU → Linear(4d,d)` lower held-out validation loss at **matched parameters and FLOPs**, changing only how the per-position hidden representation is formed — and nothing about attention, normalization, the data, the optimizer, or evaluation?

## Prior art / Background / Baselines

- **ReLU FFN.** `max(0, xW1)W2`, two matrices, `d_ff ≈ 4d`. Pass the input through one linear projection and a hard on/off threshold.
- **GELU.** `GELU(z) = z·Φ(z)`, Φ the standard-normal CDF — a smooth, stochastic-step relaxation of ReLU. This is the **default** activation in the scaffold's MLP.
- **Swish / SiLU.** `Swish_β(z) = z·σ(βz)`, found by an automated activation search whose winners all had the form `z · g(z)`.
- **Multiplicative / gating models.** A separate line in the literature couples two learned linear views of the input multiplicatively (e.g. log-bilinear language models, gated convolutional language models).

## Fixed substrate / Code framework

A nanoGPT-style GPT-2 Medium pretraining loop is frozen and must not be touched: GPT-2 Medium (24 layers, 16 heads, `n_embd=1024`, ~355M params), pre-norm `LayerNorm` (bias-free), causal self-attention (flash SDPA), tied input/output embeddings, residual-scaled init (`c_proj` weights initialized `N(0, 0.02/√(2·n_layer))`), AdamW (`β=(0.9,0.95)`, `weight_decay=0.1`, `grad_clip=1.0`), a warmup+cosine schedule (`warmup_iters = 0.04·max_iters`, decay to `lr/10`), `torch.compile`, and 2-GPU DDP. Data is FineWeb `sample-10BT` (GPT-2 tokenizer, ~7.1B training tokens), trained 12,030 iterations at micro-batch 96 with gradient accumulation 6. The transformer `Block` is fixed as `x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))`. The MLP receives `(B, T, n_embd)` and must return the same shape; `nn`, `F` (`torch.nn.functional`), and `math` are in scope.

## Editable interface

Exactly two regions of `nanoGPT/custom_pretrain.py` are editable, and every method on the ladder is a fill of the same contract:

1. The **`MLP` class** (the only architectural slot): it must accept `(B, T, n_embd)` and return the same shape, depend on nothing outside the FFN, and — when its shape differs from the default two-matrix `4d` FFN — keep parameters and FLOPs matched by re-sizing the hidden width accordingly (rounded to a multiple of 64 for matmul-friendly shapes).
2. A **`CONFIG_OVERRIDES` dict** that may override training hyperparameters from a fixed whitelist (`learning_rate`, `weight_decay`, `warmup_iters`, `min_lr`, `grad_clip`). The baselines leave it empty; they change only the MLP.

The starting point is the scaffold default — the standard two-layer GELU MLP at 4× expansion. Each method replaces exactly the `MLP` class (and, if it chooses, `CONFIG_OVERRIDES`) and nothing else.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py — default fill (GELU MLP, 4x expansion)
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):                  # x: (B, T, n_embd) -> (B, T, n_embd)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ... later in the training setup, the (default-empty) hyperparameter override hook:
# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
CONFIG_OVERRIDES = {}
```

## Evaluation settings

One seed (42). Primary metric is **FineWeb validation cross-entropy** (`val_loss`, lower is better), measured at the end of the fixed 12,030-iteration budget. Secondary metrics, all from the fixed eval scripts, are word-level perplexity on **WikiText-2** and **LAMBADA** (lower is better) and zero-shot downstream accuracy on **ARC-Easy**, **HellaSwag**, **PIQA**, and **WinoGrande** (higher is better); `elapsed` is wall-clock seconds. Because the substrate is frozen and the budget fixed, any change in `val_loss` is attributable to the *form* of the MLP alone.
