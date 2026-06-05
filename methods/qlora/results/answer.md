# QLoRA: Efficient Finetuning of Quantized LLMs

## Problem

Finetune a model as large as 65B parameters on a single 48 GB GPU while fully preserving 16-bit
finetuning quality. Even LoRA (frozen base + small trainable adapter) must hold the frozen base in
memory at 16-bit, so the base dominates. The base is frozen during PEFT (read-only), so it can be
*stored* at 4-bit — but 4-bit quantization degrades inference quality, and a small block size (needed
for precision) makes quantization metadata expensive, and gradient-checkpointing spikes OOM single
GPUs.

## Key idea

Store the frozen base in a 4-bit storage type and dequantize to BF16 on the fly for compute; let a
full-precision LoRA adapter compensate for the quantization error. Three innovations make it
lossless and fit it in memory:

- **4-bit NormalFloat (NF4).** Weights are ~zero-centered Gaussian, so use an *information-
  theoretically optimal* quantile grid (equal mass per bin) computed once for N(0,1). The 2^k code
  points are the equal-mass bin representatives:

  q_i = ½ ( Q_X( i/(2^k+1) ) + Q_X( (i+1)/(2^k+1) ) ),   Q_X = quantile function of N(0,1),

  normalized to [−1, 1]. Built asymmetrically (2^{k-1} negative, 2^{k-1}+1 positive levels, unify and
  drop the duplicate zero) so it has an *exact* 0 (lossless for padding/zeros) and uses all 2^k codes.
  Quantization is block-wise (block size 64): each block gets its own absmax scale, localizing
  outliers. NF4 beats Int4/FP4 because its grid matches the Gaussian density.

- **Double Quantization.** Small blocks → many FP32 constants (32/64 = 0.5 bits/param overhead).
  Quantize the constants themselves: center the (positive) first-level constants c₂, store them as
  FP8 with block size 256, with one FP32 second-level constant c₁ per block. Overhead drops to
  8/64 + 32/(64·256) ≈ 0.127 bits/param — saving ≈0.373 bits/param (~3 GB for 65B).

- **Paged Optimizers.** Allocate optimizer states in NVIDIA unified (paged) memory so they spill to
  CPU RAM during gradient-checkpointing activation spikes on long sequences and page back for the
  update — preventing OOM crashes without changing average memory.

Full layer (one storage type NF4, one compute type BF16; weight gradients only for the BF16 LoRA
factors, never for W):

  Y^BF16 = X^BF16 · doubleDequant(c₁^FP32, c₂^{k-bit}, W^NF4) + X^BF16 · L₁^BF16 · L₂^BF16,
  doubleDequant(c₁, c₂, W) = dequant( dequant(c₁, c₂), W ).

Finetuning choices: put LoRA on **all** linear layers (critical to match full finetuning for large
models; q/v-only is insufficient); r (e.g. 64) barely matters; α fixed and kept proportional to lr.
Adam β₂=0.999, constant LR, max grad norm 0.3, LoRA dropout 0.1 (≤13B) / 0.05 (33B/65B),
group-by-length batching, gradient checkpointing throughout.

## Code

```python
import torch
import torch.nn as nn


def make_nf4_grid(k=4):
    """Equal-mass bin representatives of N(0,1), asymmetric with exact 0, normalized to [-1,1]."""
    from torch.distributions import Normal
    Q = Normal(0.0, 1.0).icdf
    offset = 0.5 * (1 / 32 + 1 / 30)
    neg = [float(Q(torch.tensor(v))) for v in torch.linspace(offset, 0.5, 2 ** (k - 1))]
    pos = [float(Q(torch.tensor(v))) for v in torch.linspace(1 - offset, 0.5, 2 ** (k - 1) + 1)]
    g = torch.tensor(sorted(set(neg + [0.0] + pos)))
    return g / g.abs().max()


def quantize_nf4(W, grid, blocksize=64):
    blocks = W.flatten().view(-1, blocksize)
    c2 = blocks.abs().max(dim=1, keepdim=True).values                 # first-level constants
    codes = ((blocks / c2).unsqueeze(-1) - grid).abs().argmin(-1).to(torch.uint8)
    return codes, c2


def double_quantize_constants(c2, blocksize=256):
    flat = c2.flatten(); mean = flat.mean()
    blk = (flat - mean).view(-1, blocksize)                            # center -> symmetric
    c1 = blk.abs().max(dim=1, keepdim=True).values                     # second-level constants
    return to_fp8(blk / c1), c1, mean


def double_dequant(c2_fp8, c1, mean, codes, grid, blocksize, shape):
    c2 = (from_fp8(c2_fp8) * c1).flatten() + mean
    return (grid[codes.long()].view(-1, blocksize) * c2.view(-1, 1)).view(shape)


class QLoRALinear(nn.Module):
    def __init__(self, W_fp, r=64, alpha=16, blocksize=64, dropout=0.1):
        super().__init__()
        self.shape, self.blocksize = tuple(W_fp.shape), blocksize
        grid = make_nf4_grid(4); self.register_buffer("grid", grid)
        codes, c2 = quantize_nf4(W_fp, grid, blocksize)
        c2_fp8, c1, mean = double_quantize_constants(c2)
        for n, t in [("codes", codes), ("c2_fp8", c2_fp8), ("c1", c1), ("mean", mean)]:
            self.register_buffer(n, t)                                 # frozen, no gradient
        h, o = self.shape
        self.lora_A = nn.Parameter(torch.randn(h, r) / r ** 0.5)
        self.lora_B = nn.Parameter(torch.zeros(r, o))                  # B=0 => adapter starts off
        self.scaling, self.drop = alpha / r, nn.Dropout(dropout)

    def forward(self, x):
        W = double_dequant(self.c2_fp8, self.c1, self.mean,
                           self.codes, self.grid, self.blocksize, self.shape).to(x.dtype)
        return x @ W + self.scaling * ((self.drop(x) @ self.lora_A) @ self.lora_B)


def quantize_model(model, r=64, alpha=16):
    for module in model.modules():
        for cn, child in list(module.named_children()):
            if isinstance(child, nn.Linear):                           # LoRA on ALL linear layers
                setattr(module, cn, QLoRALinear(child.weight.data, r=r, alpha=alpha))
    return model


def train(model, loader):
    import bitsandbytes as bnb
    model.gradient_checkpointing_enable()
    params = [p for p in model.parameters() if p.requires_grad]        # only LoRA factors
    optim = bnb.optim.PagedAdamW(params, betas=(0.9, 0.999), lr=2e-4, max_grad_norm=0.3)
    for batch in loader:                                               # group-by-length, constant LR
        loss = model(batch["input_ids"], labels=batch["labels"]).loss
        loss.backward(); optim.step(); optim.zero_grad()
```
