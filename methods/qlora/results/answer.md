# QLoRA

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
high-fidelity and fit it in memory:

- **4-bit NormalFloat (NF4).** Weights are ~zero-centered Gaussian, so use an *information-
  theoretically optimal* quantile grid (equal mass per bin) computed once for N(0,1). The 2^k code
  points are the equal-mass bin representatives:

  q_i = ½ ( Q_X( i/(2^k+1) ) + Q_X( (i+1)/(2^k+1) ) ),   Q_X = quantile function of N(0,1),

  normalized to [−1, 1]. Built asymmetrically (2^{k-1} negative, 2^{k-1}+1 positive levels, unify and
  drop the duplicate zero) so it has an *exact* 0 (lossless for padding/zeros) and uses all 2^k codes.
  Quantization is block-wise (block size 64): each block gets its own absmax scale, localizing
  outliers. NF4 is better matched to these weights than an evenly spaced Int4/FP4 grid.

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
import torch.nn.functional as F


def make_nf4_grid(k=4):
    """The normalized asymmetric N(0,1) quantile table, with an exact zero."""
    if k != 4:
        raise ValueError("Only the NF4 table is specified here.")
    return torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367,
        -0.39491748809814453, -0.28444138169288635,
        -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725,
        0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0,
    ], dtype=torch.float32)


def _pad_flat(x, blocksize):
    flat = x.flatten()
    n = flat.numel()
    pad = (-n) % blocksize
    if pad:
        flat = F.pad(flat, (0, pad))
    return flat, n


def to_fp8(x):
    return x.to(torch.float8_e4m3fn)


def from_fp8(x):
    return x.to(torch.float32)


def quantize_nf4(W, grid, blocksize=64):
    flat, n_weights = _pad_flat(W.float(), blocksize)
    blocks = flat.view(-1, blocksize)
    c2 = blocks.abs().max(dim=1, keepdim=True).values                 # first-level constants
    safe_c2 = torch.where(c2 == 0, torch.ones_like(c2), c2)
    codes = ((blocks / safe_c2).unsqueeze(-1) - grid).abs().argmin(-1).to(torch.uint8)
    return codes, c2, n_weights


def double_quantize_constants(c2, blocksize=256):
    flat, n_constants = _pad_flat(c2.float(), blocksize)
    mean = flat.mean()
    blk = (flat - mean).view(-1, blocksize)                            # center -> symmetric
    c1 = blk.abs().max(dim=1, keepdim=True).values                     # second-level constants
    safe_c1 = torch.where(c1 == 0, torch.ones_like(c1), c1)
    return to_fp8(blk / safe_c1), c1, mean, n_constants


def double_dequant(c2_fp8, c1, mean, n_constants, codes, grid, blocksize, shape, n_weights):
    c2 = (from_fp8(c2_fp8) * c1).flatten()[:n_constants] + mean
    W = grid[codes.long()].view(-1, blocksize) * c2.view(-1, 1)
    return W.flatten()[:n_weights].view(shape)


class QLoRALinear(nn.Module):
    def __init__(self, linear, r=64, alpha=16, blocksize=64, dropout=0.1):
        super().__init__()
        W_fp = linear.weight.detach().T.contiguous()                   # source convention: x @ W
        self.shape, self.blocksize = tuple(W_fp.shape), blocksize
        grid = make_nf4_grid(4); self.register_buffer("grid", grid)
        codes, c2, self.n_weights = quantize_nf4(W_fp, grid, blocksize)
        c2_fp8, c1, mean, self.n_constants = double_quantize_constants(c2)
        for n, t in [("codes", codes), ("c2_fp8", c2_fp8), ("c1", c1), ("mean", mean)]:
            self.register_buffer(n, t)                                 # frozen, no gradient
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())
        h, o = self.shape
        self.lora_A = nn.Parameter(torch.randn(h, r) / r ** 0.5)
        self.lora_B = nn.Parameter(torch.zeros(r, o))                  # zero update at init
        self.scaling, self.drop = alpha / r, nn.Dropout(dropout)

    def forward(self, x):
        W = double_dequant(self.c2_fp8, self.c1, self.mean,
                           self.n_constants, self.codes, self.grid,
                           self.blocksize, self.shape, self.n_weights).to(x.dtype)
        base = x @ W
        if self.bias is not None:
            base = base + self.bias.to(x.dtype)
        return base + self.scaling * ((self.drop(x) @ self.lora_A) @ self.lora_B)


def quantize_model(model, r=64, alpha=16):
    for p in model.parameters():
        p.requires_grad_(False)
    for module in model.modules():
        for cn, child in list(module.named_children()):
            if isinstance(child, nn.Linear):                           # LoRA on ALL linear layers
                setattr(module, cn, QLoRALinear(child, r=r, alpha=alpha))
    return model


def train(model, loader):
    import bitsandbytes as bnb
    model.gradient_checkpointing_enable()
    params = [p for p in model.parameters() if p.requires_grad]        # only LoRA factors
    optim = bnb.optim.PagedAdamW(params, betas=(0.9, 0.999), lr=2e-4)
    for batch in loader:                                               # group-by-length, constant LR
        loss = model(batch["input_ids"], labels=batch["labels"]).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.3)
        optim.step(); optim.zero_grad()
```
