I propose the method QLoRA, short for Quantized Low-Rank Adaptation, as the canonical way to finetune very large language models on a single consumer-grade GPU without giving up the quality of full 16-bit finetuning. The core observation is simple but often overlooked: in parameter-efficient finetuning the base model weights are frozen, so they only need to be stored and read, never updated. That means the dominant memory cost, the frozen base sitting at 16-bit precision, is unnecessary. If we can store those weights at much lower precision and dequantize them on the fly for computation, we slash the memory footprint dramatically. The risk is that 4-bit weights degrade accuracy, especially at inference time, but during finetuning we have a trainable LoRA adapter that can learn to compensate for the quantization error. The design of QLoRA is therefore about making 4-bit storage as faithful as possible, keeping the overhead of quantization metadata tiny, and using standard parameter-efficient training tricks to close the gap to full finetuning.

The first design choice is the 4-bit data type. A standard Int4 or FP4 grid places its sixteen code points on a roughly uniform lattice. That is a poor match for pretrained neural-network weights, which empirically look like zero-centered Gaussians. Most of the probability mass sits near zero, while a few outliers live far out in the tails. A uniform grid wastes many code points on nearly empty tail regions and crowds the informative central values into a few bins. The information-theoretically optimal layout instead partitions the distribution into equal-mass bins and places one code point in each bin. For a Gaussian this means a quantile grid, denser near zero and sparser in the tails. Because the weights differ from block to block mainly by their scale, we can compute the quantile grid once for a standard normal and reuse it everywhere after rescaling each block by its own absmax scale. I call this data type NormalFloat, and specifically NF4 when k equals four bits. To build the grid I take adjacent quantile levels of the standard normal and average each neighboring pair to get the bin representative, then normalize the whole set to the range minus one to one. A subtle but important detail is that zero must be represented exactly, because padding tokens and structured sparsity rely on lossless zeros. An exactly symmetric Gaussian grid would not include zero, so I construct the type asymmetrically: I estimate quantiles separately for the negative half and the positive half, using seven levels for the negative side and eight for the positive side, then merge them and drop the duplicate zero. The resulting sixteen-value grid includes exact zero and devotes more precision where the Gaussian mass lives.

Quantization is applied block-wise with a small block size, typically sixty-four. Each block gets its own absmax scale, which localizes the damage of any outlier to that block. The forward pass dequantizes the NF4 codes back to the compute dtype, usually BFloat16, using the per-block scale and the fixed grid. The backward pass only needs gradients for the trainable LoRA factors, not for the frozen base weight, because the base is never updated. However, gradients still flow through the dequantized base projection so that earlier adapters receive error signals. This gives the storage-compute split that makes the whole scheme cheap: one storage type, NF4, and one compute type, BF16, with on-the-fly dequantization and weight gradients computed only for the adapter.

Small blocks improve quantization fidelity, but they create a metadata problem. Each block of sixty-four weights carries one thirty-two-bit scale constant, adding half a bit per parameter of overhead. For a sixty-five-billion-parameter model that overhead is several gigabytes, enough to matter when the goal is fitting into forty-eight gigabytes. The solution is double quantization: quantize the first-level constants themselves. The constants are positive, so I center them by subtracting their mean, then store the centered values in eight-bit precision with a coarser block size of two hundred fifty-six, and keep one thirty-two-bit second-level constant per coarse block. Eight-bit quantization is essentially lossless for these scale values, so the fidelity is preserved while the overhead drops from about half a bit per parameter to roughly an eighth of a bit per parameter. On a sixty-five-billion-parameter model that saves around three gigabytes, which is often the difference between fitting and overflowing memory.

Even with a tiny adapter and a compressed base, training can still run out of memory because of activation gradients. For a seven-billion-parameter model with LoRA on only a fraction of weights, the LoRA parameters themselves are tiny, on the order of tens of megabytes, but activation gradients can be hundreds of megabytes or more. Gradient checkpointing is therefore mandatory: it recomputes activations during the backward pass instead of storing them, cutting the activation gradient footprint to a few megabytes per sequence. Once checkpointing is in place, the LoRA parameters become a rounding error in the budget, which suggests a better placement strategy than the usual query-and-value-only adapters. Because the base is now both frozen and quantized, restricting adaptation to q and v projections gives the adapter very little room to repair the model. Since memory can afford it, QLoRA puts LoRA adapters on all linear layers of every transformer block. With broad adapter coverage, a moderate rank such as sixty-four is sufficient, and the scaling factor alpha can be kept proportional to the learning rate rather than tuned independently.

Gradient checkpointing introduces a different hazard. On long sequences, recomputing activations creates transient memory spikes that can exceed the GPU capacity and crash the run even when the steady-state footprint fits comfortably. The optimizer states are large and resident, so they are natural candidates to move out of the way during a spike. NVIDIA unified memory, also called paged memory, lets GPU memory spill to CPU RAM automatically and page back when needed. By allocating the optimizer states in paged memory, the transient spike is absorbed without manual management, and the optimizer update step still sees the full states when the spike has passed. This is not a reduction in average memory but a way to survive the worst-case moments of training.

Putting the pieces together, QLoRA defines each linear layer as the sum of a frozen quantized base projection and a trainable low-rank update. The base projection uses double dequantization: first recover the thirty-two-bit block scales from their eight-bit storage using the second-level constants, then recover the weights from NF4 using those scales and the fixed Gaussian grid. The low-rank update is computed in BFloat16, with dropout for regularization and a scalar that scales the update into the full-precision path. Training uses Adam with beta two set to 0.999, a constant learning rate, max gradient norm 0.3, group-by-length batching so similar sequence lengths are processed together, gradient checkpointing enabled throughout, and paged optimizer states to handle spikes. For models up to thirteen billion parameters a LoRA dropout around 0.1 works well; for thirty-three-billion and sixty-five-billion models I would reduce it toward 0.05.

The result is a finetuning pipeline that preserves the accuracy of full 16-bit finetuning while loading the base model into roughly a quarter of the memory. The frozen base sits in NF4 with double-quantized constants, the trainable adapter adds only a small BFloat16 footprint, and the training machinery is arranged to avoid the two remaining failure modes: activation memory via gradient checkpointing and transient spikes via paged optimizers. This is the canonical QLoRA recipe.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_nf4_grid(k=4):
    """Normalized asymmetric N(0,1) 4-bit quantile table with exact zero."""
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
    """Block-wise NF4 quantization: per-block absmax scale, nearest grid code."""
    flat, n_weights = _pad_flat(W.float(), blocksize)
    blocks = flat.view(-1, blocksize)
    c2 = blocks.abs().max(dim=1, keepdim=True).values
    safe_c2 = torch.where(c2 == 0, torch.ones_like(c2), c2)
    normed = blocks / safe_c2
    codes = (normed.unsqueeze(-1) - grid).abs().argmin(dim=-1).to(torch.uint8)
    return codes, c2, n_weights


def double_quantize_constants(c2, blocksize=256):
    """Quantize the positive first-level constants to 8-bit with second-level FP32 scales."""
    flat, n_constants = _pad_flat(c2.float(), blocksize)
    mean = flat.mean()
    centered = flat - mean
    blk = centered.view(-1, blocksize)
    c1 = blk.abs().max(dim=1, keepdim=True).values
    safe_c1 = torch.where(c1 == 0, torch.ones_like(c1), c1)
    c2_fp8 = to_fp8(blk / safe_c1)
    return c2_fp8, c1, mean, n_constants


def double_dequant(c2_fp8, c1, mean, n_constants, codes, grid, blocksize, shape, n_weights):
    """Recover the original weights from double-quantized NF4 storage."""
    c2 = (from_fp8(c2_fp8) * c1).flatten()[:n_constants] + mean
    W = (grid[codes.long()].view(-1, blocksize) * c2.view(-1, 1))
    return W.flatten()[:n_weights].view(shape)


class QLoRALinear(nn.Module):
    """Frozen NF4 base weight + trainable BF16 LoRA adapter."""
    def __init__(self, linear, r=64, alpha=16, blocksize=64, dropout=0.1):
        super().__init__()
        W_fp = linear.weight.detach().T.contiguous()
        self.shape = tuple(W_fp.shape)
        self.blocksize = blocksize
        grid = make_nf4_grid(4)
        self.register_buffer("grid", grid)
        codes, c2, self.n_weights = quantize_nf4(W_fp, grid, blocksize)
        c2_fp8, c1, mean, self.n_constants = double_quantize_constants(c2)
        for n, t in [("codes", codes), ("c2_fp8", c2_fp8), ("c1", c1), ("mean", mean)]:
            self.register_buffer(n, t)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())
        h, o = self.shape
        self.lora_A = nn.Parameter(torch.randn(h, r) * (1.0 / r ** 0.5))
        self.lora_B = nn.Parameter(torch.zeros(r, o))
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        W = double_dequant(self.c2_fp8, self.c1, self.mean,
                           self.n_constants, self.codes, self.grid,
                           self.blocksize, self.shape, self.n_weights).to(x.dtype)
        base = x @ W
        if self.bias is not None:
            base = base + self.bias.to(x.dtype)
        update = (self.drop(x) @ self.lora_A) @ self.lora_B
        return base + self.scaling * update


def quantize_model(model, r=64, alpha=16):
    """Replace every Linear with QLoRALinear and freeze the base."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in model.modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, child_name, QLoRALinear(child, r=r, alpha=alpha))
    return model


def build_optimizer(model, lr=2e-4):
    params = [p for p in model.parameters() if p.requires_grad]
    import bitsandbytes as bnb
    return bnb.optim.PagedAdamW(params, betas=(0.9, 0.999), lr=lr)


def train(model, loader):
    model.gradient_checkpointing_enable()
    optim = build_optimizer(model)
    for batch in loader:
        loss = model(batch["input_ids"], labels=batch["labels"]).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 0.3)
        optim.step()
        optim.zero_grad()
```
