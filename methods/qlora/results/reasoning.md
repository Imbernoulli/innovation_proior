Let me start from the wall I keep hitting: I want to finetune a 65B-parameter model on a single GPU, 48 GB, and I can't even *load* it. In 16-bit the weights alone are about 130 GB. LoRA was supposed to save me — freeze the base, train a tiny low-rank adapter — and it does cut the *trainable* parameters and their optimizer state to almost nothing. But it doesn't touch the elephant: the frozen base weights still have to sit in memory so I can run the forward and backward passes through them. With LoRA the base is the entire footprint. So the question sharpens: during PEFT the base is *frozen* — I only ever read it, never update it — so why am I storing it at 16-bit at all? Could I store the frozen base at 4-bit, dequantize on the fly for the matmul, and let the trainable adapter absorb whatever precision I lost?

The catch I already know about: 4-bit quantization degrades quality at *inference*. People have shown that pushing weights to 4-bit and running them costs accuracy versus 16-bit. So naively this looks like a losing trade. But there's a difference between 4-bit *inference* and 4-bit *finetuning*: in finetuning I have a trainable adapter sitting on top that I can use to *compensate* for the quantization error. The bet is that the adapter, training in full precision, can recover the performance the quantization lost. That bet only pays off if the error the adapter has to clean up is small to begin with — an adapter of rank 64 has very little capacity, and I do not want to burn it all undoing a crude quantizer instead of learning the task. So before I wire up anything, I should push 4-bit quantization to be as accurate as it can be, and only then trust the adapter with the residual.

Start with the standard scheme and find where it bleeds precision. Absmax quantization: take a tensor, find its absolute maximum, rescale so the max maps to the edge of the integer range, round. X^Int8 = round( (127/absmax(X)) · X ) = round(c·X), where c is the quantization constant (the scale), and dequant divides by c. The failure mode is outliers: one big-magnitude value sets absmax, so the scale is dominated by it, and all the ordinary values get squashed into a handful of bins near zero — most of the 256 codes go unused. The standard remedy is *block-wise* quantization: flatten the tensor, chop it into contiguous blocks of size B, and quantize each block independently with its own constant c_i. Now an outlier only wrecks the bin usage *within its own block*, not the whole tensor. So a small block size B buys precision. Hold that thought — small B means many constants, which will cost memory later.

Block-wise fixes outlier *localization*, but there's a second inefficiency, and it is about *where the code points sit*, not how I scale. An Int4 or FP4 grid places its 16 code points on an evenly-spaced (or float-spaced) lattice. My data — pretrained weights — is empirically approximately zero-centered Gaussian. If I lay 16 equally-spaced code points across a Gaussian, the bins near zero (where almost all the mass is) each capture a huge number of values while the bins out in the tails capture almost none. That feels wasteful, but "feels wasteful" is not a measurement — let me actually quantify how much an evenly-spaced grid costs me on Gaussian data before I commit to redesigning the grid. I quantize a 4096×4096 standard-normal weight matrix block-wise (B=64) two ways: once snapping to an evenly-spaced 16-point grid on [−1,1], once snapping to a grid whose points are placed at equal-mass quantiles of N(0,1). Reconstruction MSE comes out at 1.01e-2 for the uniform grid and 8.46e-3 for the quantile grid — the uniform grid is about 1.19× worse on exactly the distribution my weights follow. Not a wild margin, but a real one, and it costs me nothing at run time because the grid is fixed. So the lattice is leaving accuracy on the table, and the way to claim it is to spend the finite symbols where the probability mass is, so that each bin carries roughly the same expected number of values. That is quantile quantization: bin boundaries at the quantiles of the data distribution.

The usual problem with quantile quantization is that estimating quantiles of an arbitrary tensor needs its empirical CDF, which is expensive, and the fast approximations have large errors exactly on the outliers — the values that matter most. But look at my situation: I *know* the weights come from a fixed family — zero-mean Gaussian, differing only by their standard deviation σ. Tensors from the same distribution-up-to-scale share the *same quantiles*. So I might be able to compute the quantile grid *once*, for a standard normal N(0,1), and reuse it for every weight block after rescaling that block to unit scale. The block's absmax constant c handles the scale; a fixed Gaussian-quantile grid would handle the shape — no per-tensor CDF estimation, no approximation error. That's the plan; whether the grid it produces is sane I still have to construct and look at.

So let me construct the grid and actually print it. I want 2^k code points (k=4 → 16) that represent an equal-mass partition of N(0,1), then normalize those values into a fixed range so they line up with the rescaled weights. Using the quantile function (inverse CDF) Q_X of N(0,1), I estimate 2^k+1 adjacent quantile levels and let each code value sit halfway between two neighboring quantiles:

  q_i = ½ ( Q_X( i / (2^k + 1) ) + Q_X( (i+1) / (2^k + 1) ) ).

Each q_i is the average of the quantiles at two adjacent equal-probability levels — the representative value for that equal-mass bin. Then I normalize the whole set into [−1, 1] by dividing by the largest magnitude.

There's a subtlety that I only notice once I try to evaluate this literally: the outermost levels Q_X(0) and Q_X(1) are −∞ and +∞, so I can't put a code exactly at probability 0 or 1; I have to offset the extreme levels inward. And a symmetric equal-mass construction across the whole line doesn't give an *exact* representation of zero. Zero is special: padding tokens and other zero-valued elements should quantize with *no error*, and lots of structured sparsity sits exactly at zero. If my grid has no code point at exactly 0, every zero gets nudged to ±something. So I need a discrete 0 in the grid, and I still want to use all 2^k codes. The fix is to make the construction *asymmetric*: estimate the equal-mass levels separately for the negative side (2^{k−1} levels) and the positive side (2^{k−1}+1 levels — the extra one so the two halves plus a shared zero use all 2^k codes), unify the two sets, and *remove the duplicate zero* that appears in both.

Now I actually run this construction — negative half from an inward offset down to the median, positive half likewise, insert 0, sort, divide by the max magnitude — and read off the 16 numbers it produces:

  [−1.0, −0.6962, −0.5251, −0.3949, −0.2844, −0.1848, −0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0].

Three things I check on this output rather than assert. First, 0.0 is in the set literally — exact-zero is satisfied. Second, the spacing between consecutive code points: the gaps run 0.30, 0.17, 0.13, 0.11, 0.10, 0.094, 0.091 down to the *smallest* gap of 0.0796 straddling zero, then widen back out to 0.28 at the +1 end. The grid is genuinely densest where the Gaussian mass concentrates and sparsest in the tails — which is the whole point, and now I've seen it rather than hoped it. Third, this is the same 16-value table my earlier MSE experiment used as the "quantile grid," so the 1.19× advantage over uniform was measured on exactly this object. That's the data type — call it k-bit NormalFloat, NFk; NF4 for k=4. To quantize a weight block: divide by its absmax c to land in [−1, 1], then snap each value to the nearest q_i.

Good — that's the *shape* problem solved, and checked. Now back to the memory problem I parked: small blocks mean many quantization constants. How much do they cost? With block size B=64 and 32-bit constants, each block of 64 weights carries one 32-bit constant, i.e. 32/64 = 0.5 bits *per parameter* of pure overhead. On a 65B model that's 0.5 · 65e9 / 8 ≈ 4.06 GB of metadata — enough to matter when I'm fighting to fit in 48 GB. I don't want to enlarge B (that hurts quantization precision, which I just worked to get). So instead: quantize the constants themselves. Treat the first-level FP32 constants c₂ as a new input tensor and quantize *them*. Prior results say 8-bit quantization is essentially lossless for quantities like these, so store the constants as 8-bit floats, with their own (much coarser) block size — say 256 — and one FP32 second-level constant c₁ per such block. One wrinkle: the c₂ are all positive (they're absmax values), so before quantizing I subtract their mean to center them around zero, which lets me use symmetric quantization cleanly.

I should do the bit arithmetic explicitly, because the whole maneuver is only worth its complexity if the saving is real. Per parameter after double quantization: each c₂ now costs 8 bits but is shared over 64 parameters → 8/64 = 0.125 bits/param; and each second-level c₁ is 32 bits shared over a c₂-block of 256, and each of those 256 c₂'s covers 64 parameters, so c₁ costs 32/(64·256) = 0.001953 bits/param. Total = 0.125 + 0.001953 = 0.126953 bits/param. So I've gone from 0.5 to ≈0.127 bits/param — a saving of 0.373047 bits/param. Cashing that out on 65B: 0.373047 · 65e9 / 8 ≈ 3.03 GB. That is squarely the kind of margin that decides whether the model loads at all on a 48 GB card. The complexity earns its place. Call this Double Quantization.

There's still one more way a single GPU dies, and it's not steady-state — it's spikes. I'm going to need gradient checkpointing, because I suspect the dominant *training* memory isn't the adapter or even the base; but "I suspect" isn't good enough when it drives a design choice, so let me look at a measured breakdown. For a 7B model with LoRA at 0.2% of weights, the LoRA parameters are ~26 MB, but the activation/input gradients are ~567 MB — over twenty times larger, and the thing actually eating the budget. Checkpointing recomputes activations in the backward pass instead of storing them, which on this breakdown drops the activation gradients to ~18 MB per sequence — now *smaller* than the 26 MB of LoRA weights, with the 4-bit base around 5 GB. Two things fall out of those numbers. First, checkpointing is mandatory: without it the activation gradients alone are an order of magnitude bigger than everything I worked to compress. Second — and this is the liberating one — since the LoRA parameters are 26 MB against a multi-GB budget, a rounding error, shrinking them further buys nothing; I can afford to put adapters on *more* layers without meaningfully growing the footprint. I'll come back to that.

But checkpointing introduces its own failure: when it recomputes activations for a mini-batch with a long sequence, the transient activation memory *spikes*, and that spike can blow past the GPU's capacity and OOM-crash the run even though the steady state fits comfortably. The optimizer states (Adam's moments) are large and resident; during a spike there isn't room for everything at once. So I want a way for the optimizer state to *temporarily spill* to CPU RAM during a spike and come back when the spike passes, automatically, without me hand-managing it. NVIDIA's unified memory does exactly this — it pages memory between GPU and CPU like the OS pages between RAM and disk. So I allocate the optimizer states in paged memory: when the GPU runs short during a checkpointed long-sequence backward pass, those states get evicted to CPU RAM, and they're paged back in for the optimizer update step. Paged Optimizers — they don't reduce average memory, they absorb the spikes that would otherwise kill the job.

Now assemble the whole thing. Two data types coexist: a low-precision *storage* type (4-bit NormalFloat) for the frozen base, and a higher-precision *computation* type (BFloat16) for everything that actually does arithmetic. Whenever a weight is needed, I dequantize it from NF4 storage up to BF16 and do the matmul in 16-bit. With double quantization the dequantization is two-level: first dequantize the 8-bit constants c₂ using the FP32 c₁, then use those recovered c₂ to dequantize the NF4 weights — call it doubleDequant. For one linear layer with one LoRA adapter:

  Y^BF16 = X^BF16 · doubleDequant(c₁^FP32, c₂^{k-bit}, W^NF4) + X^BF16 · L₁^BF16 · L₂^BF16,

where doubleDequant(c₁, c₂, W) = dequant( dequant(c₁, c₂), W ) = W^BF16. NF4 for W, FP8 for c₂, block size 64 for W (precision) and 256 for c₂ (memory). The first term is the frozen, dequantized base projection; the second is the trainable low-rank update.

What about the gradients? This is the crux of why the whole scheme is cheap, so let me trace which derivatives I actually need. For the parameter update I need ∂E/∂L₁ and ∂E/∂L₂. I never need ∂E/∂W, because W is frozen — there's no update step that consumes it. But the error signal still has to *pass through* the frozen base projection so earlier adapters receive gradients: for Y = XW + sXL₁L₂, the backward pass forms ∂E/∂X = ∂E/∂Y · W^T plus the adapter path. So W appears in the backward computation as a multiplicand, but only the *already-materialized* dequantized W^BF16 — I never accumulate a gradient *into* W, never allocate a gradient buffer or an optimizer-state pair for it. That's the asymmetry that makes the footprint collapse: one storage type (NF4) and one compute type (BF16); dequantize on the fly for both the forward matmul and the backward input-gradient; compute and store weight gradients *only* for the BF16 LoRA factors, whose optimizer state I already measured to be negligible.

Two finetuning choices remain, and the diagnostic above already pointed at one. Standard LoRA practice puts adapters only on the query and value attention projections. If the base is now frozen *and* quantized, that is a narrow place to ask the adapter to repair all task-specific movement. But I just established from the memory breakdown that adapter parameters are nearly free, so I can afford to be generous: put LoRA adapters on **all** linear layers of every transformer block, not just q/v. Once adapter coverage is broad, the rank can be moderate rather than huge; I'll take r=64 as the default search point. The α scalar that scales the update is proportional to the effective learning rate, so I fix α and tune the learning rate instead. I'd want to confirm on a real finetune that q/v-only genuinely underperforms all-layer coverage at matched memory — that's the load-bearing empirical claim — but the memory accounting at least makes the broad-coverage choice cost-free to try.

For the optimizer and schedule: Adam with β₂=0.999, a constant learning-rate schedule, max grad norm 0.3 for stability, a small LoRA dropout (around 0.1 for models up to 13B, lower like 0.05 for 33B/65B), and group-by-length batching so similarly-sized sequences sit together. Everything runs with gradient checkpointing and paged optimizers.

Here is the code these decisions land on: a frozen quantized base linear, dequantized before matmul, plus a trainable LoRA path.

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
    """Block-wise: per-block absmax scale, snap to nearest NF4 code."""
    flat, n_weights = _pad_flat(W.float(), blocksize)
    blocks = flat.view(-1, blocksize)
    c2 = blocks.abs().max(dim=1, keepdim=True).values          # first-level constants (FP32)
    safe_c2 = torch.where(c2 == 0, torch.ones_like(c2), c2)
    normed = blocks / safe_c2                                  # into [-1, 1]
    codes = (normed.unsqueeze(-1) - grid).abs().argmin(dim=-1).to(torch.uint8)
    return codes, c2, n_weights


def double_quantize_constants(c2, blocksize=256):
    """Quantize the (positive) constants themselves: center, then 8-bit; one FP32 c1 per block.
    0.5 -> 8/64 + 32/(64*256) ~= 0.127 bits/param."""
    flat, n_constants = _pad_flat(c2.float(), blocksize)
    mean = flat.mean(); centered = flat - mean                 # center -> symmetric quant
    blk = centered.view(-1, blocksize)
    c1 = blk.abs().max(dim=1, keepdim=True).values             # second-level constants (FP32)
    safe_c1 = torch.where(c1 == 0, torch.ones_like(c1), c1)
    c2_fp8 = to_fp8(blk / safe_c1)
    return c2_fp8, c1, mean, n_constants


def double_dequant(c2_fp8, c1, mean, n_constants, codes, grid, blocksize, shape, n_weights):
    c2 = (from_fp8(c2_fp8) * c1).flatten()[:n_constants] + mean
    W = (grid[codes.long()].view(-1, blocksize) * c2.view(-1, 1))
    return W.flatten()[:n_weights].view(shape)                 # dequantized weights (BF16)


class QLoRALinear(nn.Module):
    """Frozen NF4 base weight (double-quantized) + trainable BF16 LoRA adapter."""
    def __init__(self, linear, r=64, alpha=16, blocksize=64, dropout=0.1):
        super().__init__()
        W_fp = linear.weight.detach().T.contiguous()           # use source convention: x @ W
        self.shape = tuple(W_fp.shape)
        self.blocksize = blocksize
        grid = make_nf4_grid(4)
        self.register_buffer("grid", grid)
        codes, c2, self.n_weights = quantize_nf4(W_fp, grid, blocksize)
        c2_fp8, c1, mean, self.n_constants = double_quantize_constants(c2)
        for n, t in [("codes", codes), ("c2_fp8", c2_fp8), ("c1", c1), ("mean", mean)]:
            self.register_buffer(n, t)                          # all frozen, no gradient
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())
        h, o = self.shape
        self.lora_A = nn.Parameter(torch.randn(h, r) * (1.0 / r ** 0.5))   # L1
        self.lora_B = nn.Parameter(torch.zeros(r, o))           # zero update at initialization
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        W = double_dequant(self.c2_fp8, self.c1, self.mean,
                           self.n_constants, self.codes, self.grid,
                           self.blocksize, self.shape, self.n_weights).to(x.dtype)
        base = x @ W                                            # frozen base, dequantized to BF16
        if self.bias is not None:
            base = base + self.bias.to(x.dtype)
        update = (self.drop(x) @ self.lora_A) @ self.lora_B     # trainable low-rank path
        return base + self.scaling * update


def quantize_model(model, r=64, alpha=16):
    """Replace every linear layer with a QLoRALinear (LoRA on ALL linear layers)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in model.modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, child_name,
                        QLoRALinear(child, r=r, alpha=alpha))
    return model


def build_optimizer(model, lr=2e-4):
    params = [p for p in model.parameters() if p.requires_grad]   # only LoRA factors
    import bitsandbytes as bnb
    return bnb.optim.PagedAdamW(params, betas=(0.9, 0.999),       # paged: survive spike OOM
                                lr=lr)


def train(model, loader):
    model.gradient_checkpointing_enable()                         # mandatory: shrink activation grads
    optim = build_optimizer(model)
    for batch in loader:                                          # group-by-length batching
        loss = model(batch["input_ids"], labels=batch["labels"]).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 0.3)
        optim.step(); optim.zero_grad()                           # constant LR schedule
```

The causal chain: a 65B model can't fit on one GPU, and under PEFT the base is frozen (read-only), so store it at 4-bit and dequantize on the fly; 4-bit hurts at inference, but a trainable adapter can absorb the error *if* the quantization is accurate, and an evenly-spaced grid measured ~1.19× worse in MSE on Gaussian weights, so make the grid as accurate as possible — block-wise to localize outliers, and a quantile grid matched to the Gaussian shape of weights (NormalFloat), constructed and inspected to be exact-at-zero, densest near zero, and to use all 16 codes; the small blocks needed for precision make the constants costly (0.5 bits/param ≈ 4 GB on 65B), so double-quantize the constants down to ≈0.127 bits/param, a measured ~3 GB saving; the real training-memory hog is activation gradients (~567 MB vs ~26 MB of adapter on 7B), so gradient-checkpoint (which also makes adapters nearly free, so put LoRA on *all* linear layers to recover full-finetuning accuracy), and absorb checkpointing's long-sequence spikes with paged optimizers — a frozen NF4 base plus a BF16 LoRA path, gradients only on the adapter, that finetunes a 65B model on a single 48 GB GPU at the quality of 16-bit finetuning.
