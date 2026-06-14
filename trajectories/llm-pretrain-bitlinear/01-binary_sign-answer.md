**Problem.** Pretrain GPT-2 Medium with every projection weight forced to one bit `{−1,+1}` on every
forward pass, from scratch. The crudest genuinely-discrete fill of `BitLinear` — the true floor for the
low-bit regime (the scaffold's pass-through default is just FP GPT-2 and does not count).

**Key idea.** Binarize the latent weight with the sign function and rescale by the per-tensor absmean:
minimizing `‖W − α·sign(W)‖²` gives both pieces — the optimal direction `sign(W)` and the optimal
scale `α = mean(|W|)`. `sign` has zero gradient a.e., so train through it with the straight-through
estimator (`(sign(W) − W).detach() + W` — forward `±1`, backward identity). Keep `self.weight` as the
float **latent** weight the optimizer accumulates into; binarize on the fly each forward pass. The
other operand is quantized too, but asymmetrically: weights are flat and binarize cleanly, activations
carry persistent outlier channels, so activations stay at **8-bit absmax** (`Q_b = 127`), also with STE.

**Why (and what this fill drops).** This rung is the *task's* binary fill, not the textbook one. It
uses **uncentered** `sign(W)` (no `sign(W − mean(W))` capacity refinement), a **per-tensor** activation
absmax (not per-token), and **no** SubLN inside `BitLinear` — the latter is correct because each block
already applies a `LayerNorm` before its projections, so `E[x̃²] ≈ 1` is supplied by the substrate and
`Var(y) = n·mean(|W|)²·E[x̃²]` stays at the float layer's scale. The inherited recipe is left untouched
(`CONFIG_OVERRIDES = {}`, peak LR `6e-4`), so the measurement isolates the cost of binarization alone.

**What to watch.** One bit per weight has no off-state — a near-zero weight is still forced to `±β`. I
expect binary GPT-2 to train but land well above the float loss, with the next rung's fix obvious: add
a third value `0` so weak connections can switch off, reusing the same absmean scale.

**Hyperparameters.** Weight grid `{−1,+1}`; weight scale `mean(|W|)`; activations 8-bit symmetric
absmax, `Q_b = 127`, clip `[−127, 127]`; STE via detached difference; `std=0.02` init; everything else
inherited (AdamW `β=(0.9,0.95)`, wd `0.1`, cosine LR peak `6e-4`, 4% warmup, 13,535 iters, bf16,
`torch.compile`).

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 38-115) -- step 1: binary sign {-1, +1}
def weight_quant(weight):
    """Binary quantization: sign(W) * mean(|W|) with STE.

    Forward: w_q = sign(W), scale = mean(|W|)
    Backward: STE (gradient passes through sign as identity)
    """
    scale = weight.detach().abs().mean()
    # STE: forward uses sign, backward treats sign as identity
    w_q = (weight.sign() - weight).detach() + weight
    return w_q, scale


def activation_quant(x):
    """Absmax 8-bit activation quantization with STE.

    Quantizes activations to 127 levels (int8 range) using per-tensor
    absmax scaling, following the original BitNet paper.
    """
    Qb = 127  # int8 range
    scale = x.detach().abs().max().clamp(min=1e-12)
    x_normed = x / scale
    x_q = (x_normed * Qb).round().clamp(-Qb, Qb)
    # STE: forward uses quantized, backward passes through
    x_q = (x_q - x_normed * Qb).detach() + x_normed * Qb
    return x_q, scale / Qb


class BitLinear(nn.Module):
    """BitNet linear layer with binary {-1, +1} weights.

    During both training and eval: weights are binarized via sign function,
    activations are quantized to int8 range. Output is rescaled by
    weight_scale * activation_scale.
    """
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
        out = F.linear(x_q, w_q, None)
        out = out * (w_scale * x_scale)
        if self.bias is not None:
            out = out + self.bias
        return out
```
