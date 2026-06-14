**Problem.** Binary GPT-2 trained but landed at val_loss 2.7352 — a *capacity* failure, not an
optimization one. The two-valued set has no off-state, so every near-zero weight is forced to `±β` with
an essentially random sign, injecting noise into every projection. Buy back the off-switch at minimal
bit cost.

**Key idea.** Add a single zero level: quantize the latent weight onto `{−1, 0, +1}` (ternary,
`log₂3 ≈ 1.58` bits) with the *same* absmean scale `γ = mean(|W|)`. Normalize `w_n = W/γ`, clamp to
`[−1, 1]`, round to the nearest of `{−1, 0, +1}`; STE carries the gradient through clamp+round
(`(round − w_q).detach() + w_q`). The threshold this induces is `|W| < γ/2` → 0, so the genuinely weak
connections switch off (a clean 0 instead of binary's random `±β`), and the term drops out of the
matmul entirely — explicit feature filtering. Activations are unchanged from rung 1: 8-bit per-tensor
absmax, `Q_b = 127`, STE.

**Why ternary, symmetric.** The zero is the one thing `{−1,+1}` provably cannot express; `{0,1}` breaks
the balanced signed matmul; richer sets cost bits and are deferred to the next rung. This is the *task's*
fill, so it **clamps-then-rounds** (folds saturation into a continuous pre-round value; same ternary map
as round-then-clip), uses the **same per-tensor activation absmax** as binary (ladder hygiene: only the
weight grid changes), and adds **no** RMSNorm/SubLN — the block's pre-projection `LayerNorm` already
holds `E[x̃²] ≈ 1`, so `Var(y) = n·γ²·E[x̃²]` stays at the float scale. Recipe inherited
(`CONFIG_OVERRIDES = {}`, peak LR `6e-4`).

**What to watch.** I expect a *modest* drop below 2.7352 (WikiText-2 → high 70s, LAMBADA → high 100s),
downstream roughly flat near binary's 47.0 / 28.7 — the zero helps the loss but does not add magnitude
resolution. If the gain is small, the diagnosis is that the missing thing was resolution, not an
off-switch, pointing to a genuine 2-bit grid.

**Hyperparameters.** Weight grid `{−1, 0, +1}`; scale `mean(|W|)` (clamped `min=1e-12`); zero-threshold
`mean(|W|)/2`; activations 8-bit symmetric absmax, `Q_b = 127`, clip `[−127, 127]`; STE via detached
difference; `std=0.02` init; all else inherited.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 38-115) -- step 2: ternary {-1, 0, +1} (1.58-bit)
def weight_quant(weight):
    """Ternary quantization: {-1, 0, +1} via absmean with STE.

    Forward: normalize by absmean, round-then-clip to {-1, 0, +1}
    Backward: STE (gradient passes through rounding as identity)
    """
    scale = weight.detach().abs().mean().clamp(min=1e-12)
    w_normed = weight / scale
    # STE round: (round(x) - x).detach() + x
    w_q = w_normed.clamp(-1, 1)
    w_q = (w_q.round() - w_q).detach() + w_q
    return w_q, scale


def activation_quant(x):
    """Absmax 8-bit activation quantization with STE.

    Quantizes activations to 127 levels (int8 range) using per-tensor
    absmax scaling, following the BitNet b1.58 paper.
    """
    Qb = 127  # int8 range
    scale = x.detach().abs().max().clamp(min=1e-12)
    x_normed = x / scale
    x_q = (x_normed * Qb).round().clamp(-Qb, Qb)
    # STE: forward uses quantized, backward passes through
    x_q = (x_q - x_normed * Qb).detach() + x_normed * Qb
    return x_q, scale / Qb


class BitLinear(nn.Module):
    """BitNet b1.58 linear layer with ternary {-1, 0, +1} weights.

    During both training and eval: weights are ternarized via absmean
    + round-clip, activations are quantized to int8 range. Output is
    rescaled by weight_scale * activation_scale.
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
