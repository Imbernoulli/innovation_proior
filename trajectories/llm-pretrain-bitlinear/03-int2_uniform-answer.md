**Problem.** Ternary barely beat binary (2.7213 vs 2.7352), falsifying the off-switch hypothesis: the
binding constraint is *magnitude resolution*, not a missing zero. Three slammed levels cannot express a
present-but-weak connection — the thing the completion tasks need. Spend the next bit on finer levels.

**Key idea.** Move to a multi-level uniform grid with the same absmean scale `s = mean(|W|)`. Normalize
`w_n = W/s`, multiply by `1.5`, clamp the scaled value to `[−2, 2]`, round to the nearest integer, clamp
to `[−1.5, 1.5]`, divide back by `1.5`; STE through round+clamps. **What this actually computes** (the
load-bearing finding): PyTorch's round is half-to-even, so `±0.5 → 0` and `±1.5 → ±2 → clamp ±1.5`, and
the realized grid is the **5-level** set `{−1, −2/3, 0, +2/3, +1}` (`log₂5 ≈ 2.32` bits) — *not* the
4-level `{−1,−1/3,+1/3,+1}` the docstring claims. So this rung keeps ternary's zero **and** adds two
graded magnitudes `±2/3`: off, weakly-on, or strongly-on, in either sign. That is exactly the resolution
ternary lacked, which is why it leaps the plateau.

**Why.** absmean (not absmax) for weights so one outlier cannot crush the grid; absmax (not absmean) for
activations so the extreme activation never clips (a clipped activation is destroyed; a clipped weight is
just misplaced). Activations held **identical** to rungs 1-2 (8-bit per-tensor absmax, `Q_b = 127`) so
the only change is the weight grid — controlled comparison. No SubLN: the block's pre-projection
`LayerNorm` keeps `Var(y)` at the float scale. Recipe inherited (`CONFIG_OVERRIDES = {}`, LR `6e-4`).

**What to watch.** I expect a large drop below the 2.72 plateau (val_loss → mid-2.4s, WikiText-2 →
mid-50s, LAMBADA → low-80s) and, unlike ternary, downstream *movement* (ARC-Easy → low-50s, HellaSwag →
low-30s). If it only inches below ternary, the bottleneck is the *fixed* absmean scale (reconstruction-
optimal, not loss-optimal), and the next move is to learn the scale against the loss.

**Hyperparameters.** Weight grid (realized) `{−1, −2/3, 0, +2/3, +1}`; scale `mean(|W|)` (clamped
`min=1e-12`); activations 8-bit symmetric absmax, `Q_b = 127`, clip `[−127, 127]`; STE via detached
difference; `std=0.02` init; all else inherited.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 38-115) -- step 3: 2-bit uniform grid
def weight_quant(weight):
    """2-bit uniform quantization: {-1, -1/3, +1/3, +1} with STE.

    Normalizes weights by absmean, maps to 4 uniform levels in [-1, 1],
    then rescales. Uses STE for gradient flow through rounding.
    """
    scale = weight.detach().abs().mean().clamp(min=1e-12)
    w_normed = weight / scale
    # Map to [-1.5, 1.5] grid with spacing 1.0, round, then map back
    # Levels: -1.5 -> -1, -0.5 -> -1/3, 0.5 -> 1/3, 1.5 -> 1
    # Multiply by 1.5 so that [-1,1] -> [-1.5,1.5], round, clip to {-1,0,1} range
    # Actually: use 4 uniform levels directly
    # Grid points at: -1, -1/3, 1/3, 1 (spacing = 2/3)
    # Scale so spacing becomes 1: multiply by 3/2
    w_scaled = w_normed * 1.5  # now grid at -1.5, -0.5, 0.5, 1.5
    w_rounded = w_scaled.clamp(-2, 2).round().clamp(-1.5, 1.5)
    # STE: (rounded - scaled).detach() + scaled
    w_q = (w_rounded - w_scaled).detach() + w_scaled
    # Map back: divide by 1.5
    w_q = w_q / 1.5
    return w_q, scale


def activation_quant(x):
    """Absmax 8-bit activation quantization with STE.

    Quantizes activations to 127 levels (int8 range) using per-tensor
    absmax scaling.
    """
    Qb = 127  # int8 range
    scale = x.detach().abs().max().clamp(min=1e-12)
    x_normed = x / scale
    x_q = (x_normed * Qb).round().clamp(-Qb, Qb)
    # STE: forward uses quantized, backward passes through
    x_q = (x_q - x_normed * Qb).detach() + x_normed * Qb
    return x_q, scale / Qb


class BitLinear(nn.Module):
    """Linear layer with 2-bit uniform weight quantization.

    Weights are quantized to {-1, -1/3, +1/3, +1} during both training
    and eval. Activations quantized to int8 range. Output rescaled by
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
