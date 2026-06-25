Ternary barely moved the needle, and that is the most informative result so far. Validation loss fell from binary's 2.7352 to 2.7213 — a drop of 0.014, about half a percent — WikiText-2 from 81.3 to 77.9, LAMBADA essentially flat at 109.8, and downstream ARC-Easy and HellaSwag if anything a hair worse (46.7 and 28.4, both inside noise). So the explicit zero bought back almost nothing, which falsifies the hypothesis I went into rung 2 with: the off-switch alone is worth half a percent of loss, not the several percent separating these models from a usable one. The re-diagnosis is sharp. Ternary can say a weight is $0$, $+\beta$, or $-\beta$, but it *cannot* say a weight is small-but-not-zero — there is no $\pm\beta/3$-flavored level — so every surviving connection is slammed to full magnitude $\pm\beta$, and the only per-weight choice the layer has is "off, or full-strength in one of two directions." The zero removed random-sign noise on the truly-dead weights, which is why the loss improved at all, but the *resolution* problem is untouched, and that is exactly what the completion tasks probe: HellaSwag and ARC-Easy ask whether the model learned finely-weighted feature combinations, and a three-level weight cannot encode a finely-weighted anything. So the next bit should buy *magnitude resolution*, not another structural state.

I propose a multi-level uniform-grid `BitLinear` keeping the absmean scale $s = \mathrm{mean}(|W|)$. The natural next step is two genuine bits — four levels — but here I must read the task's actual fill rather than the textbook 2-bit grid, because they differ, and the difference is the whole reason this rung leaps rather than crawls. The clean textbook grid is the symmetric uniform set $\{-1, -1/3, +1/3, +1\}$, four levels at exactly $\log_2 4 = 2$ bits, with no exact zero. The fill in front of me does something subtly different, and I worked out what it actually computes rather than trust its docstring. It normalizes by the absmean, multiplies by $1.5$ so the grid lands at $\{-1.5, -0.5, +0.5, +1.5\}$ in scaled space, clamps the *scaled* value to $[-2, 2]$, rounds to the nearest integer, clamps the result to $[-1.5, 1.5]$, then divides back by $1.5$. Trace where the levels land: round sends a scaled value to one of $\{-2, -1, 0, 1, 2\}$; the outer clamp pins $\pm 2$ back to $\pm 1.5$; and crucially, PyTorch's round is half-to-even, so $\pm 0.5$ rounds to $0$ (zero is even) while $\pm 1.5$ rounds to $\pm 2$ (two is even). Dividing the surviving integers by $1.5$, the realized grid is

$$\{-1,\ -\tfrac{2}{3},\ 0,\ +\tfrac{2}{3},\ +1\}$$

— **five** levels, not four, and it *contains an exact zero*. This is neither the four-level textbook grid nor ternary; it is a five-level symmetric grid at $\log_2 5 \approx 2.32$ bits that keeps the off-state *and* adds two intermediate magnitudes $\pm 2/3$. That is the best of both prior rungs at once: the zero ternary had, plus the graded magnitude ternary lacked.

It is worth being precise about why the round-half-to-even convention creates the extra zero rather than the four-level grid the code evidently intended. If round broke ties away from zero, $\pm 0.5$ would go to $\pm 1$ and $\pm 1.5$ to $\pm 2$, the outer clamp would pin $\pm 2 \to \pm 1.5$, and after dividing by $1.5$ the set would be $\{-1, -1/3, +1/3, +1\}$ with no zero. Half-to-even flips two mappings — $0.5 \to 0$ (even) and $1.5 \to 2$ (even) — which both *adds* the $0$ level and *removes* the $\pm 1/3$ levels in favor of $\pm 2/3$. So the five-level grid is an artifact of the rounding convention interacting with the $\times 1.5$ scaling, not a designed choice — but it is exactly what the model is trained on, the measurement is of the code, and it happens to land on a strictly more capable grid than the nominal one. I take the grid the code computes, and the grid the code computes is five symmetric levels.

This matters so much for my expectation because I just diagnosed that the binding constraint is magnitude resolution, and this grid attacks exactly that while *also* keeping the off-switch I confirmed is worth a small real amount. A weight can now be off ($0$), weakly on ($\pm 2/3$), or strongly on ($\pm 1$), in either sign — five symbols against ternary's three, the extra $\approx 0.74$ bits per weight multiplied across 355M projection weights being a large amount of restored representational budget. The gap between three and five graded levels is far bigger than the gap between two levels and three, so I expect this rung to break out of the 2.72 plateau, not inch below it. The rest of the layer has to be right or the resolution gain evaporates. The scale stays the absmean $s = \mathrm{mean}(|W|)$ — the L2-optimal per-tensor unit from the same least-squares argument that gave the sign grid its scale, cheap and outlier-robust, placing the typical weight near the grid's $\pm 1$ so the bulk of the distribution falls inside the representable range. I deliberately do not use absmax for weights here: one outlier weight would inflate the scale and crush all ordinary weights toward a single level, wasting the five levels on representing the outlier. The round and the two clamps are bridged by the same detached-difference STE as both prior rungs — $(w_{\text{rounded}} - w_{\text{scaled}})\texttt{.detach()} + w_{\text{scaled}}$ makes the forward use the grid value and the backward pass identity to the normalized weight, and the inner clamp defines the forward saturation, so a scaled weight driven past $\pm 1.5$ is pinned and stops receiving a magnitude-increasing gradient through the round.

The activation path stays *identical* to the two earlier rungs — 8-bit symmetric per-tensor absmax, $Q_b = 127$, clip $[-127, 127]$, STE, dequant scale $\mathrm{max}|x|/127$ — and this is the controlled-experiment spine of the whole ladder. I keep activations at int8 rather than pushing them low because the diagnostic is decisive: weights are far easier to quantize than activations, which carry per-token outlier channels with huge dynamic range, so the contribution should stay in the *weight* grid. And here I use absmax rather than the absmean I chose for weights, because the failure modes are opposite — a clipped activation is a destroyed activation, whereas a clipped weight is just a slightly misplaced grid assignment, so the activation scale must be set by the maximum to guarantee nothing clips. Holding this path fixed across all three rungs means that when int2 beats ternary, it was the five-level weight grid and nothing else. As before, there is no normalization inside `BitLinear`: the block's pre-projection `LayerNorm` holds $E[\tilde x^2] \approx 1$, so $\mathrm{Var}(y) = n s^2 E[g^2] E[\tilde x^2]$ (with $g$ a grid value, $E[g^2]$ a fixed constant) stays at the float layer's order, no SubLN needed, and the recipe stays inherited (`CONFIG_OVERRIDES` empty, peak LR `6e-4`).

So the delta from ternary is one thing: where ternary offered "off or full-strength," this grid offers $\{-1, -2/3, 0, +2/3, +1\}$, adding two intermediate magnitudes while keeping the zero. I expect a large drop in validation loss, decisively below the 2.72 plateau — into the mid-2.4s if the resolution diagnosis is right — WikiText-2 falling sharply from ~78 toward the mid-50s and LAMBADA from ~110 toward the low 80s, and critically downstream should *move* this time where it did not for ternary: ARC-Easy from ~47 toward the low-to-mid 50s, HellaSwag from ~28 toward the low 30s, because the finely-weighted feature combinations those tasks need are now representable. If instead int2 lands only a hair below ternary, then the bottleneck is not the weight grid at all but the *fixed* absmean scale — a per-tensor statistic chosen to minimize reconstruction error, not task loss — and the next move would be to stop fixing the scale by a formula and learn it against the loss.

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
