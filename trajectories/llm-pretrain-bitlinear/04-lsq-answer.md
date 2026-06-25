**Problem.** int2_uniform broke the plateau (2.4392) by giving each weight more places to land, but on
every rung the grid is *anchored* by one fixed number per tensor — the absmean `s = mean(|W|)`, chosen to
minimize reconstruction error `‖W − s·grid‖²`. That is the wrong objective: I do not care whether the
discrete weights are close to the float weights, I care whether they make the *loss* small. At a few bits
per layer, the gap between reconstruction-optimal and loss-optimal level placement is the residual gap to
a usable model.

**Key idea (LSQ — Learned Step Size Quantization).** Make the
per-layer step size a **learnable parameter**, trained jointly with the weights against the next-token
loss. Quantizer `ŵ = round(clip(w/s, −Q_N, Q_P))·s`. STE on the round; differentiate divide/clip/multiply
honestly, which yields the interior step-gradient `∂ŵ/∂s = round(w/s) − w/s` — the negative signed
residual, large near a bin transition (where `s` most affects the code) and zero on a level. That is the
sensitivity a fixed absmean scale can never have. Balance the step's update/parameter ratio with the
weights' by scaling its gradient by `g = 1/√(N_W·Q_P)` (else `R ≈ √(N_W·Q_P)` over-drives it), injected
with the same detach trick as the STE.

**Why (adapted to this task).** Canonical LSQ fine-tunes from an FP teacher with momentum SGD; here it is
*pretraining from scratch* with AdamW on the fixed schedule, so the step is initialized from the scratch
weights via `s = 2·mean(|W|)/√Q_P` and trained by the same AdamW. Bit-width **3** (signed
`Q_N=4, Q_P=3`, grid `{−4,…,+3}`, 8 levels) tests both halves of the thesis: more levels than int2's
effective five, *and* placed by the loss. Activations held **identical** to all baselines (8-bit
per-tensor absmax, `Q_b=127`, STE) to isolate the learned-weight-step contribution. No SubLN (the block's
pre-projection `LayerNorm` holds the variance). All projections quantized uniformly (the harness ties
`wte`/`lm_head`); recipe inherited (`CONFIG_OVERRIDES = {}`).

**Bar to clear.** int2's 2.4392. I expect val_loss strictly below it (low-2.4s / high-2.3s, near the
~2.37 the strongest non-baseline run reached — evidence headroom exists), WikiText-2 < 54.1, LAMBADA <
82.0, downstream ≥ 53.8 / 31.5. Failure mode: if the `√(N_W·Q_P)` gradient scale is off, the step thrashes
or stalls and the model lands back at int2; validate by watching the per-layer `weight_scale` move
smoothly off its init.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 38-115) -- finale: LSQ (learned step, 3-bit weights)
def gradscale(x, scale):
    """Transparent gradient multiplier: forward = x, backward = grad * scale.
    Used to apply LSQ's 1/sqrt(N_W * Q_P) scaling to the step-size gradient."""
    y_out = x
    y_grad = x * scale
    return (y_out - y_grad).detach() + y_grad


def roundpass(x):
    """Straight-through round: forward = round(x), backward = identity."""
    y_out = x.round()
    y_grad = x
    return (y_out - y_grad).detach() + y_grad


def weight_quant(weight, step, Qn, Qp):
    """LSQ weight quantization with a LEARNED per-tensor step size.

    Forward: w_q = round(clip(W / s, -Qn, Qp)), dequantized by s.
    Backward: STE through round; step-size gradient = round(W/s) - W/s
              (interior), scaled by g = 1/sqrt(N_W * Qp) for stable training.
    Returns (quantized_weight, scale) with quantized_weight * scale ~= weight.
    """
    g = 1.0 / ((weight.numel() * Qp) ** 0.5)
    s = gradscale(step, g)                       # transparent forward, gradient * g
    w = torch.clamp(weight / s, -Qn, Qp)         # clipped weights get zero data-gradient
    w = roundpass(w)                             # STE round; interior s-grad = round(W/s) - W/s
    return w, s                                  # w * s ~= weight


def activation_quant(x):
    """Absmax 8-bit activation quantization with STE (identical to the baselines)."""
    Qb = 127  # int8 range
    scale = x.detach().abs().max().clamp(min=1e-12)
    x_normed = x / scale
    x_q = (x_normed * Qb).round().clamp(-Qb, Qb)
    x_q = (x_q - x_normed * Qb).detach() + x_normed * Qb
    return x_q, scale / Qb


class BitLinear(nn.Module):
    """LSQ linear layer: 3-bit weights with a learnable step size, 8-bit activations.

    self.weight is the float latent weight the optimizer updates; self.weight_scale is the
    learnable per-tensor LSQ step, initialized to 2*mean(|W|)/sqrt(Qp). Same path train/eval.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        bits = 3
        self.Qn = 2 ** (bits - 1)               # 4 for 3-bit signed
        self.Qp = 2 ** (bits - 1) - 1           # 3 for 3-bit signed
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        # LSQ step-size init from the scratch weights: s = 2<|W|>/sqrt(Qp)
        with torch.no_grad():
            init_s = 2.0 * self.weight.abs().mean() / (self.Qp ** 0.5)
        self.weight_scale = nn.Parameter(init_s.clone().clamp(min=1e-5))

    def forward(self, x):
        w_q, w_scale = weight_quant(self.weight, self.weight_scale, self.Qn, self.Qp)
        x_q, x_scale = activation_quant(x)
        out = F.linear(x_q, w_q, None)
        out = out * (w_scale * x_scale)
        if self.bias is not None:
            out = out + self.bias
        return out
```
