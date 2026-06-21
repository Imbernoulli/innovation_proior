I have a billion-parameter language model whose weights must be stored at two, three, or four bits, and the two regimes I know put me in a bind. Quantization-aware training recovers low-bit accuracy by backpropagating the task loss through the quantized network so the weights can reorganize to fit the coarse grid, but for a model this size that means full-model gradients, large data, and thousands of steps — a cost I cannot pay. Post-training quantization is the opposite: round once, gradient-free, on a small calibration set, cheap enough to run — but it collapses at low bit-widths, because rounding to a handful of levels is irreversible and the round-to-nearest cover does nothing to fight the coarseness. The diagnosis is sharp. A uniform affine quantizer is $W_q = \operatorname{clamp}(\operatorname{round}(W/h) + z,\ 0,\ 2^N - 1)$, dequantized as $(W_q - z)\,h$, and the one number that controls everything is the scale $h$, set from the per-group clipping range $h = (x_{\max} - x_{\min})/(2^N - 1)$. Standard min-max PTQ takes $x_{\max}, x_{\min}$ to be the literal group extremes — a no-clip cover. Picture a group of 128 weights at two bits: four codes. Weight distributions in these models are heavy-tailed, so one or two outliers stretch $x_{\max}/x_{\min}$ out to themselves, the four codes are spaced to span that stretched range, and the 126 ordinary weights — which live in a narrow band near zero — are all snapped onto a grid far too coarse for them. The error is dominated by coarsely rounding the bulk just to faithfully cover a few outliers, and once rounded it is frozen. The competing fixes do not touch this exact point: GPTQ compensates rounding error second-order but never *learns* a clip and degrades at INT2; AWQ applies a closed-form activation-aware per-channel scale, not a per-group learned clip; percentile clipping is one global heuristic, not optimized per group against the loss.

The leverage is therefore to stop faithfully covering the outliers: clip the range inward so the few codes pack onto the bulk, accepting that outliers clamp to the rail. But the right amount of clipping depends on the group's distribution and on what the clip costs the model's output, which no fixed percentile can know — the right clip for one group is wrong for another. So I do not *set* the clip; I *learn* it. I propose OmniQuant's Learnable Weight Clipping (LWC): make the per-group clipping range a trainable quantity, optimized to minimize how much the quantized weights perturb the layer's output, and spend my small budget on exactly that one degree of freedom. Learning the clip runs into the wall that keeps PTQ gradient-free — the quantizer contains a $\operatorname{round}$, flat almost everywhere, so no gradient flows back to a range parameter. The straight-through estimator removes the wall: define $\operatorname{round\_ste}(x) = (\operatorname{round}(x) - x).\mathrm{detach}() + x$, which is $\operatorname{round}(x)$ in the forward pass and slope-one in the backward pass. With that, the rest of the quantizer — the divide by $h$, and the dependence of $h$ on the clipping extremes — is an ordinary differentiable function of the range parameters, so a gradient can reach them.

What remains is to parameterize the clip well, and three guarantees pin down the form. First, the clip should only ever shrink the range, never expand it past the literal extremes, since expanding wastes codes on values that do not exist; so the learned quantity must be a ratio in $(0,1]$ applied to the extremes, not a free scale. Second, the optimization should start somewhere sane and improve monotonically — the natural start is the min-max cover itself (ratio $\approx 1$), exactly what plain PTQ does, so I begin no worse than PTQ and learn to clip inward from there. Third, the gradient should be bounded: at four codes the reconstruction loss in the range is jagged, because nudging the range reassigns whole blocks of weights across codes, so an unbounded parameter driven by that gradient would jump around; a saturating map damps it. A sigmoid delivers all three at once. I introduce a learnable parameter per group, per side, and set the clipped extremes to
$$x_{\max} = \sigma(\gamma_{\text{up}})\cdot\max(W_{\text{group}}),\qquad x_{\min} = \sigma(\gamma_{\text{low}})\cdot\min(W_{\text{group}}).$$
Because $\sigma \in (0,1)$ the clip can only shrink the range — guarantee one. Initializing the factors to $4.0$ gives $\sigma(4) \approx 0.982$, so the grid starts essentially at the min-max cover and trims from there — guarantee two. And the derivative $\frac{d\sigma(\gamma)}{d\gamma} = \sigma(\gamma)\bigl(1 - \sigma(\gamma)\bigr)$ saturates toward zero as the factor approaches either extreme, damping the jagged range-gradient — guarantee three. Two factors per group rather than one because the positive and negative extremes can want different clips. From these clipped extremes the scale and zero-point are computed exactly as before: symmetric signed weights use $h = \max(|x_{\max}|, |x_{\min}|)/(2^{N-1} - 1)$, clamped to $[10^{-5},\,10^4]$ so a degenerate all-zero group cannot produce a zero scale (and a runaway one cannot explode), with $z = \text{None}$; asymmetric uses $h = (x_{\max} - x_{\min})/(2^N - 1)$ with $z = -\operatorname{round}(x_{\min}/h)$. The fake-quant is the straight-through $\operatorname{round\_ste}(W/h)\,(+\,z)$, clamped to the code range, times $h$. The gradient flows: output reconstruction loss $\to$ through $\operatorname{round\_ste}$ (identity backward) $\to$ through $h$, a differentiable function of $\sigma(\gamma)\cdot\text{extreme}$ $\to$ into $\gamma$.

The reason this stays at a PTQ budget is that the clipping factors are *local*: a group's scale only affects its own layer's output, so there is no need for the global task loss. I optimize block by block — for each transformer block in turn, run the full-precision block and the quantized block on the same small calibration set (about 128 sequences), and minimize the reconstruction MSE between the two block outputs over a handful of epochs, with no backprop through the rest of the model. The straight-through estimator carries the gradient through the round, the sigmoid keeps the factors bounded and stable, and the per-block locality keeps the cost at PTQ scale. This fixes the disease I diagnosed: $\gamma$ can learn to pull $x_{\max}$ down below an outlier so the codes pack onto the bulk while the outliers clamp to the rail, chosen per group by the loss rather than a heuristic; a clip-free group whose cover is already right simply keeps its factor near its init and loses nothing; and the bounded, saturating gradient keeps the optimization stable even at two bits where the range-loss is jagged. The mechanism is minimal — one extra parameter per side per group, a sigmoid, and an STE round — the smallest thing that turns the uncalibrated min-max cover into a learned, per-group, loss-aware clip. One companion problem it deliberately leaves untouched: clipping the weight range does nothing about *activation* outliers in weight-activation quantization, which are better handled by migrating the difficulty into the weights through a learnable equivalent channel-wise scale and shift that leaves the layer's function unchanged. That is a separate, complementary mechanism optimized in the same per-block loop; for weight-only quantization, LWC is the complete weight-side story, and it is the part that directly attacks the irreversible rounding error at INT2/INT3.

```python
import torch, torch.nn as nn

CLIPMIN = 1e-5

def round_ste(x):
    return (x.round() - x).detach() + x          # round forward; identity backward

class UniformAffineQuantizer(nn.Module):
    def __init__(self, n_bits, group_size, shape, symmetric=False, lwc=False):
        super().__init__()
        self.n_bits = n_bits
        self.qmin, self.qmax = 0, 2 ** n_bits - 1
        self.group_size = group_size
        self.symmetric = symmetric
        self.lwc = lwc
        self.sigmoid = nn.Sigmoid()
        if lwc:
            init_value = 4.0                      # sigmoid(4) ~ 0.982: start at min-max cover
            self.upbound_factor = nn.Parameter(torch.ones(shape) * init_value)
            self.lowbound_factor = nn.Parameter(torch.ones(shape) * init_value)

    def calibration(self, x):
        xmax = x.amax(dim=-1, keepdim=True)
        xmin = x.amin(dim=-1, keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor) * xmax   # learnable inward clip (0,1)
            xmin = self.sigmoid(self.lowbound_factor) * xmin
        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = (abs_max / (2 ** (self.n_bits - 1) - 1)).clamp(min=CLIPMIN, max=1e4)
            zero_point = None
        else:
            scale = (xmax - xmin).clamp(min=CLIPMIN) / (2 ** self.n_bits - 1)
            zero_point = -(xmin / scale).round()
        return scale, zero_point

    def fake_quant(self, x, scale, zero_point):
        x_int = round_ste(x / scale)
        if zero_point is not None:
            x_int = x_int + zero_point
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dq = x_int
        if zero_point is not None:
            x_dq = x_dq - zero_point
        return x_dq * scale

    def forward(self, x):
        scale, zero_point = self.calibration(x)
        return self.fake_quant(x, scale, zero_point)
```
