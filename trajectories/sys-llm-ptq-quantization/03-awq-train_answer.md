GPTQ closed most of the weight-only gap: 3-bit per-channel LLaMA-7B went from RTN's 25.54 down to 8.07, against FP16's 5.68. That is a real win and it makes me want to push the low-bit weight-only regime harder, because it is the one that actually speeds up batch-1 generation. But two things bother me when I try to go lower still. GPTQ leans on a full second-order reconstruction — a Hessian per layer, a Cholesky inverse, column-by-column error feedback — and because it regresses the quantized weights against the calibration set's input statistics, I worry it can *overfit* to that particular few hundred sequences. More fundamentally, GPTQ treats every weight as equally deserving of accuracy: it minimizes the total output error with no notion that *some weights matter far more than others*. If a tiny fraction of weights carries most of the model's quality, spending the bit budget uniformly is wasteful, and protecting exactly that fraction might beat reconstructing everything.

The diagnostic settles it. Keep a small fraction of weight channels — 0.1% to 1% — in full FP16 and quantize the rest to INT3, and ask which channels to keep. The obvious guess, the largest-magnitude weight channels, barely helps. That is the informative surprise: the magnitude of a weight is *not* what makes it important. Select instead by the *activations* that flow through each channel — keep in FP16 the input channels whose activations have the largest average magnitude — and this recovers almost all of the FP16 accuracy from an otherwise-INT3 model. Saliency is activation-defined, not weight-defined, which makes sense in hindsight: a weight matters only in proportion to how large the input it multiplies tends to be. But the obvious implementation is a dead end, because keeping 1% of channels in FP16 means a *mixed-precision* tensor — a ragged layout with FP16 columns interleaved among INT3 columns — which is a nightmare for a GPU kernel and throws away the clean, regular weight matrix that was the whole point.

The method is **AWQ**, activation-aware weight quantization, and the move is to get the *protection* the FP16 channels gave without the irregular layout. I do not store salient channels in higher precision; I *scale them up* before quantizing so they get more effective grid resolution, and divide the corresponding activations down by the same factor so the layer's function is unchanged. For a linear layer this is an exact equivalence transform,
$$W X = \big(W \cdot \mathrm{diag}(s)\big)\,\big(\mathrm{diag}(s)^{-1} X\big),$$
the product identical before any quantization. The payoff is what happens *after* quantization. With a round-to-nearest quantizer $Q(w) = \Delta\cdot\mathrm{round}(w/\Delta)$ and step $\Delta = \max(|w|)/2^{\,N-1}$, scaling a salient channel by $s>1$ changes its quantized contribution from $Q(w)\,x$ to $Q(ws)\,(x/s)$, and the relative error works out to
$$\frac{\mathrm{Err}\big(Q(ws)(x/s)\big)}{\mathrm{Err}\big(Q(w)x\big)} = \frac{\Delta'}{\Delta}\cdot\frac{1}{s}.$$
If $s$ is moderate enough that the scaled-up salient weight does not become the new group maximum, then $\Delta' \approx \Delta$ and the salient channel's quantization error shrinks by about $1/s$. I have given the important channel effectively finer resolution using nothing but a per-channel scale that folds away at inference — no mixed precision, no ragged layout, the stored tensor still a plain group-wise INT3/INT4 matrix.

The catch is the same $\Delta'$ in that ratio. Scale a salient channel too hard and the scaled weight *does* become the group maximum: $\Delta'$ grows, every *other* weight in the group rounds against a coarser grid, and I have protected one channel by hurting its neighbors. So there is a sweet spot, and rather than derive it analytically I parametrize the scales to track activation magnitude, $s = s_X^{\alpha}$ where $s_X$ is the per-input-channel average activation magnitude, and *search* the exponent $\alpha$ over a small grid that directly minimizes the layer's output MSE against the FP16 reference. No gradients, no Hessian, no per-weight regression — just a handful of forward passes choosing one scalar $\alpha$ per layer. At $\alpha = 0$ there is no scaling (back to RTN); as $\alpha$ grows the salient channels get more protection until the group-max penalty starts to bite, and the grid search finds the balance. The chosen scale is then *folded into the surrounding ops* — divided into the preceding LayerNorm or previous linear's output, multiplied into the current linear's weights — so the network function is exactly preserved at zero runtime overhead, and the stored model is an ordinary group-wise INT3/INT4 weight-only model. The activation statistic decided which channels deserved resolution; the equivalence transform delivered it without an irregular layout.

The honest comparison is at the *same* grouping, since g128 and per-channel numbers are not interchangeable and AWQ is naturally group-wise, so I pin against RTN-g128: Llama-2-7B INT3-g128 is 6.66. The bet is that one searched per-channel scale per layer — pure forward passes, no second-order machinery, no risk of regressing onto the calibration set — closes most of the remaining gap to FP16 ($\approx 5.47$) by spending the grid where the activations say it matters, bringing INT3-g128 toward $\sim 6.2$ and the easier INT4-g128 into the $\sim 5.6$ range. With the weight-only frontier in good shape, the next pressure has to come from the side I have ignored entirely: the *activations*.

```python
import torch

@torch.no_grad()
def get_act_scale(x):                       # per-input-channel mean |activation|
    return x.abs().view(-1, x.shape[-1]).mean(0)

def pseudo_quantize_tensor(w, n_bit=4, q_group_size=128):
    org_shape = w.shape
    w = w.reshape(-1, q_group_size)                          # group-wise affine grid
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    w = (torch.clamp(torch.round(w / scales) + zeros, 0, max_int) - zeros) * scales
    return w.reshape(org_shape)

@torch.no_grad()
def search_module_scale(module, linears, x, w_bit=4, q_group_size=128, n_grid=20):
    org_out = module(x)                                      # FP16 reference output
    x_scale = get_act_scale(x)
    org_sd = {k: v.cpu() for k, v in module.state_dict().items()}
    best_loss, best_scales = float("inf"), None
    for grid in range(n_grid):                              # search the exponent α
        alpha = grid / n_grid
        scales = x_scale.pow(alpha).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()
        for fc in linears:                                 # W·diag(s), quantize, divide back
            fc.weight.mul_(scales.view(1, -1))
            fc.weight.data = pseudo_quantize_tensor(fc.weight.data, w_bit, q_group_size) / scales.view(1, -1)
        loss = (org_out - module(x)).float().pow(2).mean().item()   # output MSE vs FP16
        if loss < best_loss:
            best_loss, best_scales = loss, scales
        module.load_state_dict(org_sd)
    return best_scales

@torch.no_grad()
def scale_ln_fcs(ln, linears, scales):      # fold diag(s)^-1 into LayerNorm, diag(s) into next linears
    ln.weight.div_(scales)
    if getattr(ln, "bias", None) is not None:
        ln.bias.div_(scales)
    for fc in linears:
        fc.weight.mul_(scales.view(1, -1))
```
