**Problem (from step 2).** GPTQ closed most of the weight-only gap with heavy second-order machinery
(per-layer Hessian, Cholesky inverse, column-wise error feedback) that regresses weights against the
calibration set, and it treats every weight as equally important. But pushing lower still, a tiny
fraction of weights carries most of the quality — and spending the bit budget uniformly is wasteful.

**Key idea.** **AWQ** (Lin 2023), activation-aware weight quantization. The salient weight channels are
*not* the largest-magnitude ones — they are the input channels whose *activations* are largest on
average (keeping ~0.1–1% of those in FP16 nearly recovers FP16 accuracy; keeping the top-magnitude
channels barely helps). But storing a mixed FP16/INT tensor is not hardware-friendly. So protect those
channels with an exact equivalence transform instead: WX = (W·diag(s))·(diag(s)⁻¹X). Scaling a salient
channel by s>1 shrinks its quantization error by ≈1/s (as long as it does not become the new group
maximum, so Δ'≈Δ); the activations are divided back by s, so the function is unchanged. Choose s =
s_X^α (per-channel mean |activation| to a power) by a small grid search over α that directly minimizes
output MSE vs FP16 — no gradients, no Hessian, no per-weight regression. Fold the scales into the
surrounding LayerNorm/linear so there is zero runtime overhead; the stored model is a plain group-wise
INT3/INT4 weight-only model.

**Why it works.** The activation statistic identifies which channels deserve resolution; the
equivalence transform delivers that resolution as a per-channel scale, not an irregular mixed-precision
layout. Searching α against true output MSE finds the sweet spot before over-scaling makes a salient
weight the group max and coarsens its neighbors.

**Change / code.** The activation-magnitude statistic, the α-search over equivalence scales, and the
group-wise quantizer they wrap.

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

**Target.** At the matched g128 grouping, beat RTN-g128 on Llama-2-7B (INT3-g128 6.66) and push INT3-g128
toward ~6.2 and INT4-g128 toward ~5.6 — a lighter, calibration-robust weight-only method, leaving the
*activations* as the only remaining frontier.
