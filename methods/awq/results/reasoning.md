I'm trying to make a large language model fast on a single device at batch size 1 — a chatbot on one GPU. When I profile it, generation is the bottleneck, and generation is memory-bound: the arithmetic intensity is about 1 FLOP per byte, miles below the roofline, and the memory traffic is dominated by hauling the weights out of DRAM one token at a time. The FLOPs are fixed and tiny; the only knob that moves the ceiling is reading *less weight memory*. So I want the weights in 4-bit (W4A16): activations stay FP16, weights drop to 4-bit, weight traffic falls ~4×, peak performance rises ~4×. The trouble is accuracy — round-to-nearest at 3–4 bits loses a lot. And whatever I do, it has to be post-training: no backprop, no per-layer regression, and ideally not even much dependence on the calibration set, so it doesn't overfit and still works across domains.

Let me test a concrete question: are all the weights equally important? The diagnostic is to quantize a layer to INT3 but keep some fraction of the weight channels in FP16, then ask how much accuracy comes back per channel kept. Keeping ~1% in FP16 recovers almost all the lost accuracy. So there's a small "salient" set doing the heavy lifting. The real question is which 1%.

The obvious guess is the large weights — keep the channels with the biggest weight magnitude / $L_2$-norm in FP16. That barely beats keeping a *random* 1%. So weight magnitude is not what makes a weight important. The other signal in the room is the activation. If the FP16-kept channels are the weight channels that multiply the *largest-magnitude input features* — selected by activation magnitude, not weight magnitude — the same 1% budget gives a large recovery. So saliency is set by the activation side. The reading is intuitive: an input feature with large magnitude contributes a lot to the output, so the weights that process it matter a lot; protect those weights and you protect the features that carry the signal.

So I have a working recipe: keep the activation-salient 1% of weight channels in FP16, quantize the rest. But it's a mixed-precision data type — some channels FP16, most INT3 — and that's a nightmare on hardware: irregular layout, special kernels, scattered memory access. The whole point was to be hardware-friendly and actually fast. I need to protect those salient weights *without literally storing them in a different precision*. Everything has to stay a uniform low-bit grid.

Let me look hard at where the quantization error in one input channel's weights actually comes from, and whether I can shrink it without changing the bit-width. Take a group of weights $\mathbf w$ with $y = \mathbf w x$, quantized as $Q(\mathbf w) = \Delta\cdot\mathrm{Round}(\mathbf w/\Delta)$, $\Delta = \max(|\mathbf w|)/2^{N-1}$ in the symmetric analysis. For one element's contribution, the absolute error is $\mathrm{Err}(Q(w)x) = \Delta\cdot|\mathrm{Round}(w/\Delta)-w/\Delta|\cdot|x|$. Call that absolute rounding residual $\mathrm{RoundErr}(w/\Delta)$. It is a grid-position error — how far $w/\Delta$ sits from the nearest integer — not a statement that larger weights automatically get larger residuals. I want to know its typical size, so let me just average $|\mathrm{Round}(t)-t|$ over $t$ spread across the grid: for a couple of million uniform draws of $t$ I get a mean of $0.2501$. So the residual averages about $0.25$, sitting roughly uniformly on $[0,0.5]$, and crucially it doesn't care about the weight's own magnitude — it cares about $\Delta$, the step size.

Now suppose an input channel has high activation magnitude, so I multiply the weights in that input channel by $s>1$ before quantizing and divide the corresponding activation by the same $s$. Before quantization this is an exact equivalence transform: for a whole linear layer,

$$\mathbf W\mathbf X = \big(\mathbf W\mathrm{diag}(\mathbf s)\big)\big(\mathrm{diag}(\mathbf s)^{-1}\mathbf X\big).$$

For one element in that scaled channel I compute $Q(w\cdot s)\cdot(x/s)$. Write it out: $Q(ws)\cdot(x/s) = \Delta'\cdot\mathrm{Round}(ws/\Delta')\cdot x\cdot(1/s)$, where $\Delta'$ is the new group step after scaling. The absolute error of this scaled-and-compensated version is

$$\mathrm{Err}\big(Q(ws)(x/s)\big) = \Delta'\cdot\mathrm{RoundErr}(ws/\Delta')\cdot|x|\cdot\frac{1}{s}.$$

Compare to the original error $\Delta\cdot\mathrm{RoundErr}(w/\Delta)\cdot|x|$. The expected rounding residual is still ~0.25 either way (I just measured that it depends on $\Delta$, not on the value being quantized), so in this coarse model that factor cancels, and the ratio of new error to old error is

$$\frac{\mathrm{Err}\big(Q(ws)(x/s)\big)}{\mathrm{Err}\big(Q(w)x\big)} \;=\; \frac{\Delta'}{\Delta}\cdot\frac{1}{s}.$$

If scaling a small fraction of channels by a modest $s$ leaves the group's maximum element alone — which I'd expect if the salient weight isn't itself the biggest weight in its group — then $\Delta'\approx\Delta$ and the ratio collapses to $1/s$. For $s>1$ that means the salient channel gets effectively finer resolution after I undo the scale on the activation side, purely from an equivalence transform: no FP16 side table, no change in bit-width.

That $\Delta'\approx\Delta$ step is an assumption, though, and the whole argument lives or dies on it. Let me actually quantize some groups and watch what happens to $\Delta'$ as I push $s$ up. I take random 8-weight groups on a 4-bit symmetric grid, mark one channel as salient, scale it by $s$, requantize, and divide the salient dequantized weight back by $s$. Over 200k trials I track the ratio of expected absolute errors directly:

| $s$ | salient err (new/old) | $1/s$ | $\mathbb E[\Delta'/\Delta]$ | non-salient err (new/old) |
|----|----|----|----|----|
| 1.0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1.5 | 0.557 | 0.667 | 1.075 | 1.092 |
| 2.0 | 0.327 | 0.500 | 1.214 | 1.255 |
| 3.0 | 0.146 | 0.333 | 1.581 | 1.667 |
| 4.0 | 0.082 | 0.250 | 2.001 | 2.127 |
| 6.0 | 0.036 | 0.167 | 2.876 | 3.078 |

Two things jump out, and one of them I did not expect. The salient error does shrink monotonically with $s$ — so the protection is real. But it shrinks a bit *less* than $1/s$ (0.557 vs 0.667 at $s{=}1.5$, 0.146 vs 0.333 at $s{=}3$), and the reason is staring at me in the third column: $\Delta'/\Delta$ is already above 1 even at $s{=}1.5$ and climbs steadily. So "$\Delta'\approx\Delta$" is not free — the very channel I'm protecting starts nudging the group maximum upward the moment I scale it, and the $1/s$ story is an optimistic idealization, valid only in the small-$s$ corner.

And the last column is the real catch. The error of every *non-salient* weight in that group is proportional to $\Delta$, so when $\Delta'/\Delta$ rises, all of those errors get *amplified* by exactly that ratio — the measured non-salient ratio (1.092, 1.255, … , 3.078) tracks $\mathbb E[\Delta'/\Delta]$ almost one-for-one. I had been mentally writing only the salient side, $(\Delta'/\Delta)(1/s)$, which keeps shrinking, and quietly ignoring that the same $\Delta'/\Delta$ multiplies everyone else's error upward. Protecting the salient weights by overscaling damages the rest of the group, and at $s{=}6$ I'm tripling the non-salient error to win back a factor of ~30 on a single channel. So there's a genuine trade-off, and I can't just crank $s$ — I have to choose it accounting for *both* the salient and the non-salient channels at once.

So I shouldn't pick $s$ to minimize one channel's error in isolation; I should pick the per-channel scale vector $\mathbf s$ to minimize the *whole layer's* output error after quantization. The honest objective is

$$\mathbf s^\* = \arg\min_{\mathbf s}\ \big\lVert Q(\mathbf W\,\mathrm{diag}(\mathbf s))\,(\mathrm{diag}(\mathbf s)^{-1}\mathbf X) - \mathbf W\mathbf X\big\rVert,$$

where $Q$ is the group-wise INT3/INT4 quantizer, $\mathbf X$ is calibration activations, and $\mathrm{diag}(\mathbf s)^{-1}\mathbf X$ can be folded into the previous operator by dividing that operator's output channels while multiplying the next linear layer's input columns. The problem is $Q$ has a $\mathrm{Round}$ in it and isn't differentiable. I could reach for straight-through or learned-step approximate gradients, but those bring back the unstable, backprop-dependent optimization I'm trying to avoid. I don't want to optimize $\mathbf s$ freely in $\mathbb R^{C_i}$ anyway; that's a huge, ill-conditioned, non-differentiable search.

I already know the one thing that determines saliency: the activation magnitude per channel. So I don't need a free search — I need a one-parameter family that dials "how much do I scale the high-activation channels," and I can grid-search that single knob cheaply. Let $\mathbf s_X$ be the average activation magnitude per input channel (from a tiny calibration pass — average, not max, because I only need the typical importance of a channel and averaging avoids overfitting to a few calibration samples). Define the search space

$$\mathbf s = \mathbf s_X^{\alpha},\qquad \alpha^\* = \arg\min_{\alpha\in[0,1]}\ \mathcal L(\mathbf s_X^{\alpha}).$$

At $\alpha=0$, $\mathbf s = \mathbf 1$ — no scaling, plain RTN. At $\alpha=1$, the scaling tracks the activation magnitude most aggressively — maximum protection of salient channels (and maximum risk of inflating $\Delta$ for the rest, the very $\Delta'/\Delta$ amplification I just watched climb to 3×). So $\alpha$ slides between "protect nobody" and "protect the salient channels hard," and the loss curve over $\alpha$ should be U-shaped: descending as protection helps the salient channels, then rising once the $\Delta'$ inflation starts hurting the rest more than the protection helps. The interior minimum is the balance point I want. Because it's one scalar, I just sweep $\alpha$ over a fine grid in $[0,1]$, and for each candidate I quantize the layer with that $\mathbf s$, run the calibration input through, and measure the actual output MSE against the FP16 output — the true objective, evaluated directly, no gradient needed, and no need to trust my coarse error model since the MSE includes whatever the model leaves out. Take the $\alpha$ with the lowest loss. I also fold a small weight-clipping step in (clip the group range to minimize the quantization MSE), which trims the worst rounding outliers. To keep the scales numerically tame I normalize $\mathbf s$ by the geometric mean of its max and min before applying it, so it neither blows up the weights nor collapses them.

One thing I should sanity-check before trusting any of this on a real layer: the affine zero-point quantizer the implementation actually uses isn't the symmetric $\Delta=\max(|\mathbf w|)/2^{N-1}$ of my analysis — it uses $(\max-\min)/(2^N-1)$ over each group, with a learned zero point. My $1/s$ argument was derived for the symmetric step, so I want to confirm the dependency survives the change of quantizer rather than assume it. The step size in both cases is "(group range) / (number of levels)", and scaling one channel up can only enlarge the group range (it pushes $\max$ up or $\min$ down), never shrink it. So $\Delta'\ge\Delta$ holds for the affine step too, and the error of a quantized element is still $\Delta'\cdot(\text{residual})$ with the residual independent of the step — the same two competing effects, salient $\propto(\Delta'/\Delta)(1/s)$ down, non-salient $\propto\Delta'/\Delta$ up. The constants differ but the trade-off and its U-shape don't, which is why scoring candidates by the real affine-quantized MSE rather than the symmetric formula is the safe move.

This stays inside the constraints I started with: no backprop, no second-order Hessian, no error-feedback regression over the calibration set. The cached calibration activations give me the per-channel average magnitude and the FP16 reference outputs used to score a tiny list of candidate scales, but I am not solving a per-weight reconstruction problem or propagating quantization error through a Hessian inverse. And since the stored result is still a uniform group-wise low-bit weight tensor, it can pack into a regular layout that an on-device kernel can dequantize and multiply.

Looking back, each step in this chain forced the next rather than being chosen. Generation is memory-bound, so weight traffic is the only lever, which points at W4A16. Plain RTN at 3–4 bits is too lossy, but the FP16-keep diagnostic showed the loss concentrates in a salient ~1% — and the magnitude-vs-activation comparison showed that 1% is picked out by *activation* magnitude, not weight magnitude. Keeping those channels in FP16 recovers the accuracy but builds a hardware-hostile mixed-precision tensor, so I needed protection without a separate precision. The equivalence transform supplies it: scaling a salient channel by $s$ and dividing its activation by $s$ shrinks that channel's error — my Monte-Carlo put the shrink a little above $1/s$ rather than exactly at it — but the same run exposed that the inflated group step $\Delta'$ amplifies every non-salient error in lockstep, so the scale has to balance the two. The activation statistic already names the right per-channel direction, which turns "find the scale vector" into "find one exponent" $\mathbf s = \mathbf s_X^{\alpha}$, and the U-shaped trade-off makes a grid search over $\alpha$ against the layer's real output MSE both cheap and well-posed. What comes out is an activation-aware, training-free, hardware-friendly 4-bit weight quantizer whose only calibration use is reading per-channel average magnitudes and scoring a handful of candidate scales — no backprop, no Hessian, no per-weight regression.

```python
import torch
import torch.nn as nn

@torch.no_grad()
def get_act_scale(x):
    # Per-input-channel average magnitude, the saliency signal.
    return x.abs().view(-1, x.shape[-1]).mean(0)


def pseudo_quantize_tensor(w, n_bit=4, q_group_size=128, zero_point=True):
    org_shape = w.shape
    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2

    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int, min_int = 2**n_bit - 1, 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        w = (torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros) * scales
    else:
        max_val = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        max_int, min_int = 2 ** (n_bit - 1) - 1, -(2 ** (n_bit - 1))
        scales = max_val / max_int
        w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

    return w.reshape(org_shape)


@torch.no_grad()
def search_module_scale(module_to_inspect, linears, inp, module_kwargs=None,
                        w_bit=4, q_group_size=128, n_grid=20):
    module_kwargs = dict(module_kwargs or {})
    module_kwargs.pop("use_cache", None)
    q_config = {"zero_point": True, "q_group_size": q_group_size}

    inp = inp.to(next(module_to_inspect.parameters()).device)
    org_out = module_to_inspect(inp, **module_kwargs)
    if isinstance(org_out, tuple):
        org_out = org_out[0]
    x_scale = get_act_scale(inp)
    org_state = {k: v.detach().cpu() for k, v in module_to_inspect.state_dict().items()}

    best_loss, best_scales = float("inf"), None
    for grid in range(n_grid):
        alpha = grid / n_grid
        scales = x_scale.pow(alpha).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()

        for fc in linears:
            fc_scales = scales.view(1, -1).to(fc.weight.device)
            fc.weight.mul_(fc_scales)
            fc.weight.data = pseudo_quantize_tensor(fc.weight.data, n_bit=w_bit, **q_config) / fc_scales

        out = module_to_inspect(inp, **module_kwargs)
        if isinstance(out, tuple):
            out = out[0]
        loss = (org_out - out).float().pow(2).mean().item()
        if loss < best_loss:
            best_loss, best_scales = loss, scales
        module_to_inspect.load_state_dict(org_state)

    return best_scales.detach().cpu()


class ScaledActivation(nn.Module):
    def __init__(self, act, scales):
        super().__init__()
        self.act = act
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        shape = [1] * (x.dim() - 1) + [-1]
        return self.act(x) / self.scales.view(*shape).to(x.device)


@torch.no_grad()
def scale_ln_fcs(ln, linears, scales):
    if not isinstance(linears, list):
        linears = [linears]
    scales = scales.to(device=ln.weight.device, dtype=ln.weight.dtype)
    ln.weight.div_(scales)
    if getattr(ln, "bias", None) is not None:
        ln.bias.div_(scales)
    for fc in linears:
        fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))


@torch.no_grad()
def scale_fc_fc(prev_fc, fc, scales):
    scales = scales.to(device=prev_fc.weight.device, dtype=prev_fc.weight.dtype)
    prev_fc.weight[-scales.numel():].div_(scales.view(-1, 1))
    if prev_fc.bias is not None:
        prev_fc.bias.div_(scales)
    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))


@torch.no_grad()
def scale_activation_fc(fc, scales):
    fc.weight.mul_(scales.view(1, -1).to(device=fc.weight.device, dtype=fc.weight.dtype))


@torch.no_grad()
def apply_scale(prev_op, named_linears, scales, input_feat=None):
    if not isinstance(named_linears, list):
        named_linears = [named_linears]
    names = [name for name, _ in named_linears]
    linears = [fc for _, fc in named_linears]

    replacement = None
    if isinstance(prev_op, nn.Linear):
        assert len(linears) == 1
        scale_fc_fc(prev_op, linears[0], scales)
    elif isinstance(prev_op, nn.LayerNorm) or prev_op.__class__.__name__.endswith("RMSNorm"):
        scale_ln_fcs(prev_op, linears, scales)
    elif isinstance(prev_op, (nn.GELU, nn.SiLU)):
        replacement = ScaledActivation(prev_op, scales)
        for fc in linears:
            scale_activation_fc(fc, scales)
    else:
        raise NotImplementedError(type(prev_op))

    if input_feat is not None:
        for name in names:
            input_feat[name].div_(scales.view(1, -1).to(input_feat[name].device))
    return replacement


@torch.no_grad()
def auto_clip_layer(w, input_feat, n_bit, q_group_size=128,
                    n_grid=20, max_shrink=0.5, n_sample_token=512):
    org_shape = w.shape
    group_size = q_group_size if q_group_size > 0 else w.shape[1]
    q_config = {"zero_point": True, "q_group_size": q_group_size}

    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    step = max(1, input_feat.shape[1] // n_sample_token)
    input_feat = input_feat[:, 0::step]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    org_max_val = w.abs().amax(dim=-1, keepdim=True)
    best_max_val = org_max_val.clone()
    min_errs = torch.ones_like(org_max_val) * 1e9
    org_out = (input_feat.to(w.device) * w).sum(dim=-1)

    for i_s in range(int(max_shrink * n_grid)):
        max_val = org_max_val * (1 - i_s / n_grid)
        cur_w = torch.clamp(w, -max_val, max_val)
        q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config)
        cur_out = (input_feat.to(w.device) * q_w).sum(dim=-1)
        err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
        better = err < min_errs
        min_errs[better] = err[better]
        best_max_val[better] = max_val[better]

    return best_max_val.squeeze(1).reshape(org_shape[0], -1, 1)


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


@torch.no_grad()
def quantize_block(block, scale_specs, input_feat, module_kwargs=None,
                   w_bit=4, q_group_size=128):
    # Each spec is (prev_op, set_prev_op, [(linear_name, linear_module), ...],
    # inspect_module, input_name). The setter installs ScaledActivation when needed.
    # The model-specific driver supplies groups such as q/k/v together, then mlp gate/up.
    for prev_op, set_prev_op, named_linears, inspect_module, input_name in scale_specs:
        linears = [fc for _, fc in named_linears]
        scales = search_module_scale(
            inspect_module, linears, input_feat[input_name], module_kwargs, w_bit, q_group_size
        )
        replacement = apply_scale(prev_op, named_linears, scales, input_feat)
        if replacement is not None:
            set_prev_op(replacement)

    for name, fc in get_named_linears(block).items():
        if any(token in name for token in ["q_", "k_", "query", "key", "Wqkv"]):
            continue
        max_val = auto_clip_layer(fc.weight, input_feat[name], w_bit, q_group_size)
        max_val = max_val.to(device=fc.weight.device, dtype=fc.weight.dtype)
        org_shape = fc.weight.shape
        fc.weight.data = fc.weight.data.reshape(*max_val.shape[:2], -1)
        fc.weight.data = torch.clamp(fc.weight.data, -max_val, max_val)
        fc.weight.data = fc.weight.data.reshape(org_shape)

    for fc in get_named_linears(block).values():
        fc.weight.data = pseudo_quantize_tensor(
            fc.weight.data,
            n_bit=w_bit,
            q_group_size=q_group_size,
            zero_point=True,
        )
```
