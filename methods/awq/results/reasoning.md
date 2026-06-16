I'm trying to make a large language model fast on a single device at batch size 1 — a chatbot on one GPU. When I profile it, generation is the bottleneck, and generation is memory-bound: the arithmetic intensity is about 1 FLOP per byte, miles below the roofline, and the memory traffic is dominated by hauling the weights out of DRAM one token at a time. The FLOPs are fixed and tiny; the only knob that moves the ceiling is reading *less weight memory*. So I want the weights in 4-bit (W4A16): activations stay FP16, weights drop to 4-bit, weight traffic falls ~4×, peak performance rises ~4×. The trouble is accuracy — round-to-nearest at 3–4 bits loses a lot. And whatever I do, it has to be post-training: no backprop, no per-layer regression, and ideally not even much dependence on the calibration set, so it doesn't overfit and still works across domains.

Let me test a concrete question: are all the weights equally important? The diagnostic is to quantize a layer to INT3 but keep some fraction of the weight channels in FP16, then ask how much accuracy comes back per channel kept. Keeping ~1% in FP16 recovers almost all the lost accuracy. So there's a small "salient" set doing the heavy lifting. The real question is which 1%.

The obvious guess is the large weights — keep the channels with the biggest weight magnitude / $L_2$-norm in FP16. That barely beats keeping a *random* 1%. So weight magnitude is not what makes a weight important. The other signal in the room is the activation. If the FP16-kept channels are the weight channels that multiply the *largest-magnitude input features* — selected by activation magnitude, not weight magnitude — the same 1% budget gives a large recovery. So saliency is set by the activation side. The reading is intuitive: an input feature with large magnitude contributes a lot to the output, so the weights that process it matter a lot; protect those weights and you protect the features that carry the signal.

So I have a working recipe: keep the activation-salient 1% of weight channels in FP16, quantize the rest. But it's a mixed-precision data type — some channels FP16, most INT3 — and that's a nightmare on hardware: irregular layout, special kernels, scattered memory access. The whole point was to be hardware-friendly and actually fast. I need to protect those salient weights *without literally storing them in a different precision*. Everything has to stay a uniform low-bit grid.

Let me look hard at where the quantization error in one input channel's weights actually comes from, and whether I can shrink it without changing the bit-width. Take a group of weights $\mathbf w$ with $y = \mathbf w x$, quantized as $Q(\mathbf w) = \Delta\cdot\mathrm{Round}(\mathbf w/\Delta)$, $\Delta = \max(|\mathbf w|)/2^{N-1}$ in the symmetric analysis. For one element's contribution, the absolute error is $\mathrm{Err}(Q(w)x) = \Delta\cdot|\mathrm{Round}(w/\Delta)-w/\Delta|\cdot|x|$. Call that absolute rounding residual $\mathrm{RoundErr}(w/\Delta)$; it is roughly uniform on $[0,0.5]$, about $0.25$ on average, and it is a grid-position error, not a statement that larger weights automatically get larger residuals.

Now suppose an input channel has high activation magnitude, so I multiply the weights in that input channel by $s>1$ before quantizing and divide the corresponding activation by the same $s$. Before quantization this is an exact equivalence transform: for a whole linear layer,

$$\mathbf W\mathbf X = \big(\mathbf W\mathrm{diag}(\mathbf s)\big)\big(\mathrm{diag}(\mathbf s)^{-1}\mathbf X\big).$$

For one element in that scaled channel I compute $Q(w\cdot s)\cdot(x/s)$. Write it out: $Q(ws)\cdot(x/s) = \Delta'\cdot\mathrm{Round}(ws/\Delta')\cdot x\cdot(1/s)$, where $\Delta'$ is the new group step after scaling. The absolute error of this scaled-and-compensated version is

$$\mathrm{Err}\big(Q(ws)(x/s)\big) = \Delta'\cdot\mathrm{RoundErr}(ws/\Delta')\cdot|x|\cdot\frac{1}{s}.$$

Compare to the original error $\Delta\cdot\mathrm{RoundErr}(w/\Delta)\cdot|x|$. Three facts. The expected absolute rounding residual is still ~0.25 either way, so that factor is unchanged in this coarse error model. Scaling a small fraction of activation-selected channels by a modest $s$ often leaves most groups' maximum unchanged, so $\Delta'\approx\Delta$ for those groups. And $\Delta$ and $x$ are represented in FP16 in this weight-only setting, so the analysis is about the integer grid applied to the weights. The ratio of new error to old error is

$$\frac{\Delta'}{\Delta}\cdot\frac{1}{s} \approx \frac{1}{s}.$$

For $s>1$, the salient channel gets effectively finer resolution after I undo the scale on the activation side. A deployed affine zero-point quantizer uses a different exact step, $(\max-\min)/(2^N-1)$ inside each group, but the same dependency remains: the compensated error shrinks with $1/s$ unless the group range grows enough to enlarge the step. I have protected those weights purely by an equivalence transform, with no FP16 side table and no change in bit-width. That's the escape from the mixed-precision data type.

The scaling diagnostic gives the next constraint. When I inspect actual weight groups under a few hypothetical values of $s$, modest scaling usually leaves the group maximum alone, so the $\Delta'\approx\Delta$ assumption is a useful model. But the fraction of changed group steps and the average $\Delta'/\Delta$ both rise as $s$ grows.

Why doesn't more scaling keep helping? Because the assumption $\Delta'\approx\Delta$ breaks when $s$ gets large. Push $s$ high enough and the scaled salient weight starts to *become* the group maximum, so $\Delta'$ grows. And the error of every *non-salient* weight in that group is proportional to $\Delta$ — so when $\Delta'/\Delta$ rises above 1, all the non-salient weights' errors get *amplified* by that ratio. I've been writing the salient error as $(\Delta'/\Delta)(1/s)$, which keeps shrinking, but I forgot the other side: the non-salient errors scale as $\Delta'/\Delta$, which keeps growing. Protecting the salient weights by overscaling quietly damages everyone else. So there's a real trade-off, and I can't just crank $s$ — I have to choose it accounting for *both* the salient and the non-salient channels.

So I shouldn't pick $s$ to minimize one channel's error in isolation; I should pick the per-channel scale vector $\mathbf s$ to minimize the *whole layer's* output error after quantization. The honest objective is

$$\mathbf s^\* = \arg\min_{\mathbf s}\ \big\lVert Q(\mathbf W\,\mathrm{diag}(\mathbf s))\,(\mathrm{diag}(\mathbf s)^{-1}\mathbf X) - \mathbf W\mathbf X\big\rVert,$$

where $Q$ is the group-wise INT3/INT4 quantizer, $\mathbf X$ is calibration activations, and $\mathrm{diag}(\mathbf s)^{-1}\mathbf X$ can be folded into the previous operator by dividing that operator's output channels while multiplying the next linear layer's input columns. The problem is $Q$ has a $\mathrm{Round}$ in it and isn't differentiable. I could reach for straight-through or learned-step approximate gradients, but those bring back the unstable, backprop-dependent optimization I'm trying to avoid. I don't want to optimize $\mathbf s$ freely in $\mathbb R^{C_i}$ anyway; that's a huge, ill-conditioned, non-differentiable search.

I already know the one thing that determines saliency: the activation magnitude per channel. So I don't need a free search — I need a one-parameter family that dials "how much do I scale the high-activation channels," and I can grid-search that single knob cheaply. Let $\mathbf s_X$ be the average activation magnitude per input channel (from a tiny calibration pass — average, not max, because I only need the typical importance of a channel and averaging avoids overfitting to a few calibration samples). Define the search space

$$\mathbf s = \mathbf s_X^{\alpha},\qquad \alpha^\* = \arg\min_{\alpha\in[0,1]}\ \mathcal L(\mathbf s_X^{\alpha}).$$

At $\alpha=0$, $\mathbf s = \mathbf 1$ — no scaling, plain RTN. At $\alpha=1$, the scaling tracks the activation magnitude most aggressively — maximum protection of salient channels (and maximum risk of inflating $\Delta$ for the rest). $\alpha$ slides between "protect nobody" and "protect the salient channels hard," which is exactly the trade-off I diagnosed. Because it's one scalar, I just sweep $\alpha$ over a fine grid in $[0,1]$, and for each candidate I quantize the layer with that $\mathbf s$, run the calibration input through, and measure the actual output MSE against the FP16 output — the true objective, evaluated directly, no gradient needed. Take the $\alpha$ with the lowest loss. I also fold a small weight-clipping step in (clip the group range to minimize the quantization MSE), which trims the worst rounding outliers. To keep the scales numerically tame I normalize $\mathbf s$ by the geometric mean of its max and min before applying it, so it neither blows up the weights nor collapses them.

This stays inside the constraints I started with: no backprop, no second-order Hessian, no error-feedback regression over the calibration set. The cached calibration activations give me the per-channel average magnitude and the FP16 reference outputs used to score a tiny list of candidate scales, but I am not solving a per-weight reconstruction problem or propagating quantization error through a Hessian inverse. And since the stored result is still a uniform group-wise low-bit weight tensor, it can pack into a regular layout that an on-device kernel can dequantize and multiply.

The causal chain now lines up: on-device generation is memory-bound, so I want W4A16; RTN at 3–4 bits is too lossy because weights aren't equally important; the salient ~1% is identified by *activation* magnitude, not weight magnitude; keeping them in FP16 works but is a hardware-hostile mixed-precision type; scaling a salient channel by $s>1$ and dividing the corresponding activation by $s$ cuts that channel's quantization error by $\sim1/s$ at no extra bits, as long as $\Delta$ does not grow much; overscaling inflates the group step $\Delta$ and amplifies the non-salient errors, so the scale must balance both; that balance is captured by a single knob over the activation-derived scale $\mathbf s = \mathbf s_X^{\alpha}$, grid-searched against the layer's real output MSE, giving an activation-aware, training-free, hardware-friendly 4-bit weight quantizer.

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
