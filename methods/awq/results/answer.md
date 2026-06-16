AWQ (Activation-aware Weight Quantization) is a post-training, weight-only low-bit quantizer for LLM inference. It targets the batch-1 generation regime, where weight traffic dominates memory movement, so reducing FP16 weights to INT4/INT3 while keeping activations in FP16 is the main lever.

The central observation is that the salient weight channels are not the channels with the largest weight norm. They are the input channels whose activations have the largest average magnitude. Keeping only a small activation-selected fraction of weight channels in FP16 is diagnostic evidence that these channels matter, but storing a mixed FP16/INT tensor is not hardware-friendly.

The replacement is an equivalence transform followed by ordinary group-wise quantization. For a linear layer,

$$\mathbf W\mathbf X = \big(\mathbf W\mathrm{diag}(\mathbf s)\big)\big(\mathrm{diag}(\mathbf s)^{-1}\mathbf X\big).$$

Before quantization the transform is exact. After quantization, scaling a salient channel by $s>1$ changes one element's compensated contribution from $Q(w)x$ to $Q(ws)(x/s)$. With $Q(\mathbf w)=\Delta\mathrm{Round}(\mathbf w/\Delta)$ and $\Delta=\max(|\mathbf w|)/2^{N-1}$ in the symmetric error analysis,

$$\frac{\mathrm{Err}(Q(ws)(x/s))}{\mathrm{Err}(Q(w)x)}=\frac{\Delta'}{\Delta}\frac{1}{s}.$$

If the scaled salient value does not change the group maximum, $\Delta'\approx\Delta$, so the salient channel's error shrinks by about $1/s$. The implementation uses an affine per-group zero-point quantizer, but the same round-to-nearest step-size logic applies. If $s$ is too large, $\Delta'$ grows and non-salient weights in the same group get more error. The scale is therefore chosen by a one-dimensional search over activation-derived scales:

$$\mathbf s=\mathbf s_X^\alpha,\qquad \alpha^*=\arg\min_\alpha \left\lVert Q(\mathbf W\mathrm{diag}(\mathbf s))(\mathrm{diag}(\mathbf s)^{-1}\mathbf X)-\mathbf W\mathbf X\right\rVert.$$

$\mathbf s_X$ is the per-input-channel average activation magnitude. A small grid over $\alpha$ directly evaluates output MSE against the FP16 block output, with no gradients, Hessian reconstruction, or per-weight calibration-set regression. The chosen scales are applied to the previous operator and the following linear layer so the network function is unchanged before quantization; then optional MSE range clipping and group-wise INT quantization produce the stored low-bit weights.

```python
import torch
import torch.nn as nn

@torch.no_grad()
def get_act_scale(x):
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
def search_module_scale(module_to_inspect, linears, x, module_kwargs=None,
                        w_bit=4, q_group_size=128, n_grid=20):
    module_kwargs = dict(module_kwargs or {})
    module_kwargs.pop("use_cache", None)
    q_config = {"zero_point": True, "q_group_size": q_group_size}

    x = x.to(next(module_to_inspect.parameters()).device)
    org_out = module_to_inspect(x, **module_kwargs)
    if isinstance(org_out, tuple):
        org_out = org_out[0]
    x_scale = get_act_scale(x)
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

        out = module_to_inspect(x, **module_kwargs)
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

The stored model remains a regular group-wise INT3/INT4 weight-only model with FP16 activations. The activation statistic chooses which channels deserve more quantization resolution; the equivalence transform supplies that protection without introducing an irregular mixed-precision layout.
