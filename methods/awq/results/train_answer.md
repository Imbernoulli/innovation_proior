On-device batch-1 LLM inference is memory-bound: the arithmetic intensity is roughly one FLOP per byte, so the only way to speed up generation is to move less weight memory. Weight-only quantization to 4 or even 3 bits (W4A16 / W3A16, with FP16 activations) cuts weight traffic about fourfold and is the natural lever. The obstacle is accuracy. Round-to-nearest at 3-4 bits degrades perplexity sharply because it treats every weight as equally important and lets a single outlier inflate the step size for an entire group. We need a post-training quantizer that recovers near-FP16 accuracy without retraining, without backpropagation, without per-layer reconstruction regression, and without introducing a hardware-unfriendly mixed-precision layout.

The existing baselines each miss one of those requirements. Round-to-nearest is hardware-friendly but accuracy-poor. GPTQ improves accuracy by minimizing layer output error through a Hessian-based column sweep, yet it still performs a calibration-set regression, can overfit, and its dense inverse and column sweep are expensive. A mixed-precision oracle that keeps the most salient 1% of weight channels in FP16 does recover accuracy, but the resulting FP16/INT tensor is irregular and difficult to deploy efficiently. The real question is therefore how to protect the salient channels while keeping every stored weight on the same low-bit grid.

The method is AWQ, Activation-aware Weight Quantization. Its central premise is that saliency is set by activations, not by weight magnitude: the weight channels worth protecting are the ones that multiply the largest-magnitude input features, not the ones with the largest weight norm. Intuitively, an input feature with large magnitude contributes heavily to the output, so the weights processing it deserve finer quantization resolution, while a channel's own weight magnitude carries no such signal. The way to tell the two hypotheses apart is to quantize a layer to a low bit-width, keep a small fraction of channels in FP16 chosen once by activation magnitude and once by weight magnitude at the same budget, and compare how much accuracy each selection recovers; the activation-based selection should recover most of what round-to-nearest loses, while the weight-magnitude selection should barely beat keeping a random fraction, since raw weight size says nothing about how much a channel actually contributes to the output. AWQ protects the activation-salient channels with an equivalence transform rather than a different data type, so the mixed-precision idea never has to survive contact with hardware.

For a linear layer WX, apply a per-channel scaling s to the weights and the inverse scaling to the activations: WX = (W diag(s)) (diag(s)^-1 X). This is exact before quantization. After group-wise quantization, a salient channel scaled by s > 1 receives Q(ws)(x/s). Compared with the unscaled error Q(w)x, the compensated error shrinks by roughly 1/s as long as the group step Δ does not grow. The transform therefore gives salient channels finer effective resolution at no extra bits. However, if s is too large the scaled weight becomes the group maximum, Δ grows, and every non-salient weight in that group suffers amplified error. The scale must balance protection of salient channels against harm to ordinary ones.

AWQ captures that balance with a one-parameter search. From a small calibration pass it computes the per-input-channel average activation magnitude s_X. It then considers scales s = s_X^α and searches a scalar α in [0, 1]. For each candidate it scales the weights, applies ordinary group-wise INT3/INT4 quantization, undoes the scale in the stored weight, and scores the real output MSE against the FP16 reference. The best α is chosen with no gradients and no Hessian. An additional per-group clip search trims the worst outliers by shrinking each group's maximum value before rounding. The final stored model remains a regular group-wise low-bit weight tensor, so it packs cleanly into hardware-aligned kernels.

This satisfies all the constraints: no retraining, no backpropagation, no second-order reconstruction, minimal dependence on calibration data, and a uniform hardware-friendly layout. The activation statistic only decides which channels need protection; the equivalence transform supplies that protection without leaving the integer grid.

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
