**Problem.** LSQ proved that *learning* the per-group grid is the right move (INT2 6.29 degradation vs
STE's 59.35), but it learns the scale as a *free, unbounded* parameter. At INT2 the four-code
step-size gradient is huge and noisy, so the free scale can drift, and a slice of LSQ's residual INT2
error is likely that drift rather than the loss-optimal grid.

**Key idea.** OmniQuant's Learnable Weight Clipping (Shao et al., ICLR 2024, arXiv:2308.13137): keep
the max-abs cover as the anchor and learn a sigmoid-gated *clipping factor* in (0,1) that shrinks it.
Per group, gate the signed extremes — `xmax ← sigmoid(upbound)·max(w_g)`, `xmin ← sigmoid(lowbound)·
min(w_g)` — then `s = max(|xmax|,|xmin|)/qmax` (clamped `[1e-5,1e4]`), and STE fake-quant
`round(clip(w/s,qmin,qmax))·s`. Factors init to 4.0 (`sigmoid(4)≈0.982`, so the grid starts at the
max-abs cover ≈ STE and learns to clip inward).

**Why it should beat LSQ at INT2.** Same learnable-grid power, tighter coordinate: γ∈(0,1) is *bounded*
(can only clip inward, the useful direction at low bits), *weight-anchored* (`s = γ·max|w_g|` rides the
weight magnitude, so only the relative clip is learned), and the sigmoid *damps the gradient*
(`dγ/dfactor = γ(1−γ)` saturates), curing exactly LSQ's free-scale drift. Wired into the task like LSQ:
`quantize_dequantize_weight` is a no-op clone, the wrapper does real LWC-grid QDQ in its eval branch,
factors are per-group `nn.Parameter`s the harness optimizer trains.

**Hyperparameters.** Same schedule as LSQ/STE: `lr=2e-5`, `num_steps=500`, `batch_size=2`,
`grad_accum=4`, `max_grad_norm=1.0`, `warmup_steps=50`, `weight_decay=0.0`; `group_size=128`;
factor init 4.0; `CLIPMIN=1e-5`; weight-only; LM head full precision.

```python
# EDITABLE region of custom_qat.py (lines 33-176) — finale: OmniQuant LWC (learnable weight clipping)

CONFIG_OVERRIDES = {
    "learning_rate": 2e-5,
    "num_steps": 500,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "warmup_steps": 50,
    "weight_decay": 0.0,
}

_LWC_INIT = 4.0       # init_value for the learnable clip factors; sigmoid(4) ~ 0.982
_CLIPMIN = 1e-5       # floor on the learned scale (OmniQuant CLIPMIN)


def _qrange(num_bits):
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))
    return qmin, qmax


def _round_ste(x):
    # straight-through round: forward = round, backward = identity
    return (x.round() - x).detach() + x


def _lwc_scale(w, up_factor, low_factor, qmax):
    # w: (out, n_groups, group_size); factors: (out, n_groups, 1).
    # Gate the per-group signed extremes by sigmoid(factor) in (0,1), then
    # take the symmetric abs-max scale.  Differentiable in the factors.
    xmax = w.amax(dim=-1, keepdim=True)
    xmin = w.amin(dim=-1, keepdim=True)
    xmax = torch.sigmoid(up_factor) * xmax
    xmin = torch.sigmoid(low_factor) * xmin
    abs_max = torch.max(xmax.abs(), xmin.abs())
    scale = (abs_max / qmax).clamp(min=_CLIPMIN, max=1e4)
    return scale


def fake_quantize_weight(weight, num_bits, group_size, up_factor=None, low_factor=None):
    qmin, qmax = _qrange(num_bits)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    w = weight.float().reshape(out_features, -1, group_size)
    if up_factor is None:
        # Prepare-time call with no learnable factors -- fall back to STE max-abs.
        w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        s = w_max / qmax
        w_q = torch.clamp(torch.round(w / s), qmin, qmax) * s
        w_dq = w + (w_q - w).detach()
    else:
        s = _lwc_scale(w, up_factor, low_factor, qmax)
        w_q = torch.clamp(_round_ste(w / s), qmin, qmax) * s
        w_dq = w_q
    return w_dq.reshape(out_features, in_features).to(weight.dtype)


def fake_quantize_activation(x, num_bits):
    return x


def quantize_dequantize_weight(weight, num_bits, group_size):
    # LWC stores learned clip factors on the wrapper; the fixed-region RTN
    # would clobber the learned grid, so leave the float weight untouched and
    # let the wrapper apply LWC-grid QDQ in eval mode below.
    return weight.clone()


class QATWrapper(nn.Module):
    def __init__(self, linear, num_bits, group_size):
        super().__init__()
        self.linear = linear
        self.num_bits = num_bits
        self.group_size = group_size
        out_features, in_features = linear.weight.shape
        n_groups = in_features // group_size
        shape = (out_features, n_groups, 1)
        # Per-group learnable clip factors, init so sigmoid(init) ~ 1 (== STE start).
        self.up_factor = nn.Parameter(torch.full(shape, _LWC_INIT, dtype=linear.weight.dtype))
        self.low_factor = nn.Parameter(torch.full(shape, _LWC_INIT, dtype=linear.weight.dtype))

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        x = fake_quantize_activation(x, self.num_bits)
        if self.training:
            w_q = fake_quantize_weight(
                self.linear.weight, self.num_bits, self.group_size,
                up_factor=self.up_factor.float(), low_factor=self.low_factor.float(),
            )
        else:
            # Eval: real no-grad quantize-dequantize on the learned LWC grid.
            qmin, qmax = _qrange(self.num_bits)
            with torch.no_grad():
                w = self.linear.weight.float().reshape(
                    self.linear.weight.shape[0], -1, self.group_size
                )
                s = _lwc_scale(w, self.up_factor.float(), self.low_factor.float(), qmax)
                w_q = torch.clamp(torch.round(w / s), qmin, qmax) * s
                w_q = w_q.reshape_as(self.linear.weight).to(self.linear.weight.dtype)
        return F.linear(x, w_q, self.linear.bias)


def prepare_qat_model(model, num_bits, group_size):
    from transformers.pytorch_utils import Conv1D

    def _replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name, QATWrapper(child, num_bits=num_bits, group_size=group_size))
            elif isinstance(child, Conv1D):
                in_f, out_f = child.weight.shape
                lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
                                device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    lin.weight.copy_(child.weight.t().contiguous())
                    if child.bias is not None:
                        lin.bias.copy_(child.bias)
                setattr(parent, name, QATWrapper(lin, num_bits=num_bits, group_size=group_size))
            else:
                _replace(child)

    _replace(model)
    for head_attr in ("lm_head", "embed_out"):
        head = getattr(model, head_attr, None)
        if isinstance(head, QATWrapper):
            setattr(model, head_attr, head.linear)
    return model
```
