**Problem.** STE QAT trains the weights against a grid whose step size is *fixed* at `max|w_g|/qmax`,
so at INT2 (four codes, spacing = group max) one or two outliers define the spacing for all 128
weights and the bulk is coarsened — STE rescued INT2 to 72.55 but no further. The leftover error is in
the *grid placement*, which STE holds fixed.

**Key idea.** Learned Step Size Quantization: make each per-group scale `s` a trainable parameter,
quantize with `round(clip(w/s, qmin, qmax))·s`, and push a gradient to `s` via a custom autograd
function — STE for the round, honest calculus elsewhere — giving the interior step-size gradient
`round(w/s) − w/s` (clip endpoints `qmin`/`qmax` outside the range), scaled by `g = 1/sqrt(N·qmax)`
with `N` the full tensor numel. Scales init to `2·|W_g|.mean()/sqrt(qmax)`.

**Why (and the task-specific wiring).** A trained scale balances rounding vs clipping error against the
loss, which the no-data max-abs statistic cannot. The harness's post-training RTN would clobber the
learned scales, so `quantize_dequantize_weight` is a deliberate no-op (`weight.clone()`) and the
wrapper does real LSQ-grid QDQ in its `if not self.training` eval branch — evaluation sees the *learned*
grid, not the max-abs grid. `g_scale` uses `w.numel()` (whole tensor), one normalizer per layer.

**Hyperparameters.** Same schedule as STE: `lr=2e-5`, `num_steps=500`, `batch_size=2`, `grad_accum=4`,
`max_grad_norm=1.0`, `warmup_steps=50`, `weight_decay=0.0`; `group_size=128`; scale init
`2·mean|W_g|/sqrt(qmax)` clamped `1e-8`; weight-only; LM head full precision.

```python
# EDITABLE region of custom_qat.py (lines 33-176) — step 4: LSQ (learned step size)

CONFIG_OVERRIDES = {
    "learning_rate": 2e-5,
    "num_steps": 500,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "warmup_steps": 50,
    "weight_decay": 0.0,
}


def _qrange(num_bits):
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))
    return qmin, qmax


class _LSQQuant(torch.autograd.Function):
    """LSQ quantize-dequantize with the learned step-size gradient."""

    @staticmethod
    def forward(ctx, w, scale, qmin, qmax, g_scale):
        # w: (out, n_groups, group_size); scale: (out, n_groups, 1) broadcastable.
        w_div = w / scale
        w_clip = torch.clamp(w_div, qmin, qmax)
        w_round = torch.round(w_clip)
        ctx.save_for_backward(w_div, scale)
        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.g_scale = g_scale
        return w_round * scale

    @staticmethod
    def backward(ctx, grad_out):
        w_div, scale = ctx.saved_tensors
        qmin, qmax, g = ctx.qmin, ctx.qmax, ctx.g_scale
        # Gradient w.r.t. w: pass-through inside the clip range.
        in_range = (w_div > qmin) & (w_div < qmax)
        grad_w = torch.where(in_range, grad_out, torch.zeros_like(grad_out))
        # Gradient w.r.t. s: the learned step-size gradient.
        below = (w_div <= qmin).float() * float(qmin)
        above = (w_div >= qmax).float() * float(qmax)
        inside = in_range.float() * (torch.round(w_div) - w_div)
        grad_s_per_elem = (below + above + inside) * grad_out
        grad_s = grad_s_per_elem.sum(dim=-1, keepdim=True) * g
        return grad_w, grad_s, None, None, None


def fake_quantize_weight(weight, num_bits, group_size, scale=None):
    qmin, qmax = _qrange(num_bits)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    w = weight.float().reshape(out_features, -1, group_size)
    if scale is None:
        # No learnable scale supplied (prepare-time call) -- fall back to STE.
        w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        s = w_max / qmax
        w_q = torch.clamp(torch.round(w / s), qmin, qmax) * s
        w_dq = w + (w_q - w).detach()
    else:
        n_elem = w.numel()
        g_scale = 1.0 / max(1.0, math.sqrt(n_elem * qmax))
        w_dq = _LSQQuant.apply(w, scale, qmin, qmax, g_scale)
    return w_dq.reshape(out_features, in_features).to(weight.dtype)


def fake_quantize_activation(x, num_bits):
    return x


def quantize_dequantize_weight(weight, num_bits, group_size):
    # LSQ stores learned scales on the wrapper; the fixed-region
    # apply_real_quantization would clobber them with a max-abs RTN, so we
    # return the float weight unchanged and let the wrapper apply LSQ-grid
    # QDQ in eval mode below.
    return weight.clone()


class QATWrapper(nn.Module):
    def __init__(self, linear, num_bits, group_size):
        super().__init__()
        self.linear = linear
        self.num_bits = num_bits
        self.group_size = group_size
        qmin, qmax = _qrange(num_bits)
        out_features, in_features = linear.weight.shape
        n_groups = in_features // group_size
        # LSQ initial scale: 2 * |W|.mean() / sqrt(qmax).
        with torch.no_grad():
            w = linear.weight.float().reshape(out_features, n_groups, group_size)
            init = 2.0 * w.abs().mean(dim=-1, keepdim=True) / max(1.0, math.sqrt(qmax))
            init = init.clamp(min=1e-8)
        # Shape (out_features, n_groups, 1) so it broadcasts over group_size.
        self.lsq_scale = nn.Parameter(init.to(linear.weight.dtype))

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
                scale=self.lsq_scale.float(),
            )
        else:
            # Eval: produce a *real* quantize-dequantize on the LSQ grid.
            qmin, qmax = _qrange(self.num_bits)
            with torch.no_grad():
                w = self.linear.weight.float().reshape(
                    self.linear.weight.shape[0], -1, self.group_size
                )
                s = self.lsq_scale.float()
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
