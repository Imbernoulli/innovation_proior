**Problem.** The `finetune_then_ptq` control collapses at INT2 because its training forward is pure
fp32 — the optimizer never sees the grid, so it cannot place weights that survive rounding. The fix is
to put the rounding *in the forward during training*, so every gradient already accounts for the grid.

**Key idea.** Straight-through-estimator QAT: fake-quantize the weight in the forward with the
per-group symmetric RTN (`w_q = clamp(round(w/S), qmin, qmax)·S`, `S = max|w_g|/qmax`), but on the
backward pass treat the round/clamp as the identity — `w_dq = w + (w_q − w).detach()` is `w_q` forward,
slope-1 backward. The fp32 `linear.weight` is the master copy the optimizer accumulates into; the scale
is recomputed every forward so the grid tracks the weight. After training, the same no-grad RTN
materializes the integer model.

**Why.** The honest derivative of round is zero a.e. and freezes the weight; the identity proxy has the
right sign for one layer and, unlike a bell-shaped surrogate, keeps the gradient alive near transitions
where it is needed most. The scale stays a *fixed* max-abs statistic, so this rung isolates exactly the
gradient contribution — the gap to `finetune_then_ptq` (identical schedule) is pure QAT signal — and
leaves *learning* the step size to the next rung.

**Hyperparameters.** `lr=2e-5`, `num_steps=500`, `batch_size=2`, `grad_accum=4`, `max_grad_norm=1.0`,
`warmup_steps=50`, `weight_decay=0.0`; `group_size=128`; `1e-12` floor on `max|w|`; LM head and
activations full precision.

```python
# EDITABLE region of custom_qat.py (lines 33-176) — step 3: STE QAT

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


def fake_quantize_weight(weight, num_bits, group_size):
    qmin, qmax = _qrange(num_bits)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    w = weight.float().reshape(out_features, -1, group_size)
    # Recompute scale on-the-fly each forward (max-abs / qmax).
    w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = w_max / qmax
    w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
    # Straight-through: forward = quantized, backward = identity.
    w_dq = w + (w_q - w).detach()
    return w_dq.reshape(out_features, in_features).to(weight.dtype)


def fake_quantize_activation(x, num_bits):
    return x


def quantize_dequantize_weight(weight, num_bits, group_size):
    qmin, qmax = _qrange(num_bits)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    with torch.no_grad():
        w = weight.float().reshape(out_features, -1, group_size)
        w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = w_max / qmax
        w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
        return w_q.reshape(out_features, in_features).to(weight.dtype)


class QATWrapper(nn.Module):
    def __init__(self, linear, num_bits, group_size):
        super().__init__()
        self.linear = linear
        self.num_bits = num_bits
        self.group_size = group_size

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        x = fake_quantize_activation(x, self.num_bits)
        w_q = fake_quantize_weight(self.linear.weight, self.num_bits, self.group_size)
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
