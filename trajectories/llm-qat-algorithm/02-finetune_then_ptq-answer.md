**Problem.** Because QAT here finetunes and evaluates on the same domain (WikiText-2), a QAT method
that beats `no_qat` cannot be credited to the *quantization-aware* part until the plain in-domain
finetune effect is subtracted out. This rung is the control that measures that effect: a
full-precision finetune followed by the identical RTN PTQ, with *no* QAT signal in the forward.

**Key idea.** Run the standard text-modeling finetune in pure fp32 — `fake_quantize_weight` is the
identity, so the grid is never simulated during training — then apply the same no-grad per-group
symmetric RTN as `no_qat` once at the end. The finetune buys domain adaptation (lower test
cross-entropy) and a flatter loss basin (where rounding hurts less), but is blind to the grid, so it
cannot target the cross-channel range spread or outliers and still pays the `S/2` floor.

**Why.** Weight-space distance is the wrong objective; the right one is the task loss, which the
pretrained weights do not minimize. A plain fp finetune lowers that loss without seeing the grid; the
gap between this control and a real QAT method (which puts the grid in the forward) is the genuine QAT
contribution, with the finetune held constant by using the identical schedule.

**Hyperparameters.** Same schedule as every QAT rung: `lr=2e-5`, `num_steps=500`, `batch_size=2`,
`grad_accum=4`, `max_grad_norm=1.0`, `warmup_steps=50`, `weight_decay=0.0`; `group_size=128`;
`1e-12` floor on `max|w|`; LM head and activations full precision.

```python
# EDITABLE region of custom_qat.py (lines 33-176) — step 2: finetune_then_ptq (control)

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
    # Identity: no fake quant in forward -- pure FP finetune.
    return weight


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
        # Pure FP forward during training (no fake quant).  At eval the real
        # QDQ has already been applied to linear.weight, so this still
        # produces the genuine INT-N output.
        return F.linear(x, self.linear.weight, self.linear.bias)


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
