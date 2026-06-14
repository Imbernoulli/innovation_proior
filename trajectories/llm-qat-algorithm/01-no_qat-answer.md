**Problem.** Low-bit weight quantization of Pythia-1.4B is an irreversible step: snap every weight onto
a signed `B`-bit per-group grid, once. The first rung asks what that costs with *no* algorithm on top —
pure post-training round-to-nearest, the no-calibration lower bound that isolates the format's damage
from any training contribution.

**Key idea.** Per-group symmetric round-to-nearest with the max-abs scale `S = max|w| / qmax` (groups
of 128 columns, `qmin = −2^{B−1}`, `qmax = 2^{B−1}−1`), applied once to the pretrained weights with no
fine-tune. The differentiable straight-through fake-quant is left in place but never exercised: the
training loop is disabled, so the only thing that runs is the real no-grad QDQ at materialization time
and the wrapper's plain-linear forward on the already-quantized weight.

**Why.** The affine map is forced by wanting integer arithmetic to track real arithmetic; `Z = 0`
falls out of zero-centered symmetric weights; max-abs is the smallest no-clip scale and needs no data;
round-to-nearest minimizes one-shot error (stochastic rounding only helps under repeated rounding).
Per-group localizes the two one-shot failure modes (cross-channel range spread, outliers). There is no
repair mechanism — the `S/2` floor is frozen — so this rung must collapse at low bit-widths, motivating
the rest of the ladder.

**Hyperparameters.** `num_steps=0`, `learning_rate=0` (training is a no-op); `group_size=128`;
`1e-12` floor on `max|w|`; LM head (`embed_out`) and activations kept full precision.

```python
# EDITABLE region of custom_qat.py (lines 33-176) — step 1: no_qat (PTQ-only control)

CONFIG_OVERRIDES = {
    "learning_rate": 0.0,
    "num_steps": 0,
    "batch_size": 2,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "warmup_steps": 0,
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
    w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = w_max / qmax
    w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
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
        # PTQ-only: by eval the real QDQ has already replaced linear.weight,
        # so just run a plain linear on the already-quantized weight.
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
