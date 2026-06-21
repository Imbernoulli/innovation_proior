The full-precision control settled the confound and drew a clean line. `finetune_then_ptq` beat blind `no_qat` everywhere the grid left room — 12.61 at INT4 (degradation $-0.59$, below the 13.2033 baseline), 31.41 at INT3 (down from 42.54) — but it stayed catastrophically collapsed at INT2, 94034, barely distinguishable from `no_qat`'s 104422 on any scale that matters. That INT2 result is the whole point of the control: a finetune that is *blind to the grid* cannot rescue two-bit quantization no matter how much it lowers the loss it rounds from, because four codes destroy the weights through a rounding the finetune never sees. The reason points straight at the fix. The control's training forward is pure fp32 — `fake_quantize_weight` is the identity — so every gradient it takes is the gradient of the *full-precision* function, which has no term saying "sit where rounding with step $S$ is cheap." If I want the optimizer to place weights that survive rounding, the rounding has to be *in the forward pass during training*.

I propose **STE QAT**: fake-quantize the weight in the forward with the per-group symmetric RTN, so the loss is computed on the quantize-dequantized weights and every gradient already accounts for the grid, while a straight-through estimator carries that gradient back through the non-differentiable round. Putting the grid in the forward immediately hits the wall that made the control take the easy way out. The fake-quant is $\hat w = S\cdot\mathrm{clamp}(\mathrm{round}(w/S), q_{\min}, q_{\max})$, and $\mathrm{round}$ is a staircase — slope zero almost everywhere, undefined at the half-integers — while $\mathrm{clamp}$ is flat outside its range. Differentiate this honestly and the chain rule multiplies by that zero Jacobian, annihilating the gradient: nothing reaches $w$, the weight is frozen. This is the same wall that pushed the field off hard-threshold units onto sigmoids decades ago — a constant-by-parts graph kills gradient learning — so the honest derivative is useless and I need a nonzero proxy on the backward pass while keeping the forward exactly the hard quantizer.

The proxy is the simplest possible one, the straight-through estimator (Bengio et al., 2013): back-propagate through the quantize-dequantize *as if it had been the identity*, so the gradient arriving at the quantized output is sent unchanged to the underlying weight, $\partial L/\partial w = \partial L/\partial \hat w$. In code this is the single line $w_{dq} = w + (w_q - w)\texttt{.detach()}$ — equal to $w_q$ in the forward, slope-one in the backward. Let me sanity-check the sign, because a proxy that points the wrong way is worse than useless. The quantizer is monotone increasing in $w$: raise $w$, you raise $\mathrm{round}(w/S)$, you raise $\hat w$ within the clip range. So the true sign of how $\hat w$ responds to $w$ is positive, and the identity proxy assigns slope $+1$, also positive. For a single quantized layer the straight-through gradient therefore has the right sign — if downstream wants $\hat w$ larger, it pushes $w$ larger. It is biased, the magnitude is fabricated, but the direction is correct where it matters, and that is what lets the weight reorganize to be quantization-friendly instead of frozen by a zero Jacobian.

A less crude proxy looks tempting — back-propagate as if the round were a smooth surrogate, multiplying the incoming gradient by a bell-shaped factor that peaks at a bin center and decays toward the transitions — but it is the wrong move, because it *attenuates the gradient exactly where I need it most*. A weight sitting confidently in the middle of a code cell is already well-quantized; the weights I need to move are the ones near a transition, where a small nudge flips which code they round to and where the rounding error is largest. The plain identity proxy keeps the signal alive everywhere — slope one regardless of position in the cell — so a weight that is rounding badly still gets a full push. The cruder proxy is the better one, and not by accident: it refuses to attenuate where the useful signal lives.

One piece of bookkeeping the identity proxy hides is load-bearing. I cannot store the weight at low precision, because each weight has to absorb a long run of tiny, noisy gradient steps, and averaging those out needs real resolution — coarse storage would round every infinitesimal step to zero and nothing would move. So I keep a *full-precision master weight*, quantize it on the fly each forward, use the quantized value in the matmul with straight-through carrying the gradient back to the master, and let the optimizer accumulate updates in the master. In this edit surface that master *is* `self.linear.weight`, the fp32 parameter the harness's AdamW already optimizes, and the scale is recomputed from it every forward so the per-group grid tracks the current weight. The fixed loop makes this automatic — it builds the optimizer over all `requires_grad` parameters and runs the finetune through the wrapper's fake-quant forward — and at the end the harness's no-grad `quantize_dequantize_weight` materializes the genuine integer model, with no straight-through needed because nothing differentiates anymore.

What I deliberately hold fixed defines what this rung isolates. The scale here is a fixed max-abs statistic, $S = \max|w_g| / q_{\max}$, recomputed each forward but *not* learned. Keeping it plain isolates exactly what straight-through buys on its own — the ability to send a gradient through the round so the master weight reorganizes to be quantization-friendly — and leaves the obvious next idea, *learning* the step size to place the levels more cleverly, for the rung after this. Against the control the only structural change is that the grid is now in the forward, so the gap to `finetune_then_ptq` (identical schedule) is by construction the pure QAT contribution. I can already feel where the construction will hit its own ceiling: at INT2, $q_{\max}=1$, so $S = \max|w_g|$ per group and straight-through can move the *weights* to fit those levels but cannot move the *levels* to fit the weights — the single fixed knob places them crudely, so I expect INT2 rescued from catastrophe but still well above full precision, with the remaining error living in the grid placement that this rung holds fixed.

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
