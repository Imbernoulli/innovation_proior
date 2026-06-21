The whole task is to design a training-side algorithm, but that algorithm bolts onto a fixed format — per-group symmetric round-to-nearest — and I cannot reason about what training buys until I know exactly what the format costs on its own. So I begin at the honest floor: pure post-training quantization, the irreversible rounding applied once to the pretrained Pythia-1.4B weights with *nothing* trained on top. This first rung is deliberately a do-nothing control. Its value is precisely that it isolates the damage of the format from the contribution of any algorithm, and the error it leaves behind is exactly what every later rung will be trying to claw back.

I propose the no-QAT control: **per-group symmetric round-to-nearest with a max-abs scale, applied once, with the finetune loop switched off.** Let me lay out why the format has the shape it does, because that shape is the substrate the whole ladder stands on. An integer code $q$ is by itself just an index; it becomes a number only through a rule $q \mapsto r$. A linear layer is a matrix multiply — sums of products of weights and activations — and I want that accumulation to run on the codes, converting back only at the end, so products and sums of codes must track products and sums of reals. The only map for which "multiply the reals" reduces to "do integer work on the codes plus a cheap fix-up" is an *affine* one, $r = A\,q + B$; anything nonlinear and $g(q_1)\,g(q_2)$ stops being expressible through $q_1 q_2$ plus corrections, and the reason for using integers evaporates. So the map is affine, forced rather than chosen.

The additive term needs care. Attention and MLP plumbing pad and zero things, and in the quantized world a padded position carries whatever real value its code denotes; if real $0$ is not exactly representable, every padded slot contributes the same small nonzero value — a *bias*, the one error that compounds across depth instead of averaging out. So I reparameterize the same affine map as $r = S\,(q - Z)$, a positive real step size $S$ and an integer zero-point $Z$ that maps to real $0$ by construction. A trained weight tensor is roughly zero-centered and symmetric, so the grid is centered at zero and $Z = 0$ falls out, collapsing the map to $r = S\,q$. This is not just tidy: in the integer matmul the general scheme expands $(q_1 - Z_1)(q_2 - Z_2)$ into the core $q_1 q_2$ plus zero-point cross terms I would have to subtract off, and $Z = 0$ on the weights kills every weight cross term, leaving the leanest kernel — the $O(N^3)$ accumulation on small integers with only a per-output scalar $S_1 S_2 / S_3$ surviving in floating point.

That leaves a single knob, the scale $S$. With no appetite for tuning the range against data — this is the no-calibration control — the honest choice is the max-magnitude one. In the signed code range $q_{\min} = -2^{B-1}$, $q_{\max} = 2^{B-1}-1$, tie the scale to the positive endpoint, $S = \max|w| / q_{\max}$: the smallest step that covers the weights without clipping. This is the crux the whole ladder will revisit — a *smaller* $S$ would round the bulk more finely but clip the few weights beyond the tighter range to the rail, trading rounding error for clipping error, a balance that requires looking at data. The max-abs scale is exactly the choice that needs *none*. I pair it with round-to-nearest because for a one-shot conversion it minimizes each element's error, $|\text{error}| \le S/2$, with residual variance about $S^2/12$. I deliberately do not reach for stochastic rounding (Gupta et al., 2015): its unbiasedness only pays when the *same* value is rounded repeatedly so a per-step bias accumulates, and PTQ rounds each weight exactly once, so it would buy variance for nothing.

The one real refinement is to make the scale per-group rather than per-tensor. A single tensor-wide $S$ is wrecked by the two known one-shot failure modes: weight ranges that differ over a hundredfold across output channels, so a narrow channel gets a grid far too coarse for its own magnitude; and lone outlier weights that inflate $\max|w|$ for the whole tensor and coarsen everyone. Partitioning each row into contiguous blocks of $\text{group\_size}=128$ columns, each with its own $\max|w|$ and its own $S$, localizes both — an outlier only coarsens its own 128 weights — for the negligible cost of one extra scale per group, with a $10^{-12}$ floor on $\max|w_g|$ so an all-zero group does not divide by zero.

What makes this rung a *control* and not a method is that the conversion is blind: it looks only at the weights, never at the data or the loss, and once it rounds, the error is frozen. The floor is $S/2$, and at INT2, $q_{\max}=1$, so $S = \max|w_g|$ and the floor is half the largest weight in the group; the signed container has four codes $\{-2,-1,0,1\}$ but with weights inside $[-\max|w_g|, \max|w_g|]$ the in-range reconstruction points are spaced a whole group-max apart — the extra negative code does not make the cells finer. Grouping localizes outliers and range spread but does nothing to this underlying coarseness. There is no mechanism here to *repair* the error, which is exactly why every later rung exists: to let the weights move *before* the rounding, which needs the loss, the gradients, and the optimization this control forbids.

Concretely, the edit makes the harness do nothing where it counts. I set `num_steps = 0` and `learning_rate = 0`, which turns the AdamW loop into a no-op and leaves the pretrained fp32 weights untouched; then the fixed post-step does the only real work, calling the no-grad `quantize_dequantize_weight` once on every wrapped linear to snap the weights to their per-group integer grid, and evaluates. Because no training runs, the wrapper's `forward` needs no fake-quant inside any loop — by eval time the weight has already been overwritten by its dequantized version, so the wrapper is a bare `F.linear` on `self.linear.weight`. The differentiable `fake_quantize_weight` is left present and correct but is never exercised, since zero gradients are ever taken. Activations stay full precision (weight-only), and the LM head (`embed_out` for this GPTNeoX backbone) is restored to a plain Linear so the accuracy-critical output projection escapes quantization.

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
