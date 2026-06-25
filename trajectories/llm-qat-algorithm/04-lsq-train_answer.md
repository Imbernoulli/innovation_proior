STE QAT drew the next problem with a sharp edge. At INT4 it reached 11.70 (degradation $-1.50$), essentially matching the control and sitting below full precision. At INT3 the first real separation appeared: 13.75, within a hair of the 13.2033 baseline, against the grid-blind control's 31.41 — that gap is the pure QAT signal I built the control to isolate, and it is large. But INT2 is where the construction shows its ceiling. STE brought INT2 from the control's 94034 down to 72.55 — a genuine rescue, five digits to two — yet 72.55 against a 13.20 baseline is a degradation of 59.35, alive but nowhere near usable. And I know exactly why, because I left the reason in place deliberately: STE trains the weights against a grid whose step size is *fixed* at $S = \max|w_g|/q_{\max}$ per group. At four codes $q_{\max}=1$, so $S = \max|w_g|$ and the in-range levels are spaced a whole group-max apart; straight-through can move the *weights* to fit those levels but cannot move the *levels* to fit the weights. The leftover INT2 error is not in the weights anymore — STE already optimized those against the grid — it is in the *grid placement*, which STE holds fixed.

I propose **LSQ, Learned Step Size Quantization**: make each per-group scale $s$ a trainable parameter, learned jointly with the weights against the task loss, so the optimizer places both the weights *and* the levels. Why this should help when the weights are already optimized for the fixed step is that the max-abs scale is a *statistic*, not an optimum — the smallest step that covers the group without clipping, a no-data choice — and at four codes the no-clip constraint is exactly the wrong thing to respect. Setting $s = \max|w_g|$ lets one or two outliers define the spacing for all 128 weights, and the bulk, which are much smaller, get snapped to a grid far too coarse for them. A *smaller* step rounds the bulk more finely at the cost of clipping the outliers to the rail — and at the task loss, clipping a couple of outliers can be far cheaper than coarsening 126 ordinary weights. The right step balances rounding error against clipping error, and that balance depends on the weight distribution *and* on the loss, which only a trained scale can find.

The obstacle that made STE necessary returns, but now I must push a gradient not only to the weight but to the *scale*. The quantizer is $\hat w = \mathrm{round}(\mathrm{clip}(w/s, q_{\min}, q_{\max}))\cdot s$. The round is flat almost everywhere, so the path from $s$ through $w/s$ into the integer code has zero ordinary derivative; ordinary backprop sees the final multiply by $s$ but misses how $s$ moves values toward or away from bin transitions. The fix for the round is the same straight-through identity I already use — but I must be careful: STE applies only to the *round node*, while the divide-by-$s$, the clip, and the multiply-by-$s$ I should differentiate honestly. Cutting corners here is what costs accuracy, so I do the calculus. In the interior, $-q_{\min} < w/s < q_{\max}$, the quantizer is $\hat w = \mathrm{round}(w/s)\cdot s$, and the product rule gives $\partial\hat w/\partial s = [\partial\,\mathrm{round}(w/s)/\partial s]\cdot s + \mathrm{round}(w/s)$. STE treats round as identity for differentiation, so $\mathrm{round}(w/s)\approx w/s$ and $\partial(w/s)/\partial s = -w/s^2$; times $s$ that first term is $-w/s$, and the second is $\mathrm{round}(w/s)$, leaving

$$\frac{\partial \hat w}{\partial s} \;=\; \mathrm{round}(w/s) - w/s$$

in the interior — the negative signed residual between $w/s$ and the integer it rounds to. This is exactly the sensitivity a fixed scale throws away: the gradient to the step size is largest when $w/s$ sits near a bin transition (where a small nudge to $s$ flips the code, a big jump in $\hat w$) and vanishes when $w/s$ sits on a level. In the clipped regions it is simpler — if $w/s \le q_{\min}$ the code pins at $q_{\min}$, so $\hat w = q_{\min} s$ and $\partial\hat w/\partial s = q_{\min}$; symmetrically $q_{\max}$ above. And the weight gradient is the plain STE: $\partial\hat w/\partial w = 1$ inside the clip range, $0$ outside, so clipped weights get no weight gradient. I implement this as a custom `torch.autograd.Function` whose `backward` returns the in-range-masked `grad_out` for $w$ and the $(\text{below}\cdot q_{\min} + \text{above}\cdot q_{\max} + \text{inside}\cdot(\mathrm{round}(w/s)-w/s))\cdot\text{grad\_out}$ sum for $s$.

The step-size gradient needs one more thing, and here I read this task's implementation rather than import the generic recipe. A single scalar $s$ per group is optimized by the same AdamW, same global learning rate, as the millions of weights, and training behaves well only when the ratio of update magnitude to parameter magnitude is balanced across parameters. The step-size gradient sums over the elements it touches, so it scales like the square root of that count times the typical per-element gradient — out of balance with a single weight's gradient by a factor that grows with the count and with $q_{\max}$. The cure is a multiplier $g_{\text{scale}} = 1/\sqrt{N\cdot q_{\max}}$ that cancels the imbalance, and the diff I honor exactly is that this task computes $N = w.\texttt{numel()}$ — the *whole weight tensor's* element count — not the per-group count, applying one $g_{\text{scale}}$ to every group's scale gradient. So this is a single tensor-level normalizer shared across all groups of the layer, coarser than the generic per-group $1/\sqrt{N_W\cdot q_{\max}}$, but it still brings the scale gradients into a stable band relative to the weights, and I land it as the harness has it. I likewise follow the task's initialization: each per-group scale starts at $2\,\overline{|W_g|}/\sqrt{q_{\max}}$, clamped to $10^{-8}$, with shape $(\text{out\_features}, n_\text{groups}, 1)$ so it broadcasts over the 128 columns of each group.

A second place I follow this harness and not the generic recipe is the eval-time materialization. The fixed post-training step calls `quantize_dequantize_weight` and overwrites every wrapped weight with its return value — but a *max-abs* RTN there would clobber the learned LSQ scales, snapping the weights back onto the very grid I trained away from. So here `quantize_dequantize_weight` is a deliberate *no-op*, `return weight.clone()`, leaving the fp32 weights untouched, and the real quantization lives in the wrapper's `forward`, which branches on `self.training`: during training it calls the differentiable LSQ fake-quant with the learnable scale; at eval it performs a genuine no-grad $\mathrm{round}(\mathrm{clip}(w/s))\cdot s$ on the *learned* LSQ grid. That is how evaluation still sees a properly quantized model even though the harness's RTN was neutralized — the weights are snapped to the LSQ grid, not the max-abs grid. Everything else — the 500-step schedule, the wrap logic, the full-precision head and activations — is held constant from STE, so the gap to STE is by construction exactly what *learning the grid* buys over training the weights against a fixed grid. The one seam this leaves open is whether a free, unbounded $s$ trained by SGD is *stable* enough at four codes — which is exactly the question the next rung presses on.

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
    """LSQ quantize-dequantize with the step-size gradient derived above."""

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
        # Gradient w.r.t. s: the LSQ step-size gradient.
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
