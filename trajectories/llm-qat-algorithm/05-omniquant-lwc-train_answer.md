LSQ closed the ladder's biggest gap and, in doing so, exposed the one seam left to press. At INT4 it landed at 12.17 (degradation $-1.04$), essentially STE's 11.70 — the learnable-scale machinery neither helping nor hurting much when 16 codes are already fine. At INT3, 13.49, within a third of a point of full precision. And at INT2 it delivered 19.50 (degradation 6.29), the first method genuinely usable at two bits, beating the `finetune_then_ptq` control's 94034 by orders of magnitude — learning the per-group step size let the optimizer place the *levels* as well as the weights, clipping outliers and rounding the bulk finely, exactly what the fixed max-abs scale could not do. But the INT2 number sharpens a question I left open. LSQ learns the scale $s$ as a *free, unbounded* parameter, and at four codes the loss surface in $s$ is brutal: a small change in $s$ reassigns large blocks of weights across codes, so the step-size gradient is huge and noisy near transitions. An unbounded $s$ driven by that gradient can drift — overshoot to a scale that clips too aggressively, or undershoot to one that wastes codes — and there is nothing in the parameterization to keep it sane. The diagnosis is specific: LSQ's *degree of freedom* is right (the grid should be learned) but its *parameterization* is too loose. The move is to keep the learnable grid and **bound** it.

I propose **OmniQuant's Learnable Weight Clipping (LWC)** (Shao et al., ICLR 2024): go back to the max-abs scale as the *anchor* — the no-clip cover STE used — and learn a multiplicative *clipping factor* that shrinks it, where the factor is forced into $(0,1)$ by a sigmoid. Per group, take the raw signed extremes $x_{\max} = \max(w_g)$ and $x_{\min} = \min(w_g)$, introduce two learnable per-group parameters `upbound_factor` and `lowbound_factor`, gate them, and clip the extremes:

$$x_{\max} \leftarrow \sigma(\text{upbound})\cdot x_{\max}, \qquad x_{\min} \leftarrow \sigma(\text{lowbound})\cdot x_{\min}.$$

For symmetric signed weight quantization the scale is then $s = \max(|x_{\max}|, |x_{\min}|)/q_{\max}$, clamped to a safe range $[10^{-5}, 10^4]$, and the fake-quant is the straight-through $\mathrm{round}(\mathrm{clip}(w/s, q_{\min}, q_{\max}))\cdot s$. The gradient reaches the bound factors through the differentiable $\sigma\cdot\text{extreme}/q_{\max}$ scale; the round is STE as before. I set the factors' `init_value = 4.0` so that $\sigma(4)\approx 0.982$ — the grid starts essentially at the max-abs cover, identical to STE, and *learns to clip inward* from there — with `CLIPMIN = 1e-5` and per-group factors of shape $(\text{out\_features}, n_\text{groups}, 1)$.

Why this should beat LSQ at INT2 is the whole bet, and the two are close cousins: both learn a per-group grid, but they differ in the *coordinate* they learn it in. LSQ learns $s\in(0,\infty)$ directly; LWC learns $\gamma = \sigma(\text{factor})\in(0,1)$ and sets $s = \gamma\cdot\max|w_g|/q_{\max}$. Three consequences fall out. First, the search space is *bounded*: $\gamma$ cannot exceed 1, so the learned scale can never blow past the max-abs cover — it can only clip inward, exactly the useful direction at low bits (clip the outliers, round the bulk finely), whereas LSQ's scale has no ceiling and can wander above the cover, wasting codes. Second, the parameterization is *anchored*: because $s = \gamma\cdot\max|w_g|$, the scale rides the current weight magnitude, so as the weights move during training the grid moves with them automatically and the factor only has to learn the relative clip — a far gentler quantity to optimize than an absolute scale. Third, the sigmoid *bounds the gradient*: $d\gamma/d\text{factor} = \gamma(1-\gamma)$ saturates as $\gamma\to 0$ or $1$, damping the huge, noisy step-size gradients that destabilize LSQ at four codes — the parameterization itself provides the stability LSQ has no mechanism for. Starting at $\sigma(4)\approx 0.982$ means LWC begins where STE sits and improves monotonically by clipping inward, so it should never be worse than STE while inheriting LSQ's "learn the grid" power with a tighter, gradient-damped parameterization — the cure for exactly the drift I suspect in LSQ's INT2 number.

One thing demands care, because it is where the original form of this clip and this harness part ways. The clip's natural home is a *per-block local reconstruction*: quantize one transformer block, minimize the MSE between its output and the full-precision block's output on a small calibration set, no gradient through the rest of the model — a PTQ-grade budget. This harness does not expose a per-block reconstruction loop; it exposes the *global* QAT loop — full-model forward, cross-entropy on WikiText-2, AdamW over all `requires_grad` parameters for 500 steps. So I do not get that cheap local objective, I get the same end-to-end finetune every other rung uses, and the learnable factors are trained by the global task loss alongside the weights. That is actually a *stronger* training signal than a local reconstruction — the factors are optimized against the very metric I am scored on — so the LWC parameterization should, if anything, do better here than its budget-constrained origin. The one piece I therefore drop is the per-block scheduling; the parameterization itself — sigmoid-gated per-group clip factors, init 4.0 — is exactly the canonical one, layered on top of STE's weight-QAT, both driven by the global loss.

I land it in the edit surface following the same wiring LSQ forced, because the harness's fixed post-training RTN would clobber a learned grid just as it would have clobbered LSQ's scales. So `quantize_dequantize_weight` is again a deliberate no-op `weight.clone()`, and the real quantization lives in the wrapper's `forward`, branching on `self.training`: during training the differentiable LWC fake-quant with the sigmoid-gated factors; at eval a genuine no-grad $\mathrm{round}(\mathrm{clip}(w/s))\cdot s$ on the *learned* clipped grid, recomputing the per-group extremes and scale from the final weights and factors, so evaluation sees the LWC grid and not the max-abs grid. The factors `up_factor` and `low_factor` are registered as `nn.Parameter`s of shape $(\text{out\_features}, n_\text{groups}, 1)$ initialized to 4.0, so the harness's optimizer-over-all-`requires_grad`-params trains them alongside the weights with no extra plumbing. One detail I must get right so the clip is faithful: I compute raw $x_{\max}$/$x_{\min}$ as the per-group *signed* max and min (not max-abs), gate each by its own sigmoid factor, and *then* take $\max(|x_{\max}|, |x_{\min}|)$ for the symmetric scale — keeping both factors and the signed-extreme computation rather than collapsing to a single max-abs factor, which would be a different and less faithful method. Activations stay full precision, the LM head is restored to plain Linear, and the same 500-step schedule is held constant, so the only change from LSQ is the coordinate the grid is learned in.

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
