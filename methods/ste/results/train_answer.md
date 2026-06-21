I want a unit inside a neural network whose output is a single hard bit — on or off — and I still want to train the whole network end to end by gradient descent. The motivating case is a gating unit for conditional computation: a cheap gater decides, per example, whether a much more expensive block of the network is computed at all, and if those gates produce genuine zeros most of the time, large chunks of the computation can be skipped, so the cost per example collapses from "touch every parameter" to "touch only the active subset." The same hard-decision unit shows up wherever I want binary codes, hard temporal events, or sparse representations. The trouble is the unit itself. A bit-valued output is the indicator of a threshold being crossed: write it as a sigmoid-parameterized Bernoulli, with pre-activation $a = Wx + b$ and $z \sim U[0,1]$, output $h = \mathbf{1}_{z < \mathrm{sigm}(a)}$ so that $P(h{=}1) = \mathrm{sigm}(a)$. The firing *probability* $\mathrm{sigm}(a)$ is a smooth function of $a$, but the realized *bit* is not, and it is the bit that flows downstream. Nudging $a$ never changes which side of the threshold a fixed draw lands on except at the single crossing point, so $\partial h/\partial a = 0$ almost everywhere and is undefined at the jump. The chain rule then gives $\partial L/\partial a = (\partial L/\partial h)\cdot 0 = 0$: a zero factor annihilates everything, and back-propagation transmits no signal to $W$, to $b$, or to anything feeding the unit. This is precisely the wall that pushed the field off hard-threshold neurons and onto sigmoids in the first place.

The existing routes for getting a signal to such a unit all fall short in a specific way. Finite differences perturb one parameter at a time and read off $(L(u + \varepsilon e_i) - L(u - \varepsilon e_i))/2\varepsilon$; it works through any black box, but at $O(N^2)$ it is $N$ times slower than one back-propagation pass and wants $\varepsilon$ small, the wrong regime for all-or-none events. SPSA (Spall 1992) rescues the cost by perturbing all parameters at once with a single vector $z$ and estimating $\partial L/\partial u_i \approx (L(u{+}z) - L(u{-}z))/(2 z_i)$ in two evaluations, but it is unbiased only as $z \to 0$ and *divides* by the perturbation — exactly wrong when the decisions are inherently large, 0 versus 1. The principled route is the likelihood ratio: REINFORCE (Williams 1992) treats the bit as a sampled action and, for any baseline $b$, gives the unbiased estimate $E_h[(R - b)\,\partial \log p_\theta(h)/\partial\theta] = \partial E_h[R]/\partial\theta$, which works for hard outputs because it differentiates the smooth log-probability, not the sample. Specialized to the Bernoulli-logistic unit, $\partial \log g/\partial a = h - \mathrm{sigm}(a)$, so the estimator is $(h_i - \mathrm{sigm}(a_i))\,L$ with a per-unit variance-optimal baseline. It is certified unbiased, but it is high variance, it needs the (possibly global, possibly delayed) loss broadcast to *every* unit, and it throws away the structured local credit assignment that back-propagation hands me for free. Finally one can sidestep the threshold entirely — keep a sigmoid and anneal it toward binary, as in semantic hashing where growing weights saturate the sigmoid — but the hard zeros then only materialize at the *end* of training, which defeats the whole point of conditional computation, whose purpose is to save computation *during* training.

I propose the **straight-through estimator (STE)**, and the idea is to refuse to multiply by zero on the backward pass. The forward pass stays exactly the hard threshold — the true bit, the true zeros, the real sparsity — because that is where the skipped computation comes from; the only place left to intervene is the backward edge. So I replace the threshold's honest zero Jacobian with a nonzero proxy that carries a usable sign. The simplest proxy is the identity: back-propagate through the hard op *as if it had been the identity function*, sending the incoming gradient straight through unchanged,
$$g_a = \frac{\partial L}{\partial h}\qquad\text{(pretend } \partial h/\partial a = 1\text{)}.$$
The forward still computes $h = \mathbf{1}_{z < \mathrm{sigm}(a)}$; only the backward edge lies. The first thing to check is the sign, because a proxy pointing the wrong way is worse than useless. The threshold is monotone increasing in $a$ — raising $a$ raises $\mathrm{sigm}(a)$ raises $P(h{=}1)$ — so the true effect of $a$ on $h$ is positive, and the identity's slope $+1$ matches it. For a single layer the straight-through gradient therefore has the right sign: if downstream wants $h$ larger it pushes $a$ larger, exactly as it should.

It is tempting to use a *less* crude proxy, namely the derivative of the smooth thing the bit samples from, $\mathrm{sigm}(a)(1-\mathrm{sigm}(a))$, multiplying the incoming gradient by that. This looks more principled, but it is worse here, and not by accident. That factor peaks at $a=0$ and decays to nearly zero as $|a|$ grows, so it *attenuates the gradient in saturation* — and a confidently-thresholded, mostly-off gate is exactly the saturated regime I want to live in. The "principled" proxy would zero out the gradient for precisely the most decided units, softly reintroducing the very vanishing I was trying to escape. The plain identity keeps the signal alive in saturation, full slope everywhere, so a saturated-but-wrong gate still gets a full push to flip. The cruder proxy is better *because* it refuses to attenuate where I need signal most.

This is not merely a hack; it has a descent story. Condition on everything downstream of one hard unit and write the two loss values as $L(1)$ and $L(0)$. With $p = \mathrm{sigm}(a)$ the expected loss is $F(a) = p\,L(1) + (1-p)\,L(0)$, so the honest gradient is
$$F'(a) = p(1-p)\,(L(1) - L(0)).$$
Linearly extending the downstream loss across the bit, $L(h) = h\,L(1) + (1-h)\,L(0)$, the gradient arriving at the bit is $\partial L/\partial h = L(1) - L(0)$, and the identity STE sends exactly that back: $g_{\mathrm{ST}} = L(1) - L(0)$. Hence $F'(a) = p(1-p)\,g_{\mathrm{ST}}$, equivalently $g_{\mathrm{ST}} = F'(a)/\big(p(1-p)\big)$ for finite $a$. The factor $p(1-p)$ is strictly positive, so $g_{\mathrm{ST}}$ is a positive coordinate-wise rescaling of the true one-layer gradient: across a layer it is $g_{\mathrm{ST}} = D^{-1}\nabla F$ with $D_{ii} = p_i(1-p_i) > 0$, hence $\nabla F \cdot g_{\mathrm{ST}} = \sum_i (\partial F/\partial a_i)^2 / D_{ii} > 0$ unless the true gradient vanishes, so $-g_{\mathrm{ST}}$ is a descent direction for the expected loss. I should name the honest limit too: "right sign" is a *single-layer* guarantee, and stacking many hard units lets the fabricated $\partial h/\partial a = 1$ factors compound until the backward direction's sign no longer matches the true effect on the loss. So STE is a biased estimator, most trustworthy for one hard (e.g. gating) layer. The trade against REINFORCE is deliberate: REINFORCE is unbiased but high-variance, non-local, and needs the loss broadcast everywhere; STE is biased but has essentially zero added variance (it is deterministic given the backward pass) and is fully local (it reuses the structured back-propagation signal, just routed through one extra edge). For learning, low variance with the right sign tends to beat unbiased with enormous variance. That is the whole bet.

In code STE has two equivalent forms. The explicit one is a custom autograd primitive whose forward applies the hard op and whose backward returns the incoming gradient untouched. The slicker one needs no class: write $h = a + (\mathrm{hard}(a) - a)\texttt{.detach()}$, whose forward *value* is $\mathrm{hard}(a)$ but whose only differentiable term is the bare $a$ with derivative one, so the backward gradient is exactly the identity.

```python
import torch

# Explicit custom autograd primitive: forward = hard op, backward = identity.
class _StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        return (a > 0).float()        # the true hard decision (binary threshold / sign / round ...)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output            # identity: dL/da = dL/dh

# The detach trick (no custom class): forward value = hard(a), backward gradient = 1.
def straight_through(a, hard):
    return a + (hard(a) - a).detach()
```

The same backward convention drives the canonical heavy use, quantization-aware training, where I force weights to a small discrete set so matmuls become cheap integer ops. The quantizer's $\mathrm{round}(\cdot)$ and $\mathrm{clamp}(\cdot)$ are both non-differentiable — round is a staircase with slope $0$ a.e., clamp is flat outside its range — so I wrap the whole quantize-dequantize in straight-through. But there is a subtlety the single-bit gate hid: I cannot store the learned weight at low precision, because each weight must absorb a long run of tiny, noisy SGD steps, and coarse storage would round every infinitesimal update to zero so nothing would ever move. The load-bearing structure is therefore a **full-precision shadow weight**: I quantize it on the fly each forward pass, use the quantized value in the matmul, let straight-through carry the gradient back to the shadow, and accumulate the updates in the shadow at full precision. The quantized value is recomputed every step. For $B$-bit per-group symmetric signed quantization the integer codes lie in $[q_\min, q_\max] = [-2^{B-1}, 2^{B-1}-1]$; I group the input dimension into blocks so each gets its own scale, set $s = \max|w| / q_\max$ per (row, group) floored away from zero, and form the fake-quant $w_q = \mathrm{clamp}(\mathrm{round}(w/s), q_\min, q_\max)\cdot s$ — recomputing the scale from the shadow every forward so the bounded code grid tracks the current group. The quantizer is deliberately plain (round-to-nearest, max-abs scale, no learned step size): the contribution is the *gradient*, the minimal thing that lets the shadow reorganize to be quantization-friendly instead of being frozen by a zero Jacobian. After training, a no-grad version of the same quantize-dequantize materializes the deployed integer model.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Differentiable fake-quant for the QAT forward: per-(row, group) symmetric round/clamp,
    with straight-through so the gradient reaches the full-precision shadow weight."""
    qmin, qmax = _qrange(num_bits)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    w = weight.float().reshape(out_features, -1, group_size)
    w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)   # floor avoids /0
    scale = w_max / qmax
    # round + clamp are non-differentiable; the detach below makes the backward an identity
    w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
    w_dq = w + (w_q - w).detach()                                 # straight-through
    return w_dq.reshape(out_features, in_features).to(weight.dtype)


def fake_quantize_activation(x, num_bits):
    return x                                                      # weight-only QAT: identity


def quantize_dequantize_weight(weight, num_bits, group_size):
    """Real (no-grad) quantize-dequantize used after training to materialize the integer model."""
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
        self.linear = linear                                     # full-precision shadow weight
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
    # keep the output projection at full precision
    for head_attr in ("lm_head", "embed_out"):
        head = getattr(model, head_attr, None)
        if isinstance(head, QATWrapper):
            setattr(model, head_attr, head.linear)
    return model
```
