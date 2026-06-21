We want deep networks to run inference cheaply by carrying their convolution and fully-connected layers in low-precision integers — 2, 3, or 4 bits for both weights and activations — so that the heavy matrix multiplies become low-precision integer operations. A uniform quantizer maps a continuous value $v$ to a small set of integer levels by dividing by a per-layer step size $s$, clipping, and rounding: $\bar v = \mathrm{round}(\mathrm{clip}(v/s,\,-Q_N,\,Q_P))$ and the dequantized value is $\hat v = \bar v \cdot s$, with $Q_N=0,\,Q_P=2^b-1$ for unsigned activations and $Q_N=2^{b-1},\,Q_P=2^{b-1}-1$ for signed weights. At 2–4 bits there are only a handful of representable levels per layer, so *where* those levels land matters enormously, and where they land is governed entirely by one number per layer, the step size. Set it too small and the tails get clipped away; set it too large and the few precious levels are wasted on coarse spacing. The honest answer is not to set $s$ by hand or from a fixed statistic but to *learn* it, jointly with the weights, against the actual task loss.

The obstacle is that the round is flat almost everywhere, so the path from $s$ through $v/s$ into the integer code has zero ordinary derivative; backprop sees the final multiply by $s$ but misses how $s$ moves values toward or away from bin transitions. The prior step-size and clip-learning methods cut corners exactly here, and it cost them. PACT learns a clipping value but, by removing the round from its forward approximation and algebraically cancelling terms, ends up with $\partial\hat v/\partial(\text{param})=0$ for $v$ strictly inside the quantized range — no gradient signal at all from the interior transitions. QIL learns a transformation applied entirely *before* discretization, so its gradient is sensitive only to distance from the clip points, blind to where a value sits relative to the interior levels. DoReFa and ordinary fixed-step quantization-aware training simply do not learn the step size, so level placement is never optimized against the loss. In every case the relative proximity of $v$ to the nearest quantization transition — precisely the thing that decides how a value reacts to a nudge in $s$ — does not enter the gradient.

I propose Learned Step Size Quantization (LSQ): make each weight-layer and activation-layer step size a trainable scalar parameter and give it a gradient by applying the straight-through estimator to the round node alone while differentiating the divide, clip, and multiply honestly. Take the interior region first, where $-Q_N < v/s < Q_P$ so the clip is inactive and $\hat v = \mathrm{round}(v/s)\cdot s$. By the product rule $\partial\hat v/\partial s = [\partial\,\mathrm{round}(v/s)/\partial s]\cdot s + \mathrm{round}(v/s)$. The STE treats the round as identity for differentiation, so $\mathrm{round}(v/s)\approx v/s$ and $\partial(v/s)/\partial s = -v/s^2$; multiplied by $s$ that first term is $-v/s$, and the second term is $\mathrm{round}(v/s)$. In the clipped regions the argument is pinned to an integer endpoint, so $\hat v = -Q_N\cdot s$ or $\hat v = Q_P\cdot s$ and the derivative is just $-Q_N$ or $Q_P$. Collecting,

$$\frac{\partial \hat v}{\partial s}=\begin{cases}-\,v/s+\mathrm{round}(v/s), & -Q_N < v/s < Q_P\\[2pt]-\,Q_N, & v/s\le -Q_N\\[2pt]\;\;Q_P, & v/s\ge Q_P.\end{cases}$$

The interior branch is where the payoff sits. Writing $z=v/s$, $n=\mathrm{round}(z)$, and the residual $r=z-n\in[-\tfrac12,\tfrac12]$, the interior branch is $-z+n=-r$, the negative signed residual between $z$ and the integer level it rounds to. The quantization levels are the integers and the bin transitions are the half-integers, where $|r|$ is largest; so $\partial\hat v/\partial s$ is large in magnitude exactly when $z$ sits near a transition and vanishes when $z$ sits right on a level. That is the correct sensitivity: a value near a transition is the one a small change in $s$ can flip into a different integer code, producing a large jump in $\hat v$, so its gradient to $s$ should be large. This is precisely what the cancel-the-round trick zeroed out and what the pre-discretization transforms could not see; here it falls straight out of doing the STE honestly on the round and the calculus honestly on everything else — nothing exotic, I simply did not cancel the term they cancelled. The data gradient follows from the same STE: $\partial\hat v/\partial v = 1$ when $-Q_N < v/s < Q_P$ and $0$ otherwise, identity inside the range and killed for clipped values, which still trains the weights and activations under fp32 shadow weights in the usual quantization-aware way.

One scalar $s$ per layer is now being optimized by the same SGD and the same global learning rate as millions of weights, and that should make me nervous, because training behaves well only when the ratio of average update magnitude to parameter magnitude is roughly balanced across layers — a parameter whose updates are huge relative to its own size overshoots, one whose updates are tiny stalls. So I estimate the imbalance ratio $R=(\nabla_s L/s)/(\lVert\nabla_w L\rVert/\lVert w\rVert)$. For the parameter-magnitude piece, an L2 norm over $N_W$ weights grows like $\sqrt{N_W}$ times a typical weight magnitude, and anchoring at $Q_P=1$ the step size should be about the average weight magnitude (splitting the distribution into zero / non-zero), while for larger $Q_P$ the step should shrink like $1/\sqrt{Q_P}$ as more levels give a finer step; so the typical weight magnitude is $\approx s\sqrt{Q_P}$ and $\lVert w\rVert/s\approx\sqrt{N_W Q_P}$. For the gradient piece, $\nabla_s L=\sum_{i=1}^{N_W}(\partial L/\partial\hat w_i)(\partial\hat w_i/\partial s)$; treating the per-element factor $\partial\hat w_i/\partial s$ as a constant in the same heuristic sense and the $\partial L/\partial\hat w_i$ as uncorrelated zero-mean, the sum of $N_W$ terms gives $\mathbb{E}[(\nabla_s L)^2]\approx N_W\,\mathbb{E}[(\partial L/\partial\hat w)^2]$, the very same right-hand side as $\mathbb{E}[\lVert\nabla_w L\rVert^2]$ since $\partial\hat w/\partial w\approx 1$ for unclipped weights. So the scalar $\nabla_s L$ and the norm $\lVert\nabla_w L\rVert$ are of the same order, and

$$R \approx \frac{\nabla_s L}{s}\cdot\frac{\lVert w\rVert}{\lVert\nabla_w L\rVert}\approx 1\cdot\frac{\lVert w\rVert}{s}\approx\sqrt{N_W Q_P}.$$

The imbalance is not 1; it grows like $\sqrt{N_W Q_P}$, worse for wider layers and higher precision, so $s$ would be over-driven and training destabilized. The fix is to cancel it directly by scaling the step-size gradient by $g=1/R$. For weight step sizes $g=1/\sqrt{N_W Q_P}$ with $N_W$ the number of weights; for activation step sizes the relevant count is the number of features $N_F$ in the layer (with a preceding batch-norm whose learned scale drives the pre-quantization changes, the same $\sqrt{N_F Q_P}$ imbalance appears), so $g=1/\sqrt{N_F Q_P}$. With this multiplier the expected update-to-parameter scale for $s$ is brought into line with the weights, and a single learning rate trains everyone.

The remaining question is how to inject a gradient *scale* $g$ — and the STE on the round — without disturbing the forward value, and the answer is one detach primitive (identity in forward, gradient blocked in backward). Define $\mathrm{gradscale}(x,g)=\mathrm{detach}(x-g\,x)+g\,x$: in forward the detach is identity so this is exactly $x$, leaving $s$ untouched, while in backward the gradient flows only through the $+g\,x$ branch and arrives scaled by $g$ — a transparent gradient multiplier. The round STE is the same construction, $\mathrm{roundpass}(x)=\mathrm{detach}(\mathrm{round}(x)-x)+x$, which rounds in forward and passes gradient $1$ in backward. To start SGD off near the data, each per-layer step size is initialized to $s=2\langle|v|\rangle/\sqrt{Q_P}$ from the initial weights or the first activation batch, where the $\sqrt{Q_P}$ denominator correctly makes more levels give a finer initial step; from there $s$ is learned. The rest is the standard quantized-network path: keep fp32 stored weights and use quantized weights and activations in the forward and backward passes, quantize all matrix-multiplication layers except the first and last (which stay 8-bit) with other parameters in fp32, initialize from a trained full-precision model, and fine-tune in the quantized space with momentum SGD, softmax cross-entropy, and cosine learning-rate decay.

```python
import torch
import torch.nn as nn

def detach(x):
    return x.detach()                     # identity forward; blocks gradient backward

def gradscale(x, g):
    return detach(x - g * x) + g * x      # forward: x; backward: gradient to x scaled by g

def roundpass(x):
    return detach(torch.round(x) - x) + x # forward: round; backward: straight-through (grad=1)

def nfeatures(v):
    return v[0].numel()

def nweights(v):
    return v.numel()

def qparams(bits, is_activation):
    if is_activation:
        return 0, 2 ** bits - 1, nfeatures
    return -2 ** (bits - 1), 2 ** (bits - 1) - 1, nweights

def quantize(v, s, bits, is_activation):
    qmin, qmax, count_fn = qparams(bits, is_activation)
    g = 1.0 / (count_fn(v) * qmax) ** 0.5 # 1/sqrt(N_features*Qp) or 1/sqrt(N_weights*Qp)
    s = gradscale(s, g)
    v_scaled = torch.clamp(v / s, qmin, qmax)
    v_bar = roundpass(v_scaled)
    return v_bar * s

class QuantLayer(nn.Module):
    def __init__(self, bits, is_activation):
        super().__init__()
        self.bits, self.is_activation = bits, is_activation
        self.s = nn.Parameter(torch.tensor(1.0))
        self.inited = False

    def init_step(self, v):
        _, qmax, _ = qparams(self.bits, self.is_activation)
        init = 2 * v.detach().abs().mean() / (qmax ** 0.5)
        with torch.no_grad():
            self.s.copy_(init)
        self.inited = True

    def forward(self, v):
        if not self.inited:
            self.init_step(v)
        return quantize(v, self.s, self.bits, self.is_activation)
```
