Let me start from the pain. I want 2-, 3-, 4-bit weights and activations and I want to keep full-precision accuracy. At that few bits there are only a handful of representable levels per layer, so *where* those levels land matters enormously, and where they land is set entirely by one number per layer: the step size s. Pick s too small and I clip away the tails; too large and I waste my precious few levels on coarse spacing. So the real question is how to *set s per layer*, and the honest answer is: don't set it by hand or by some fixed statistic — *learn* it, jointly with the weights, against the actual task loss. A per-layer scalar that the network tunes to minimize what I actually care about.

The obstacle is immediate. The quantizer is

```
v̄ = round( clip(v/s, −Q_N, Q_P) ),   v̂ = v̄ · s.
```

with Q_N = 0, Q_P = 2^b − 1 for unsigned activations and Q_N = 2^{b−1}, Q_P = 2^{b−1} − 1 for signed weights. The round is flat almost everywhere, derivative zero, so ∂L/∂s through it is zero — the loss can't see s. Standard move for the round itself: straight-through estimator — pretend the round is identity on the backward pass, ∂round(z)/∂z = 1. Fine. But I have to be careful: STE applies to the round *node*, while the *rest* of the expression — the divide by s, the clip, the multiply by s — I should differentiate honestly. People who tried to learn a clip parameter before me cut corners here, and it cost them, so let me actually do the calculus.

Take the interior region first, where −Q_N < v/s < Q_P so the clip is inactive. Then

```
v̂ = round(v/s) · s.
```

Differentiate w.r.t. s, product rule:

```
∂v̂/∂s = [∂ round(v/s)/∂s] · s + round(v/s) · 1.
```

For the first term, STE says treat round as identity, so round(v/s) ≈ v/s for differentiation purposes, and ∂(v/s)/∂s = −v/s². Multiply by s: that term is −v/s. The second term is round(v/s). So

```
∂v̂/∂s = −v/s + round(v/s),   for −Q_N < v/s < Q_P.
```

Now the clipped regions. If v/s ≤ −Q_N, the clip pins the argument at −Q_N, so v̄ = round(−Q_N) = −Q_N (an integer), and v̂ = −Q_N · s, whose derivative w.r.t. s is just −Q_N. Symmetrically, if v/s ≥ Q_P then v̂ = Q_P · s and ∂v̂/∂s = Q_P. So:

```
              ⎧ −v/s + round(v/s),   if −Q_N < v/s < Q_P
∂v̂/∂s  =      ⎨ −Q_N,                if v/s ≤ −Q_N
              ⎩  Q_P,                if v/s ≥ Q_P.
```

Let me stare at the interior branch, −v/s + round(v/s), because this is where the payoff is. Write v/s = n + f where n = round(v/s) and f ∈ (−½, ½] is the signed distance to the nearest integer (transition midpoint). Then −v/s + round(v/s) = −(n+f) + n = −f. So in the interior, ∂v̂/∂s = −(fractional distance of v/s from its nearest level). The gradient to the step size is *large in magnitude exactly when v sits near a transition between bins, and goes to zero when v sits right on a level.* That's the behavior I want, and it's the behavior the prior methods lacked: a value near a transition is the one most likely to jump to a different integer code under a small change in s (a smaller nudge to s flips it), which produces a large jump in v̂ — so its gradient to s should be large. The earlier clip-learning approaches either zeroed this interior gradient out entirely (cancel-the-round trick → ∂v̂/∂s = 0 inside the range) or made the gradient depend only on distance to the *clip* points, blind to the interior transitions. Here the right sensitivity falls straight out of just doing the STE honestly on the round and the calculus honestly on everything else. Nothing exotic — I just didn't cancel the term they cancelled.

I also need the gradient to flow to the *data* v for training the weights/activations, same STE on the round:

```
∂v̂/∂v = 1 if −Q_N < v/s < Q_P, else 0,
```

i.e. identity inside the range, killed outside (clipped values get no gradient). And I'll keep fp32 shadow weights, quantize in forward/backward, accumulate updates on the fp32 copy — the usual quantization-aware training setup.

Initialization of s. It should land the levels roughly where the data lives. A simple, scale-aware choice: set s so that the typical magnitude of v maps to around the middle-ish of the available positive range. Use s = 2⟨|v|⟩ / √Q_P, computed from the initial weights (per weight layer) or the first batch (per activation layer). The √Q_P denominator says: more levels ⇒ finer step ⇒ smaller s, which is the right direction. Good enough as a start; the point is s is then *learned*.

Now I run this and there's a problem I should anticipate before it bites me. I have one scalar s per layer being optimized by the same SGD, same global learning rate, as millions of weights. There's a known principle that training behaves well when, *across layers*, the ratio of average update magnitude to average parameter magnitude is about the same — if some parameter gets updates that are huge relative to its own size it overshoots, if tiny it stalls. So I'd like each step size's (update magnitude)/(parameter magnitude) to sit in the same band as the weights'. Let me check whether it does, by estimating the ratio

```
R = (∇_s L / s) / (‖∇_w L‖ / ‖w‖).
```

If R ≈ 1, s and w are balanced and a single learning rate serves both. If R is far from 1, s will be mis-trained relative to the weights. Let me actually estimate R, because I bet it's not 1.

First ‖w‖ / s. For a layer of N_W weights, an L2 norm grows like √(number of elements), so ‖w‖ ∝ √N_W times a typical weight magnitude. And what's s relative to the typical weight magnitude? Anchor at Q_P = 1: with a single positive level, s should be about the average weight magnitude, to split the distribution into zero / non-zero roughly evenly. For larger Q_P, I argued s should shrink like √(1/Q_P) (more levels, finer step, clip points sQ_P move out to catch outliers). So typical weight magnitude ≈ s·√Q_P, giving

```
‖w‖ / s ≈ √N_W · √Q_P = √(N_W Q_P).
```

Now the gradient norms. By chain rule,

```
∇_s L = Σ_{i=1}^{N_W} (∂L/∂ŵ_i)(∂ŵ_i/∂s).
```

The factor ∂ŵ_i/∂s is, from the interior branch above, −f_i ∈ (−½, ½], i.e. order 1; take it ≈ 1 in magnitude on average. Treat the ∂L/∂ŵ_i as uncorrelated, zero-mean random variables. Then the sum of N_W such terms has

```
E[(∇_s L)²] ≈ N_W · E[(∂L/∂ŵ)²].
```

For the weight-gradient norm, with ∂ŵ/∂w ≈ 1 for most (unclipped) weights,

```
E[‖∇_w L‖²] ≈ N_W · E[(∂L/∂ŵ)²].
```

Same right-hand side. So ∇_s L (a single scalar) and ‖∇_w L‖ (a norm over N_W weights) are of the *same order* — both ≈ √(N_W · E[(∂L/∂ŵ)²]). The numerator of R is ∇_s L / s and the denominator is ‖∇_w L‖ / ‖w‖, so

```
R ≈ (∇_s L / s) · (‖w‖ / ‖∇_w L‖) ≈ 1 · (‖w‖/s) ≈ √(N_W Q_P).
```

There it is — R is not 1, it's √(N_W Q_P), and it grows with both layer width and precision. The step-size update is too large relative to its magnitude by exactly that factor: bigger layers and higher precision make it worse. So s will be over-driven and training destabilized, and the bigger the layer the more so.

The fix is to cancel the imbalance directly: scale the gradient to s by g = 1/R = 1/√(N_W Q_P) for weight step sizes. For activations, the relevant count is the number of features N_F in the layer (assuming a preceding batch-norm whose learned scale is the main driver of pre-quantization activation changes, the same √(N_F Q_P) imbalance appears), so g = 1/√(N_F Q_P). Multiply the step-size gradient by g and now (update/parameter) for s matches the weights, and one learning rate trains everyone.

How do I inject a gradient *scale* g cleanly, without touching the forward value of s? The detach trick. Define

```
gradscale(x, g) = detach(x − g·x) + g·x.
```

Forward: detach is identity, so this is (x − g·x) + g·x = x — s is unchanged. Backward: detach blocks gradient through its argument, so ∂/∂x flows only through the +g·x branch and arrives scaled by g. Exactly a transparent gradient multiplier. The round STE I get the same way: roundpass(x) = detach(round(x) − x) + x — forward rounds, backward passes gradient = 1 (the round(x) − x is detached). Two custom-gradient ops, both built from one detach primitive.

Bookkeeping for inference: I quantize all matmul layers except the first and last (keep those 8-bit — standard, big accuracy payoff at the input/output), everything else fp32 master copies; initialize from a trained fp model and fine-tune in the quantized space. Train with momentum SGD, cross-entropy, cosine LR decay.

Assemble it:

```python
import torch, torch.nn as nn

def detach(x):
    return x.detach()  # identity forward; blocks gradient backward

def gradscale(x, g):
    # forward: returns x unchanged; backward: gradient to x is multiplied by g
    return detach(x - g * x) + g * x

def roundpass(x):
    # forward: round; backward: straight-through (gradient = 1)
    return detach(torch.round(x) - x) + x

def quantize(v, s, Qn, Qp, n_elems):
    g = 1.0 / (n_elems * Qp) ** 0.5      # gradient scale 1/sqrt(N*Qp) to balance s vs weights
    s = gradscale(s, g)
    v = v / s
    v = torch.clamp(v, Qn, Qp)           # values outside range get zero data-gradient (STE)
    v_bar = roundpass(v)                 # round with STE; interior s-gradient becomes -frac(v/s)
    v_hat = v_bar * s
    return v_hat

class QuantLayer(nn.Module):
    def __init__(self, bits, is_activation):
        super().__init__()
        self.bits, self.is_act = bits, is_activation
        self.s = nn.Parameter(torch.tensor(1.0))  # learnable per-layer step size
        self.inited = False
    def init_step(self, v):
        # s = 2<|v|>/sqrt(Qp): place levels around the data scale
        Qp = (2 ** self.bits - 1) if self.is_act else (2 ** (self.bits - 1) - 1)
        self.s.data = 2 * v.abs().mean() / (Qp ** 0.5)
        self.inited = True
    def forward(self, v):
        if not self.inited:
            self.init_step(v)
        if self.is_act:
            Qn, Qp = 0, 2 ** self.bits - 1
            n = v[0].numel()                      # features per sample
        else:
            Qn, Qp = -2 ** (self.bits - 1), 2 ** (self.bits - 1) - 1
            n = v.numel()                          # weights in the layer
        return quantize(v, self.s, Qn, Qp, n)
```

So the causal chain: at 2–4 bits the only knob that matters is the per-layer step size, so make it a trained parameter; the round kills its gradient, so apply STE to the round but differentiate the divide/clip/multiply honestly — and the interior branch −v/s + round(v/s) = −frac(v/s) drops out, giving exactly the missing sensitivity-to-transitions that prior clip-learners lacked; then, checking the update/parameter balance, the step-size gradient is too large by √(N·Q_P), so divide it by that — g = 1/√(N_W Q_P) for weights, 1/√(N_F Q_P) for activations — and inject g and the round-STE through a single detach-based gradient trick; initialize s ≈ 2⟨|v|⟩/√Q_P and fine-tune from a pretrained full-precision model.
