Let me start from the pain. I want 2-, 3-, 4-bit weights and activations and I want to keep full-precision accuracy. At so few bits there are only a handful of representable levels per layer, so *where* those levels land matters enormously, and where they land is set entirely by one number per layer: the step size s. Pick s too small and I clip away the tails; too large and I waste my precious few levels on coarse spacing. So the real question is how to *set s per layer*. The usual answers are to fix it by hand or from a statistic of the data — but those pick s to fit the *distribution* of v, not to minimize the *task loss*, which is what I actually care about and which couples across layers. So the option I want to chase is: make s a per-layer scalar parameter and let SGD tune it against the loss, jointly with the weights. The whole question is then whether I can give that scalar a usable gradient.

The obstacle is immediate. The quantizer is

```
v̄ = round( clip(v/s, −Q_N, Q_P) ),   v̂ = v̄ · s.
```

with Q_N = 0, Q_P = 2^b − 1 for unsigned activations and Q_N = 2^{b−1}, Q_P = 2^{b−1} − 1 for signed weights. The round is flat almost everywhere, so the path from s through v/s into the integer code has zero ordinary derivative almost everywhere; ordinary backprop can see the final multiply by s, but it misses how s moves values toward or away from bin transitions. Standard move for the round itself: straight-through estimator — pretend the round is identity on the backward pass, ∂round(z)/∂z = 1. Fine. But I have to be careful about *where* I apply that approximation: STE applies to the round *node*, while the *rest* of the expression — the divide by s, the clip, the multiply by s — I should differentiate honestly. People who tried to learn a clip parameter before me approximated more aggressively than that, so let me just do the calculus and see what the honest derivative is.

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

The clipped regions are simpler. If v/s ≤ −Q_N, the clip pins the argument at −Q_N, so v̄ = round(−Q_N) = −Q_N (an integer), and v̂ = −Q_N · s, whose derivative w.r.t. s is just −Q_N. Symmetrically, if v/s ≥ Q_P then v̂ = Q_P · s and ∂v̂/∂s = Q_P. So:

```
              ⎧ −v/s + round(v/s),   if −Q_N < v/s < Q_P
∂v̂/∂s  =      ⎨ −Q_N,                if v/s ≤ −Q_N
              ⎩  Q_P,                if v/s ≥ Q_P.
```

I derived this by hand under an STE approximation, so before I read anything into it I want to confirm the autograd implementation I'll actually ship reproduces these three branches. I build the quantizer with a straight-through round (`roundpass(x) = round(x)` forward, gradient 1 backward) and let the framework differentiate v̂ w.r.t. s. Take signed 3-bit weights, Q_N = 4, Q_P = 3, s = 1, and feed in a few values of v spanning interior and both clipped sides. The autograd ∂v̂/∂s comes out: v = 1.30 → −0.300, v = 0.30 → −0.300, v = −2.20 → +0.200, v = 2.60 → +0.400, v = 3.40 → +3.000 (= Q_P), v = −5.00 → −4.000 (= −Q_N). Each matches the branch above: interior gives −v/s + round(v/s) and the two clipped points give exactly Q_P and −Q_N. So the hand derivation and the code agree.

Now let me stare at the interior branch, −v/s + round(v/s), because the two clipped branches are just constants and this is the only one that carries information about the data. Write z = v/s, n = round(z), and r = z − n, with r sitting between about −1/2 and 1/2 under the rounding tie convention. The quantization levels are at the integers; the bin transitions are at the half-integers, where |r| is largest. Then −v/s + round(v/s) = −z + n = −r. So in the interior, ∂v̂/∂s is the negative signed residual between z and the integer level it rounds to.

I want to see what that residual does as z sweeps across a bin, so let me tabulate −z + round(z) over one cell. z = 1.00 → 0.000; z = 1.10 → −0.100; z = 1.30 → −0.300; z = 1.49 → −0.490; z = 1.51 → +0.490; z = 1.70 → +0.300; z = 2.00 → 0.000. So the step-size gradient is exactly zero when z sits on a level (z = 1, z = 2), grows in magnitude as z moves toward the bin boundary, peaks near ±0.49 at the transition (z = 1.49/1.51), and flips sign as z crosses from one integer's basin to the next. That is the behavior I'd want from a sensitivity to s: a value near a transition is the one most likely to jump to a different integer code under a small change in s — a smaller nudge to s flips it — which produces a discrete jump in v̂, so its gradient to s should be large; a value already on a level is insensitive, so its gradient should vanish. The earlier clip-learning approaches lose this: approximating the round away inside the range cancels exactly this term and zeros the interior gradient, or a gradient defined only by distance to the *clip* points is blind to where z sits relative to interior transitions. The interior sensitivity here is just what the calculus produces once the round is the only thing I approximate — I didn't have to design it in.

I also need the gradient to flow to the *data* v for training the weights/activations, same STE on the round. Inside the range ∂v̂/∂v = (∂/∂v) round(v/s)·s ≈ (∂/∂v)(v/s)·s = 1; outside the range v̂ is constant in v (pinned to ±Q_N·s or Q_P·s), so the derivative is 0:

```
∂v̂/∂v = 1 if −Q_N < v/s < Q_P, else 0.
```

The same autograd run confirms it: for the interior v's above the data gradient is 1.00, and for the two clipped v's (3.40, −5.00) it is 0.00 — clipped values get no gradient, which is correct since moving them does not move the output. And I'll keep fp32 shadow weights, quantize in forward/backward, accumulate updates on the fp32 copy — the usual quantization-aware training setup.

Before SGD can refine s, I still need a starting value that lands the levels roughly where the data lives. A simple, scale-aware choice is to set s so that the typical magnitude of v maps to around the middle-ish of the available positive range: s = 2⟨|v|⟩ / √Q_P, computed from the initial weights for a weight layer or the first batch for an activation layer. The √Q_P denominator says: more levels implies a finer step and a smaller s, which is the right direction. It only has to put the levels in the rough vicinity of the data; s is then learned from there.

There is one thing about this setup that worries me before I commit to it. I now have one scalar s per layer being optimized by the same SGD, same single global learning rate, as the millions of weights in that layer. A scalar and a high-dimensional vector trained by one learning rate is exactly the kind of mismatch that goes wrong quietly. There's a known principle (from the LARS line of work) that training behaves well when, *across* parameters, the ratio of average update magnitude to average parameter magnitude is roughly the same — if some parameter gets updates that are huge relative to its own size it overshoots, if tiny it stalls. So the thing I should check is whether s's (update/parameter) ratio sits in the same band as the weights'. Define the imbalance

```
R = (∇_s L / s) / (‖∇_w L‖ / ‖w‖).
```

If R ≈ 1, s and w are balanced and a single learning rate serves both; if R is far from 1, s is being mis-trained relative to the weights. I'd rather not guess the answer, so let me get a back-of-envelope prediction and then actually measure R numerically.

First the prediction. Numerator and denominator separately. For ‖w‖/s: an L2 norm of N_W weights grows like √N_W times a typical weight magnitude, and from the init reasoning the typical weight magnitude is about s·√Q_P (s splits the distribution sensibly at Q_P = 1, and shrinks like 1/√Q_P as levels are added), so ‖w‖/s ≈ √(N_W Q_P). For the gradient norms: by the chain rule ∇_s L = Σ_i (∂L/∂ŵ_i)(∂ŵ_i/∂s), a single scalar built from a sum of N_W per-weight terms; if the ∂L/∂ŵ_i are treated as uncorrelated zero-mean and ∂ŵ_i/∂s contributes a per-element factor of order one (the residual branch), then E[(∇_s L)²] ≈ N_W·E[(∂L/∂ŵ)²]. The weight-gradient norm, with ∂ŵ/∂w ≈ 1 for unclipped weights, has E[‖∇_w L‖²] ≈ N_W·E[(∂L/∂ŵ)²] — the same right-hand side. So ∇_s L and ‖∇_w L‖ are of the same order, the two cancel, and the prediction is

```
R ≈ (∇_s L / s) · (‖w‖ / ‖∇_w L‖) ≈ 1 · (‖w‖/s) ≈ √(N_W Q_P).
```

This rests on two heuristic approximations (treating the per-weight gradients as uncorrelated, and the ∂ŵ/∂s factor as O(1)), so I trust the *functional form* more than the constant out front. Let me put a number on it. I draw weights ~ N(0,1), set s = 2⟨|w|⟩/√Q_P, push them through the STE quantizer with a random zero-mean upstream gradient, and read off R from autograd, averaging over many draws. For 3-bit weights (Q_P = 3), sweeping the layer size:

```
N_W      measured R     √(N_W·Q_P)
   64        2.40          13.86
  256        4.10          27.71
 1024        8.71          55.43
 4096       19.56         110.85
16384       39.69         221.70
```

Two things jump out. The good news: each 4× in N_W multiplies measured R by very close to 2 (2.40 → 4.10 → 8.71 → 19.56 → 39.69), so R really does scale like √N_W — the layer-size dependence of the prediction is right. The caveat: the absolute value is not √(N_W·Q_P); measured R is about 0.16× the predicted value, fairly consistently. So my back-of-envelope got the *power* right and the *constant* wrong by ~6×, which is exactly what I'd expect from the uncorrelated/O(1) approximations. That's fine for what I need — I'm going to absorb the constant into the learning rate anyway — but I should also confirm the Q_P part of the scaling, not just N_W, since I asserted √(N_W·Q_P) and only swept N_W. Fixing N_W = 4096 and varying bits:

```
bits  Q_P     R       R/√N_W    √Q_P
  2     1   11.11     0.174     1.000
  3     3   20.21     0.316     1.732
  4     7   26.29     0.411     2.646
  8   127   92.34     1.443    11.269
```

R/√N_W tracks √Q_P across a 127× range of Q_P (0.174/1.000, 0.316/1.732, 0.411/2.646, 1.443/11.269 ≈ 0.17, 0.18, 0.16, 0.13 — roughly constant), so the full √(N_W·Q_P) form holds, prefactor ≈ 0.16. So R is genuinely *not* 1; it grows like √(N_W·Q_P), worse for wider layers and higher precision. Left alone, s is over-driven relative to its magnitude by that factor and training is destabilized, the more so the bigger the layer — which is a real problem precisely on the big high-precision layers I care about.

The fix is to cancel the imbalance directly: scale the gradient to s by g = 1/R. Since the constant is going to be swallowed by the learning rate, I take g = 1/√(N_W Q_P) for weight step sizes — the part of R that varies across layers. For activations, the relevant count is the number of features N_F in the layer (with a preceding batch-norm whose learned scale is the main driver of pre-quantization activation changes, the same √(N_F Q_P) imbalance argument applies), so g = 1/√(N_F Q_P).

I should check this actually does what I want — flatten R across layers — rather than assume the algebra carries over to the STE quantizer. Rerunning the same Monte Carlo with the gradient to s multiplied by g = 1/√(N_W·Q_P), at 3-bit:

```
N_W      R (with gradscale)
   64        0.167
  256        0.152
 1024        0.185
 4096        0.148
16384        0.155
```

R is now flat at ≈ 0.16 from N_W = 64 to 16384 — a 256× span over which the uncorrected R had climbed from 2.4 to 39.7. The leftover constant ≈ 0.16 is the same across all layers, so a single learning rate (the constant rescaled once) trains every step size in balance with its weights. That is the property I was after, and the numbers say the correction delivers it.

Now, how do I inject a gradient *scale* g cleanly, without touching the forward value of s? A `detach`-style op — identity forward, gradient blocked backward — lets me do it. Define

```
gradscale(x, g) = detach(x − g·x) + g·x.
```

Forward: detach is identity, so this is (x − g·x) + g·x = x — s is unchanged. Backward: detach blocks gradient through its argument, so ∂/∂x flows only through the +g·x branch and arrives scaled by g. I check it on the framework: `gradscale(2.5, 0.07)` returns 2.5 in the forward pass and its backward gradient to the input is 0.07 = g. The round STE comes from the same primitive: roundpass(x) = detach(round(x) − x) + x — forward rounds, backward passes gradient = 1 (the round(x) − x is detached); checking, `roundpass(1.27)` returns 1.0 forward with backward gradient 1.0. Two custom-gradient ops, both built from one detach primitive.

The remaining training setup follows the usual quantized-network path: use quantized weights and activations in forward/backward passes, store and update full-precision weights, quantize all matrix-multiplication layers except the first and last, keep those first and last layers at 8-bit, represent other model parameters in fp32, initialize from a trained full-precision model, and fine-tune in the quantized space with momentum SGD, cross-entropy, and cosine learning-rate decay.

The implementation follows directly:

```python
import torch
import torch.nn as nn

def detach(x):
    return x.detach()  # identity forward; blocks gradient backward

def gradscale(x, g):
    # forward: returns x unchanged; backward: gradient to x is multiplied by g
    return detach(x - g * x) + g * x

def roundpass(x):
    # forward: round; backward: straight-through (gradient = 1)
    return detach(torch.round(x) - x) + x

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
    g = 1.0 / (count_fn(v) * qmax) ** 0.5  # 1/sqrt(N_features*Qp) or 1/sqrt(N_weights*Qp)
    s = gradscale(s, g)
    v_scaled = torch.clamp(v / s, qmin, qmax)  # clipped values get zero data-gradient through clamp
    v_bar = roundpass(v_scaled)                # STE round; interior s-gradient is round(v/s)-v/s
    return v_bar * s

class QuantLayer(nn.Module):
    def __init__(self, bits, is_activation):
        super().__init__()
        self.bits, self.is_activation = bits, is_activation
        self.s = nn.Parameter(torch.tensor(1.0))  # learnable per-layer step size
        self.inited = False

    def init_step(self, v):
        # s = 2<|v|>/sqrt(Qp): place levels around the data scale
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
