The 2-bit grid finally broke the plateau. Validation loss fell from the 2.72 the binary and ternary rungs sat on to 2.4392, WikiText-2 from ~78 to 54.1, LAMBADA from ~110 to 82.0, and — the tell I was watching for — downstream *moved*: ARC-Easy 47→53.8, HellaSwag 28→31.5. So the resolution diagnosis was right: the binding constraint was not the off-switch but the inability to express a graded weight magnitude, and the five-level grid $\{-1, -2/3, 0, +2/3, +1\}$ the int2 fill actually realizes fixed exactly that. But the win also locates the next limit. The thing that moved the model was giving each weight more places to land; the thing I have not touched on any rung is *where those places are*. On all three baselines the grid is anchored by one number per tensor — the absmean $s = \mathrm{mean}(|W|)$ — chosen to minimize the reconstruction error $\|W - s\cdot\mathrm{grid}\|^2$, i.e. to make the discrete weights *close to the float weights*. But I do not care whether the discrete weights are close to the float weights; I care whether they make the *loss* small. Those are different objectives, and at two or three bits, where only a handful of levels exist per layer, the gap between them is exactly the gap between int2 and a usable model. With so few levels, the single scalar $s$ decides everything about placement: too small and I clip away the tails, throwing the large-magnitude weights into the top level and losing them; too large and I waste my few levels on coarse spacing, so the dense core near zero gets one level where it wanted three. The absmean balances reconstruction across the whole tensor, but reconstruction weights every weight equally while the loss does not — some weights matter enormously and some not at all — and there is no formula for the loss-optimal $s$.

So I propose LSQ — Learned Step Size Quantization (Esser et al., ICLR 2020) — adapted to this substrate: make the per-layer step size a **learnable parameter**, trained jointly with the weights against the actual next-token cross-entropy, so the network tunes its own quantization grid to minimize the thing I measure. It is a single extra scalar per `BitLinear`, optimized by the same AdamW that trains the weights. The quantizer is

$$\hat w = \mathrm{round}\!\big(\mathrm{clip}(w/s,\ -Q_N,\ Q_P)\big)\cdot s,$$

and the obstacle is the same wall every rung hit, only sharper: the round is flat almost everywhere, so the path from $s$ through $w/s$ into the integer code has zero ordinary derivative, and ordinary backprop sees the final multiply by $s$ but is blind to how $s$ moves weights *toward or away from bin transitions* — which is the whole reason $s$ matters. The fix for the round is the STE I have used throughout, treating round as identity on the backward pass; but here I must be more careful than on the baselines, because they only needed a gradient to the *weight* and the easy thing — cancelling the round entirely — would zero out the interior step-gradient and learn nothing. So I apply STE to the round node *only*, and differentiate the divide, clip, and multiply honestly. In the interior, where $-Q_N < w/s < Q_P$ and the clip is inactive, $\hat w = \mathrm{round}(w/s)\cdot s$, and by the product rule $\partial\hat w/\partial s = [\partial\,\mathrm{round}(w/s)/\partial s]\cdot s + \mathrm{round}(w/s)$. STE makes the first term use $\mathrm{round}(w/s)\approx w/s$, so $\partial(w/s)/\partial s = -w/s^2$ times $s$ gives $-w/s$; with the second term, $\partial\hat w/\partial s = \mathrm{round}(w/s) - w/s$. Writing $z = w/s$, $n = \mathrm{round}(z)$, $r = z - n$ (the residual, between about $-\tfrac12$ and $\tfrac12$):

$$\frac{\partial\hat w}{\partial s} = n - z = -r,$$

the negative signed residual between $z$ and the level it rounds to. That is the payoff. The step-size gradient is *largest in magnitude exactly when $z$ sits near a bin transition* ($|r|$ near $\tfrac12$) and *zero when $z$ sits right on a level* ($r=0$) — precisely the sensitivity I want, because a weight near a transition is the one a small change in $s$ will flip to a different code, producing a large jump in $\hat w$. The fixed-absmean baselines had no version of this at all: their $s$ was a constant of the weights, never a parameter, so it could never respond to where the weights sit relative to the grid. The clipped regions are simpler — if $w/s \le -Q_N$ then $\hat w = -Q_N s$ and $\partial\hat w/\partial s = -Q_N$; if $w/s \ge Q_P$ then $\partial\hat w/\partial s = Q_P$ — and the data gradient is the usual STE: $\partial\hat w/\partial w = 1$ inside the range, $0$ outside, which is correct because pushing a clipped latent weight further changes nothing in the forward pass.

One more thing has to be right or the learned step destabilizes training, and it is the part careless versions miss. I now have one scalar $s$ per layer optimized by the same AdamW, at the same global learning rate, as the layer's hundreds of thousands of weights. Training behaves well when, across parameters, the ratio of update magnitude to parameter magnitude sits in the same band — if $s$ gets updates huge relative to its own size it overshoots, if tiny it stalls. Check the ratio $R = (\nabla_s L / s)\,/\,(\|\nabla_w L\|/\|w\|)$. For the parameter sizes, $\|w\| \propto \sqrt{N_W}$ times a typical weight magnitude, and that magnitude is about $s\sqrt{Q_P}$ (with $Q_P$ levels the step shrinks like $1/\sqrt{Q_P}$), so $\|w\|/s \approx \sqrt{N_W Q_P}$. For the gradients, $\nabla_s L$ sums over all $N_W$ weights of $(\partial L/\partial\hat w)(\partial\hat w/\partial s)$, and treating the per-weight loss gradients as uncorrelated zero-mean with the $\partial\hat w/\partial s$ factor an order-one constant, $E[(\nabla_s L)^2] \approx N_W\,E[(\partial L/\partial\hat w)^2]$ — the *same order* as $E[\|\nabla_w L\|^2]$. So numerator and denominator of $R$ differ exactly by $\|w\|/s$, giving $R \approx \sqrt{N_W Q_P}$. That is not 1 — it grows with layer width and precision, so the step is over-driven by roughly that factor, worst in the widest layers. I cancel it directly by scaling the gradient flowing to $s$ by $g = 1/\sqrt{N_W Q_P}$, injected as a transparent gradient multiplier with the same detach trick as the STE: $\mathrm{gradscale}(s, g) = (s - g\,s)\texttt{.detach()} + g\,s$ is $s$ in the forward pass and multiplies the gradient by $g$ in the backward, so $s$ trains in the same update/parameter band as the weights and one AdamW learning rate serves both.

The canonical LSQ recipe assumes things this harness does not provide, and I must not import them blindly. Canonical LSQ *fine-tunes from a trained full-precision model* with momentum SGD and cosine decay; here I am pretraining from scratch with AdamW on the fixed 13,535-iteration schedule, so there is no FP teacher to initialize the step from — the latent weights start at the scaffold's $\texttt{std}=0.02$ init and the step must be initialized from them. The scale-aware initializer $s = 2\langle|w|\rangle/\sqrt{Q_P}$ does exactly that from the initial weights, so I set the step to $2\cdot\mathrm{mean}(|W|)/\sqrt{Q_P}$ at construction and let AdamW take it from there; the $\sqrt{Q_P}$ denominator correctly says "more levels $\Rightarrow$ finer initial step." Canonical LSQ keeps the first and last layers at 8 bits, but the harness ties `wte` to `lm_head` and quantizes every projection uniformly, so I keep the uniform treatment rather than carve out exceptions the scaffold cannot express. And I keep the activation path *identical* to the three baselines — 8-bit per-tensor absmax, $Q_b = 127$, STE — for two reasons: it isolates the contribution to the *weight* quantizer, the honest comparison against int2, and learning a second step for activations from scratch on top of an unstable-by-nature low-bit pretraining run is a risk the controlled experiment does not need. The bit-width I take is three: signed range $Q_N = 2^{b-1} = 4$, $Q_P = 2^{b-1}-1 = 3$, an eight-level grid $\{-4,\dots,+3\}$. Three bits keeps me firmly in the few-bit native-low-bit regime and deliberately gives the learned step *more* levels to place than int2's effective five, so the finale tests both halves of the thesis at once: more levels, and levels placed by the loss rather than by a reconstruction formula. No SubLN inside the layer, same as every rung — the block's pre-projection `LayerNorm` holds the variance.

So the finale adds a learnable `weight_scale` per `BitLinear` initialized to $2\cdot\mathrm{mean}(|W|)/\sqrt{Q_P}$, the two helper ops `roundpass` (STE round) and `gradscale` (the $1/\sqrt{N_W Q_P}$ step-gradient multiplier), and a `weight_quant` that clips $w/s$ to $[-Q_N, Q_P]$, STE-rounds, and rescales by the gradient-scaled $s$; the activation path and latent-weight machinery are unchanged. The bar is int2's 2.4392, and I expect val_loss strictly below it — into the low-2.4s or high-2.3s, near the ~2.37 the strongest non-baseline run on this task reached, evidence that headroom below int2 exists — with WikiText-2 below 54.1, LAMBADA below 82.0, and downstream at or above 53.8 / 31.5, because graded, loss-placed magnitudes are exactly what the completion tasks reward. The way I would be wrong is if the learned step does not train cleanly under AdamW from a scratch init: if the $\sqrt{N_W Q_P}$ gradient scaling is even slightly off, the step either oscillates and the grid thrashes or stalls and I have a fixed-step quantizer landing right back at int2 with one wasted parameter. I validate by watching the per-layer `weight_scale` trajectories — if they move smoothly off their absmean-flavored init and settle, the learned placement is doing real work. The falsifiable claim is simply this: a *learned* step at three bits beats the *fixed* absmean step at int2's effective resolution, val_loss strictly under 2.4392.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 38-115) -- finale: LSQ (learned step, 3-bit weights)
def gradscale(x, scale):
    """Transparent gradient multiplier: forward = x, backward = grad * scale.
    Used to apply LSQ's 1/sqrt(N_W * Q_P) scaling to the step-size gradient."""
    y_out = x
    y_grad = x * scale
    return (y_out - y_grad).detach() + y_grad


def roundpass(x):
    """Straight-through round: forward = round(x), backward = identity."""
    y_out = x.round()
    y_grad = x
    return (y_out - y_grad).detach() + y_grad


def weight_quant(weight, step, Qn, Qp):
    """LSQ weight quantization with a LEARNED per-tensor step size.

    Forward: w_q = round(clip(W / s, -Qn, Qp)), dequantized by s.
    Backward: STE through round; step-size gradient = round(W/s) - W/s
              (interior), scaled by g = 1/sqrt(N_W * Qp) for stable training.
    Returns (quantized_weight, scale) with quantized_weight * scale ~= weight.
    """
    g = 1.0 / ((weight.numel() * Qp) ** 0.5)
    s = gradscale(step, g)                       # transparent forward, gradient * g
    w = torch.clamp(weight / s, -Qn, Qp)         # clipped weights get zero data-gradient
    w = roundpass(w)                             # STE round; interior s-grad = round(W/s) - W/s
    return w, s                                  # w * s ~= weight


def activation_quant(x):
    """Absmax 8-bit activation quantization with STE (identical to the baselines)."""
    Qb = 127  # int8 range
    scale = x.detach().abs().max().clamp(min=1e-12)
    x_normed = x / scale
    x_q = (x_normed * Qb).round().clamp(-Qb, Qb)
    x_q = (x_q - x_normed * Qb).detach() + x_normed * Qb
    return x_q, scale / Qb


class BitLinear(nn.Module):
    """LSQ linear layer: 3-bit weights with a learnable step size, 8-bit activations.

    self.weight is the float latent weight the optimizer updates; self.weight_scale is the
    learnable per-tensor LSQ step, initialized to 2*mean(|W|)/sqrt(Qp). Same path train/eval.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        bits = 3
        self.Qn = 2 ** (bits - 1)               # 4 for 3-bit signed
        self.Qp = 2 ** (bits - 1) - 1           # 3 for 3-bit signed
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        # LSQ step-size init from the scratch weights: s = 2<|W|>/sqrt(Qp)
        with torch.no_grad():
            init_s = 2.0 * self.weight.abs().mean() / (self.Qp ** 0.5)
        self.weight_scale = nn.Parameter(init_s.clone().clamp(min=1e-5))

    def forward(self, x):
        w_q, w_scale = weight_quant(self.weight, self.weight_scale, self.Qn, self.Qp)
        x_q, x_scale = activation_quant(x)
        out = F.linear(x_q, w_q, None)
        out = out * (w_scale * x_scale)
        if self.bias is not None:
            out = out + self.bias
        return out
```
