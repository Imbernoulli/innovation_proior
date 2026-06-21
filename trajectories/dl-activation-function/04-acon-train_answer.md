GELU closed the baseline ladder as the strongest of the three: resnet20-cifar10 93.11, vgg16bn-cifar100 71.38, mobilenetv2-fmnist 94.75, against Mish's 92.78/70.50/94.70 and SiLU's 92.72/70.38/94.69. The distribution-matched-gate argument held, and it won by the most exactly where the headroom was, the deep VGG-16-BN/CIFAR-100 run, jumping from 70.50 to 71.38 — a full +0.88 — while it nudged the saturated runs by only ~0.05–0.33. But the trend across the whole ladder is the tell: the VGG run went 70.38 → 70.50 → 71.38 and CIFAR-10 went 92.72 → 92.78 → 93.11, real but each rung a single fixed parameter-free curve, and I have now tried the three most defensible ones — the simplest smooth gate, the better-conditioned-gradient gate, the distribution-matched gate. Every one commits every neuron in every network to one hand-picked shape, and the best that buys is GELU's row. The lever that is left is not "pick a better fixed curve"; it is "stop fixing the curve" — let each channel choose its own shape and, the part the fixed family cannot express at all, choose *how nonlinear to be*, including being nearly linear where linearity helps.

To get there without guessing I need the structure behind the curves I already have. ReLU is $\max(x,0)$; Leaky ReLU and PReLU are $\max(x,px)$; absolute value is $\max(x,-x)$ — all *max of two linear functions*, the Maxout family $\max(\eta_a(x),\eta_b(x))$. And SiLU, the curve I opened with, is *smooth*. The bridge between "hard max of linear pieces" and "smooth activation" is the smooth maximum, the differentiable surrogate for $\max$,
$$S_\beta(x_1,\dots,x_n)=\frac{\sum_i x_i e^{\beta x_i}}{\sum_i e^{\beta x_i}},$$
a temperature-$\beta$ softmax-weighted average of the values: as $\beta\to\infty$ the weight piles onto the largest argument and $S_\beta\to\max$, while as $\beta\to 0$ the weights equalize and $S_\beta\to$ the arithmetic mean, so $\beta$ slides between hard max and plain average. (This is the softmax-weighted *value* itself, distinct from softplus, which is the LogSumExp smoothing of $\max(x,0)$; that distinction is what produces a self-gated curve rather than another positive smooth ReLU.) Apply it to a two-piece max. Writing $S_\beta(\eta_a,\eta_b)$, dividing each fraction through and using $\sigma(-z)=1-\sigma(z)$ collapses it to
$$S_\beta(\eta_a,\eta_b)=(\eta_a-\eta_b)\,\sigma\!\big(\beta(\eta_a-\eta_b)\big)+\eta_b.$$
Plug in ReLU's pieces, $\eta_a=x,\ \eta_b=0$, and $S_\beta(x,0)=x\,\sigma(\beta x)$ — that is SiLU at $\beta=1$. The curve I started the ladder with is *the smooth maximum of $\{x,0\}$*, and the $\beta$ inside it is a temperature sliding the neuron between linear ($\beta\to0$, the mean of $x$ and 0, i.e. $x/2$) and ReLU-like max ($\beta\to\infty$). The fixed curves are all frozen instances of this one construction, which is the door.

I propose **ACON-C**: take the general two-piece member with learnable linear pieces $\eta_a=p_1x,\ \eta_b=p_2x$,
$$f(x)=(p_1-p_2)\,x\,\sigma\!\big(\beta(p_1-p_2)x\big)+p_2x,$$
the smooth maximum of two learnable linear functions through the origin, with $p_1,p_2,\beta$ learnable *per channel*. Initialize $p_1=\beta=1,\ p_2=0$ and at the first step it *is* $x\sigma(x)$ — exactly the SiLU curve rung one measured at 92.72/70.38/94.69 — so the finale starts sitting on top of the weakest baseline and is free to crawl away from it, per channel, during the same frozen 200-epoch cosine run. The worst case is that the parameters never move and it reproduces SiLU, so it should do no worse than SiLU at minimum.

What makes it beat the *best* fixed curve, not just SiLU, shows in the gradient. With $u=(p_1-p_2)x$,
$$f'(x)=(p_1-p_2)\big[\sigma(\beta u)+\beta u\,\sigma(\beta u)(1-\sigma(\beta u))\big]+p_2.$$
As a consistency check the asymptotes are $f'\to p_1$ as $x\to+\infty$ and $f'\to p_2$ as $x\to-\infty$ (taking $p_1>p_2$) — exactly the two linear pieces, as a smoothed $\max(p_1x,p_2x)$ must have. But the interesting part is the *bounds* of $f'$, the largest and smallest gradient the unit can pass. Let $a=p_1-p_2$, $y=\beta a x$; the bracket is $h(y)=\sigma(y)+y\sigma(y)(1-\sigma(y))$, so $f'=a\,h(y)+p_2$ and $f''=\beta a^2 h'(y)$. With $s=\sigma(y)$, $h'(y)=s(1-s)[2+y(1-2s)]$; since $s(1-s)\neq0$ for finite $y$, the extrema satisfy $2+y(1-2s)=0$, and substituting $1-2s=(1-e^y)/(1+e^y)$ and clearing denominators gives the transcendental equation
$$(y-2)e^{y}=y+2,\qquad y=(p_1-p_2)\beta x,$$
whose roots are $y\approx\pm2.39936$. At them $h\approx1.0998$ and $h\approx-0.0998$, so substituting back into $f'=a\,h+p_2$,
$$\max(f')\approx1.0998\,p_1-0.0998\,p_2,\qquad \min(f')\approx1.0998\,p_2-0.0998\,p_1.$$
For SiLU/Swish ($p_1=1,p_2=0$) those are the *constants* $\approx1.0998$ and $\approx-0.0998$ — fixed, with $\beta$ only setting how fast the derivative approaches those walls, never the walls themselves. Every fixed curve on my ladder, GELU included, has gradient bounds it cannot move; the three rungs only changed the *shape* of the gate between fixed walls, which is why their gains decayed. ACON-C's bounds depend on $p_1$ and $p_2$, which are learnable, so each channel sets its own gradient ceiling and floor. The smooth-maximum derivation shows the bound is exactly what $p_1,p_2$ control while $\beta$ *independently* controls how fast the derivative asymptotes — the nonlinear *degree* — two knobs the fixed curves conflated by pinning both.

And there is the qualitative payoff the fixed family cannot touch: $\beta\to\infty$ makes $f\to\max(p_1x,p_2x)$ (fully nonlinear, the neuron *activates*) while $\beta\to0$ makes $f\to\tfrac{p_1+p_2}{2}x$ (linear, the neuron does *not* activate). A learnable $\beta$ lets each channel decide *whether to be a nonlinearity at all* — the activate-or-not that ACON names. On these three networks that is the right freedom: the saturated ResNet-20 and MobileNetV2 runs near their ceiling may benefit from some channels going nearly linear (less to over-fit, smoother optimization), while the deep VGG run, where all the visible headroom is, can keep sharp the channels that need sharp nonlinearity and relax the rest. A fixed curve makes one compromise for the whole network; ACON-C makes it per channel and learns it. (The construction extends further — generating $\beta$ from the input via an SE-style channel bottleneck $\beta=\sigma(W_1W_2\cdot\mathrm{GAP}(x))$ gives a per-sample switch, meta-ACON — but that adds routing machinery beyond what this rung needs, so I land the per-channel learnable-$\beta$ form here.)

The literal edit has to respect the harness contract exactly. `CustomActivation` is instantiated with no arguments everywhere — in ResNet-20's BasicBlocks and stem, every VGG-16-BN Conv–BN block and the classifier head, every MobileNetV2 inverted residual — so I cannot pass a channel count into `__init__`. The clean way to honor "per-channel learnable parameters" without changing the construction sites is to size the parameters *lazily on the first forward* from the channel dimension of the incoming tensor: `x.shape[1]` for the 4-D conv feature maps, the last dim for the 2-D classifier activation, registering $p_1,p_2,\beta$ as `nn.Parameter`s shaped to broadcast over that channel axis, on the same device/dtype as `x`. The forward then computes the ACON-C formula, fully differentiable and shape-preserving. Because the lazy parameters are registered before the first forward, they join the same frozen SGD optimizer via `model.parameters()`; the fixed weight decay and learning rate apply to them too, which is fine — they start at the SiLU point and the cosine schedule anneals around it. Nothing else (schedule, init, data, loop) changes; this is the one rung on the whole ladder where new parameters enter, and the lazy sizing is what lets them enter without touching a single call site. The bar to clear is GELU's row, especially vgg16bn-cifar100 (71.38), with the near-ceiling runs expected to hold at least even; to confirm the mechanism rather than luck I would watch the learned per-channel $\beta$ distribution spread away from the $\beta=1$ SiLU point — some channels near-max and nonlinear, some near-mean and linear — the per-neuron activate-or-not behavior no fixed curve could ever express.

```python
# EDITABLE region of pytorch-vision/custom_activation.py (lines 32-49) -- finale: ACON-C (learnable)
class CustomActivation(nn.Module):
    """ACON-C: smooth maximum of two learnable linear pieces, per channel.

        f(x) = (p1 - p2) * x * sigmoid(beta * (p1 - p2) * x) + p2 * x

    p1, p2, beta are per-channel learnable parameters, sized lazily on the
    first forward from the channel axis (so __init__ stays argument-free, as
    the scaffold instantiates CustomActivation() everywhere). Init p1=beta=1,
    p2=0, so the module starts as SiLU x*sigmoid(x) and specializes per channel.
    Asymptotic slopes -> p1 (x->+inf), p2 (x->-inf); first-derivative bounds
    (~1.0998 p1 - 0.0998 p2, ~1.0998 p2 - 0.0998 p1) are learnable, unlike the
    fixed SiLU/GELU bounds. beta switches each channel between nonlinear
    (beta->inf, max) and linear (beta->0, mean). Differentiable, shape-preserving.
    """

    def __init__(self):
        super().__init__()
        self.p1 = nn.parameter.UninitializedParameter()
        self.p2 = nn.parameter.UninitializedParameter()
        self.beta = nn.parameter.UninitializedParameter()

    def _param_shape(self, x):
        if x.dim() == 2:
            return (1, x.shape[-1])
        if x.dim() > 2:
            return (1, x.shape[1], *([1] * (x.dim() - 2)))
        return (x.shape[0],)

    def _init_params(self, x):
        shape = self._param_shape(x)
        self.p1.materialize(shape, device=x.device, dtype=x.dtype)
        self.p2.materialize(shape, device=x.device, dtype=x.dtype)
        self.beta.materialize(shape, device=x.device, dtype=x.dtype)
        nn.init.ones_(self.p1)
        nn.init.zeros_(self.p2)
        nn.init.ones_(self.beta)

    def forward(self, x):
        if isinstance(self.p1, nn.parameter.UninitializedParameter):
            self._init_params(x)
        diff = (self.p1 - self.p2) * x
        return diff * torch.sigmoid(self.beta * diff) + self.p2 * x
```
