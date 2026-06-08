**Problem.** The baseline ladder (SiLU → Mish → GELU) is three fixed, parameter-free curves; GELU wins
(93.11 / 71.38 / 94.75) but each rung commits every neuron in every network to one hand-picked shape,
and the gains are decaying. The remaining lever is to stop fixing the curve and let each channel learn
its own — under the same frozen pipeline, editing only `CustomActivation`.

**Key idea.** SiLU/ReLU/PReLU are all members of the Maxout family `max(η_a(x), η_b(x))`; the smooth
maximum `S_β(x_1,…,x_n)=Σ x_i e^{βx_i}/Σ e^{βx_i}` bridges hard-max to a smooth curve. Its two-piece
form is `S_β(η_a,η_b)=(η_a−η_b)·σ(β(η_a−η_b))+η_b`, and `S_β(x,0)=x·σ(βx)` is exactly SiLU — so the
ladder's curves are frozen instances of one construction. Make the pieces learnable per channel:

`ACON-C:  f(x) = (p_1−p_2)·x·σ(β(p_1−p_2)x) + p_2 x`,  init `p_1=β=1, p_2=0`.

At init this *is* SiLU `x·σ(x)` (the rung-1 curve) and learns away from it per channel.

**Why it can beat the best fixed curve.** Fixed curves differ only in gate shape (worth
hundredths–tenths here). ACON-C adds two things they cannot express. (1) Its first-derivative bounds
(from `f″=0` ⇒ `(y−2)e^y=y+2`, `y=(p_1−p_2)βx≈±2.39936`) are `≈1.0998 p_1−0.0998 p_2` and
`≈1.0998 p_2−0.0998 p_1` — *learnable* per channel, where SiLU/GELU's are *fixed* (≈1.0998, −0.0998) and
`β` only sets asymptotic speed. (2) `β→∞` ⇒ `max(p_1x,p_2x)` (activate, nonlinear); `β→0` ⇒
`((p_1+p_2)/2)x` (do not activate, linear) — each channel learns *whether to be a nonlinearity*. That
per-channel control of gradient bounds and nonlinear degree is exactly the freedom the fixed ladder
lacked, and it has the most to exploit on the deep/high-capacity run where every prior gain concentrated.

**Edit mechanics.** `CustomActivation()` is built with no args at every call site, so size `p_1, p_2, β`
*lazily on the first forward* from the channel axis (`x.shape[1]` for 4-D conv maps, last dim for the
2-D classifier activation), as broadcasting `nn.Parameter`s. The forward is the ACON-C formula:
differentiable, shape-preserving. The new parameters join the same frozen SGD/cosine optimizer via
`model.parameters()`; nothing else (schedule, init, data, loop) changes.

**Hyperparameters.** Per-channel `p_1, p_2, β`, init `1, 0, 1` (= SiLU at start). No new training
hyperparameters; uses the fixed SGD/cosine pipeline.

**Bar / what I'd validate.** Must clear GELU's row, especially vgg16bn-cifar100 (71.38); expect the
near-ceiling resnet20-cifar10 (93.11) and mobilenetv2-fmnist (94.75) to hold at least even. To confirm
the mechanism rather than luck, check the learned per-channel `β` distribution spreads away from the
`β=1` SiLU point (some channels near-max/nonlinear, some near-mean/linear) — the per-neuron
activate-or-not behavior no fixed curve can express.

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
        self.p1 = None
        self.p2 = None
        self.beta = None

    def _init_params(self, x):
        if x.dim() >= 2:
            channels = x.shape[1]
            shape = [1, channels] + [1] * (x.dim() - 2)
        else:
            channels = x.shape[-1]
            shape = [channels]
        kw = dict(device=x.device, dtype=x.dtype)
        self.p1 = nn.Parameter(torch.ones(shape, **kw))
        self.p2 = nn.Parameter(torch.zeros(shape, **kw))
        self.beta = nn.Parameter(torch.ones(shape, **kw))

    def forward(self, x):
        if self.p1 is None:
            self._init_params(x)
        diff = (self.p1 - self.p2) * x
        return diff * torch.sigmoid(self.beta * diff) + self.p2 * x
```
