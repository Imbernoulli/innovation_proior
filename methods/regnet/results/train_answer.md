I want something out of convolutional architecture design that neither of my two tools delivers. Manual design — the line from LeNet through AlexNet, VGG, ResNet — gave me principles I can reuse: stack small 3×3 convolutions, double the width when you halve the resolution, go deep, add residual connections. Principles are precious because they transfer to budgets and settings I never tested. But the moment I am juggling per-stage widths, depths, group widths, and bottleneck ratios all at once, hand-tuning collapses under the combinatorics. Neural architecture search solves the search efficiently, but its output is a single network instance tuned to one FLOP budget and one accelerator. That instance teaches me nothing about *why* it is good, and gives me no rule for what to change when the compute regime changes. What I want is the best of both: an automated procedure whose output is a transferable principle and a whole family of good models, not a single point.

The move that makes this possible is to flip the object of study. Instead of searching for the best model *inside* a fixed space, I improve the *space itself*. If I can take an enormous, unconstrained space of architectures and progressively carve it into a small, simple space where good models are everywhere, then the carving steps *are* the design principles, and sampling a handful of models from the final space hands me a good network for any budget. To carve, I need a way to declare one space better than another without exhaustively searching either, and comparing "best model found in A" against "best model found in B" is fragile because it depends on how hard I looked. So I compare *distributions*: sample $n$ models from a space, train them, collect errors $e_1, \dots, e_n$, and look at the error empirical distribution function $F(e) = \frac{1}{n}\sum_i \mathbf{1}[e_i < e]$, the fraction of sampled models with error below $e$. A space whose EDF sits to the left — lower error across the whole population, not just at the tip — is the better space. This is robust, it summarizes typical quality through the area under the curve, and it is cheap: in a low-compute regime of 400 MFLOPs and 10 epochs, training 100 such models costs about what one ResNet-50 at 4 GFLOPs for 100 epochs costs. When I want to know which value of a knob the good models prefer, I bootstrap the (knob, error) pairs, repeatedly take the lowest-error one, and read off a confidence interval.

The method I arrive at is RegNet, and it is built by exactly this design-the-space procedure. First I fix everything I am not studying so that structure is isolated: a stem that is a single stride-2 3×3 conv with 32 channels, a head that is global average pool then a linear classifier, and a body of four stages at decreasing resolution, each a stack of identical blocks where only the first block carries the stride-2 downsample. The block is the standard ResNeXt residual bottleneck — $1{\times}1$ to a bottleneck width, a $3{\times}3$ group convolution, $1{\times}1$ back out, BatchNorm and ReLU after each conv, and a residual add — which I call the X block. The only thing left varying is structure: for each of the four stages, the depth $d_i$, width $w_i$, bottleneck ratio $b_i$, and group width $g_i$. This initial space is AnyNetX: $4 \times 4 = 16$ degrees of freedom, on the order of $10^{18}$ configurations. I am not going to search $10^{18}$ models; I am going to find principles that shrink it.

The shrinking proceeds by measured EDF comparisons, each step either free or beneficial. Tying the bottleneck ratio to one shared $b$ across stages (AnyNetX_B) leaves the EDF unchanged — three degrees of freedom removed for free — and, as a bonus, with a single $b$ the bootstrap now exposes a trend that was previously smeared across four dimensions: $b \le 2$ is best. Tying the group width to one shared $g$ (AnyNetX_C) is again free, and reveals $g > 1$ is preferred. Now I stop simplifying blindly and *look* at the good models: plotting the per-block width of the lowest- and highest-error networks, the good ones have widths that *increase* through the network. Imposing $w_{i+1} \ge w_i$ (AnyNetX_D) improves the EDF substantially — increasing width is causal, not incidental. Looking again, the stage depths of the best models also tend to increase, so imposing $d_{i+1} \ge d_i$ (AnyNetX_E) improves it once more. Each monotonicity constraint also divides the space by $4!$ by killing all orderings but one.

The decisive observation comes from pushing past the qualitative. Overlaying the per-block width curves of the top models on a log-scale plot, the individual curves are staircases — quantized and piecewise-constant because each stage shares one width — but in aggregate they hug a straight line: the width of the $j$-th block of a good network is, on average, an affine function of $j$. So I parametrize the continuous per-block width as a line,
$$u_j = w_0 + w_a \cdot j, \qquad 0 \le j < d,$$
with three scalars: depth $d$, an initial width $w_0 > 0$, and a slope $w_a > 0$. Real networks need quantized, piecewise-constant widths, and I want the staircase to multiply width by a roughly constant factor each stage — the doubling heuristic generalized — so I re-express the line multiplicatively. Introducing $w_m > 1$,
$$u_j = w_0 \cdot w_m^{\,s_j}, \qquad s_j = \frac{\log(u_j / w_0)}{\log w_m},$$
where $s_j$ is generally non-integer. I quantize by rounding each $s_j$ to the nearest integer and setting
$$w_j = w_0 \cdot w_m^{\,\lfloor s_j \rceil}.$$
Now all blocks sharing the same $\lfloor s_j \rceil$ get the same width — they form a stage — so stage $i$ has width $w_i = w_0 \cdot w_m^{\,i}$ and depth $d_i = \#\{\,j : \lfloor s_j \rceil = i\,\}$. The entire staircase, every per-stage width and depth, is determined by four scalars $d, w_0, w_a, w_m$, and this quantization bakes in both monotonicity constraints automatically: $w_a > 0$ forces increasing widths, and the geometric staircase forces a sensible increasing depth pattern. The choice of a geometric (log-step) quantization rather than a naive linear rounding is what makes the per-stage widths multiply by a near-constant factor, recovering the time-tested doubling rule as a special case rather than fighting it.

I check this is descriptive of good models rather than a tidy invention. For each AnyNetX model I fix $d$ to its depth and grid-search $w_0, w_a, w_m$ to minimize a fitting error $e_{\text{fit}}$, the mean log-ratio between predicted and observed per-block widths. The quantized line lands almost exactly on the best networks' staircases; plotting $e_{\text{fit}}$ against actual error across AnyNetX_C, D, E, the low-error models all cluster at small $e_{\text{fit}}$, and $e_{\text{fit}}$ drops on average from C to E — the linear parametrization enforces the same thing the monotonicity constraints did, only more tightly. So I make a space out of exactly this. A network is six numbers: $d, w_0, w_a, w_m$, plus the shared $b$ and $g$. Given them, generate the per-stage widths and depths by the linear-then-quantize procedure, build the X-block body. This is RegNet, the space of simple, regular networks: about $3 \times 10^8$ configurations, ten orders of magnitude smaller than AnyNetX_A, with an EDF that beats AnyNetX across the board, so efficient that sampling about 32 models reliably turns up a good one. Critically it does not overfit the regime it was designed in — its EDF ordering over AnyNetX holds at higher FLOPs, longer schedules, five stages instead of four, and different block types.

Because the whole promise is interpretability, I read principles off RegNetX, sharpening trends with 100 models trained 25 epochs and bootstrapping each of the six parameters against error. The best depth is *stable* at roughly 20 blocks (~60 layers) regardless of budget — contradicting the reflex of using deeper models for bigger budgets; past ~20 blocks you should add width and groups, not depth. The best bottleneck ratio is $b = 1.0$, meaning *no* bottleneck: the reduce-then-expand everyone uses is not helping structure-optimized nets, and indeed the inverted bottleneck ($b<1$, MobileNetV2-style) degrades the EDF and pushing to depthwise ($g=1$) is worse still. The width multiplier of good models is $w_m \approx 2.5$, close to but not exactly doubling. And $w_a, w_0, g$ all grow with FLOPs. I also respect runtime, not just FLOPs: a network's *activations* — the total size of all conv output tensors — track wall-clock time on memory-bound accelerators better than FLOPs do, and fitting the best models shows activations grow like $\sqrt{\text{FLOPs}}$ while parameters grow linearly, so the final constrained RegNetX caps activations and parameters alongside FLOPs at no accuracy cost. Varying input resolution (EfficientNet-style compound scaling) hurts; fixed $224\times224$ is best. The one ingredient that helps is Squeeze-and-Excitation — global-average-pool the feature map, run it through a small reduce–ReLU–expand–sigmoid MLP, and rescale each channel — dropped into the X block to give the Y block and the RegNetY space, whose EDF improves cleanly, with SE reduction ratio 0.25.

The implementation is just "six scalars → per-stage width and depth lists, fed to a standard bottleneck-net builder." The continuous widths are `arange(d)*w_a + w_0`, the exponents are `round(log(ws/w_0)/log(w_m))`, the quantized widths are `w_0*w_m^exponent` snapped to a multiple of $q=8$ so channel counts are hardware-friendly, and counting equal consecutive widths recovers the per-stage widths and depths. One buildability subtlety: the bottleneck width $w \cdot b$ must be divisible by the group width $g$, so I adjust each width slightly — take the inner width $v = w \cdot b$, clamp $g$ to $v$, round $v$ to a multiple of $g$ (or $\text{lcm}(g,b)$ when $b>1$), and recover $w = v/b$ — before handing the lists to the generic builder.

```python
import numpy as np
import torch.nn as nn

def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Six scalars -> per-stage widths and depths via the quantized linear function."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0                          # u_j = w_0 + w_a * j
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))          # s_j = log(u_j/w_0)/log(w_m)
    ws_all = w_0 * np.power(w_m, ks)                            # w_j = w_0 * w_m^round(s_j)
    ws_all = np.round(np.divide(ws_all, q)).astype(int) * q     # snap to multiple of q
    ws, ds = np.unique(ws_all, return_counts=True)              # -> per-stage w_i and d_i
    return ws.tolist(), ds.tolist(), len(ws)

def adjust_block_compatibility(ws, bs, gs):
    """Ensure bottleneck width w*b is divisible by group width g."""
    vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]
    gs = [int(min(g, v)) for g, v in zip(gs, vs)]
    ms = [np.lcm(g, int(b)) if b > 1 else g for g, b in zip(gs, bs)]
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    ws = [int(v / b) for v, b in zip(vs, bs)]
    return ws, bs, gs

def conv2d(w_in, w_out, k, stride=1, groups=1):
    return nn.Conv2d(w_in, w_out, k, stride=stride, padding=(k - 1) // 2, groups=groups, bias=False)
def norm2d(w): return nn.BatchNorm2d(w)
def activation(): return nn.ReLU(inplace=True)
def gap2d(): return nn.AdaptiveAvgPool2d((1, 1))

class SE(nn.Module):
    def __init__(self, w_in, w_se):
        super().__init__()
        self.avg_pool = gap2d()
        self.f_ex = nn.Sequential(conv2d(w_in, w_se, 1), activation(),
                                  conv2d(w_se, w_in, 1), nn.Sigmoid())
    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))

class BottleneckTransform(nn.Module):
    """1x1 -> 3x3 group conv [+SE] -> 1x1."""
    def __init__(self, w_in, w_out, stride, bot_mul, group_w, se_r):
        super().__init__()
        w_b = int(round(w_out * bot_mul))
        groups = w_b // group_w
        self.a = conv2d(w_in, w_b, 1); self.a_bn = norm2d(w_b); self.a_af = activation()
        self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups); self.b_bn = norm2d(w_b); self.b_af = activation()
        self.se = SE(w_b, int(round(w_in * se_r))) if se_r else None
        self.c = conv2d(w_b, w_out, 1); self.c_bn = norm2d(w_out)
    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class ResBottleneckBlock(nn.Module):
    def __init__(self, w_in, w_out, stride, bot_mul, group_w, se_r):
        super().__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride); self.bn = norm2d(w_out)
        self.f = BottleneckTransform(w_in, w_out, stride, bot_mul, group_w, se_r)
        self.af = activation()
    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj is not None else x
        return self.af(x_p + self.f(x))

class AnyStage(nn.Module):
    def __init__(self, w_in, w_out, stride, d, bot_mul, group_w, se_r):
        super().__init__()
        for i in range(d):
            self.add_module(f"b{i+1}", ResBottleneckBlock(w_in, w_out, stride, bot_mul, group_w, se_r))
            stride, w_in = 1, w_out
    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

class RegNet(nn.Module):
    def __init__(self, w_a, w_0, w_m, depth, bot_mul=1.0, group_w=8, se_r=0.0,
                 stem_w=32, num_classes=1000):
        super().__init__()
        ws, ds, _ = generate_regnet(w_a, w_0, w_m, depth)
        bs, gs = [bot_mul] * len(ws), [group_w] * len(ws)
        ws, bs, gs = adjust_block_compatibility(ws, bs, gs)
        self.stem = nn.Sequential(conv2d(3, stem_w, 3, stride=2), norm2d(stem_w), activation())
        prev_w = stem_w
        for i, (d, w, b, g) in enumerate(zip(ds, ws, bs, gs)):
            self.add_module(f"s{i+1}", AnyStage(prev_w, w, 2, d, b, g, se_r))
            prev_w = w
        self.head = nn.Sequential(gap2d(), nn.Flatten(), nn.Linear(prev_w, num_classes))
    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

# RegNetX-200MF: w_a=36.44, w_0=24, w_m=2.49, depth=13, group_w=8, bot_mul=1.0
# RegNetY = same with se_r=0.25
```
