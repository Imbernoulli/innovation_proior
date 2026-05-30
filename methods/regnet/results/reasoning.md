Let me think about what I actually want out of architecture design, because I'm dissatisfied with both tools I have. Manual design — VGG, ResNet — gave me principles I can reuse: stack 3×3s, double width when you halve resolution, go deep, add residuals. That's gold, because a principle transfers. But once I'm juggling per-stage widths, depths, group widths, and bottleneck ratios all at once, hand-tuning gets hopeless. NAS, on the other hand, will happily search a fixed space and hand me one excellent network — but it's *one* network, tuned to one FLOP budget and one accelerator, and it tells me nothing about *why*, and nothing about what to do when I change the budget. I want the best of both: an automated procedure that nonetheless outputs a transferable principle and a whole family of good models, not a single point.

So flip the object of study. Instead of searching for the best model *in* a space, let me try to improve the *space itself*. If I can take an enormous, unconstrained space of architectures and progressively carve it down into a small, simple space where good models are everywhere, then the carving steps *are* the design principles — and sampling a handful of models from the final space gives me a good net for any budget.

For that I need a way to say one space is better than another without searching either of them. Comparing "best model found in A" vs "best model found in B" is fragile — it depends on how hard I searched. Better to sample a population from each space, train them, and compare the *distributions* of error. Concretely: sample n models, get errors e_1…e_n, and look at the error empirical distribution function F(e) = (1/n) Σ_i 1[e_i < e] — the fraction of sampled models below error e. A space whose EDF sits to the left (lower error across the whole population, not just at the tip) is the better space. This is robust: it's a statement about typical quality, and the mean error (related to the area under the curve) summarizes it. And it's affordable — if I work in a low-compute regime, say 400 MFLOPs and 10 epochs, training 100 such models costs about what one ResNet-50 at 4 GFLOPs for 100 epochs costs. Ten epochs is plenty to rank *populations* even if it's noisy per model. When I want to know which value of some knob the good models prefer, I'll bootstrap: resample the (knob, error) pairs, repeatedly take the lowest-error one, and read off a confidence interval for the best knob value.

Fix the parts I'm not studying so I can isolate structure. Stem: a single stride-2 3×3 conv, 32 channels. Head: global average pool then a linear classifier. Body: four stages at decreasing resolution, each a stack of identical blocks, the first block in a stage carrying the stride-2 downsample. The block itself I'll keep standard — the ResNeXt residual bottleneck: 1×1 to a bottleneck width, a 3×3 group convolution, 1×1 back out, BatchNorm and ReLU after each, residual add. I'll call it the X block. Now the only thing that varies is *structure*: for each of the 4 stages, the number of blocks d_i, the block width w_i, the bottleneck ratio b_i, and the group width g_i.

Call this initial space AnyNetX. Four stages times four per-stage parameters is 16 degrees of freedom. Sampling: depths d_i ≤ 16, widths w_i ≤ 1024 and divisible by 8, bottleneck ratios b_i ∈ {1,2,4}, group widths g_i ∈ {1,…,32}. That's on the order of (16·128·3·6)^4 ≈ 10^18 configurations. I am not going to search 10^18 models. I'm going to find principles that shrink this.

First simplification to try: do the four stages really each need their own bottleneck ratio? Tie them — one shared b for all stages. Call this AnyNetX_B. Sample 500, train, plot the EDF against AnyNetX_A's. The two EDFs sit right on top of each other, in both the average and the best case. So coupling the bottleneck ratios across stages costs nothing — three fewer degrees of freedom for free. And there's a bonus: with a single b, I can bootstrap on it, and a trend appears — with 95% confidence b ≤ 2 is best in this regime. No such trend was visible in the four separate b_i, because the signal was smeared across four dimensions.

Same move for group width: tie g across stages → AnyNetX_C. EDF essentially unchanged again. I'm now at 10 degrees of freedom, and the space is nearly four orders of magnitude smaller than where I started, with no loss in population quality. And tying g lets me see g > 1 is preferred — worth remembering.

Now stop simplifying blindly and *look* at the good models. Take AnyNetX_C, pull out the networks with the lowest error and the highest error, and plot, for each, the width of every block from the first to the last. A pattern jumps out: good networks have widths that *increase* through the network, while bad ones often don't. So impose that as a constraint — w_{i+1} ≥ w_i — giving AnyNetX_D. The EDF improves substantially. Increasing widths isn't just common, it's causal for population quality.

Look again. Beyond widths increasing, the stage *depths* of the best models tend to increase too (not always in the very last stage, but generally). Test d_{i+1} ≥ d_i → AnyNetX_E. The EDF improves again. Each of these monotonicity constraints also shrinks the space by 4! (it kills all the orderings but one), so I'm now at roughly 10 orders of magnitude smaller than AnyNetX_A while the population keeps getting better.

Here's where I want to push harder. "Widths increase, depths increase" is qualitative. Can I capture the actual *shape* of a good network's width profile? Take the top 20 models from AnyNetX_E and overlay their per-block width curves on one plot, log-scale on the y-axis. The individual curves are staircases — quantized, piecewise constant, because each stage holds a fixed width. But in aggregate, the staircases hug a straight line. I can literally draw w_j ≈ 48·(j+1), block index j, and it tracks the population's width growth. That's striking: the width of the j-th block of a good network is, on average, an affine function of j.

So the underlying degrees of freedom might be far fewer than the staircase suggests. Let me parametrize the *continuous* per-block width as a line,

  u_j = w_0 + w_a · j,  for 0 ≤ j < d,

with three scalars: depth d, an initial width w_0 > 0, and a slope w_a > 0. That generates a distinct width for every block — but real networks need quantized, piecewise-constant widths (a stage shares one width). I need a principled way to snap this line to a staircase, and crucially the staircase should multiply width by a roughly constant factor each stage (that's the "double width per stage" heuristic generalized). So rewrite the line in terms of a multiplicative step. Introduce w_m > 1 and define, for each block, an exponent s_j by

  u_j = w_0 · w_m^{s_j},   so   s_j = log(u_j / w_0) / log(w_m).

s_j is generally non-integer. To quantize, round it to the nearest integer ⌊s_j⌉ and set the quantized per-block width

  w_j = w_0 · w_m^{⌊s_j⌉}.

Now all blocks sharing the same ⌊s_j⌉ get the same width — they form a stage. Reading off per-stage quantities: stage i has width w_i = w_0 · w_m^i, and its depth d_i is just the count of blocks with ⌊s_j⌉ = i. The whole staircase — every per-stage width and depth — is determined by four scalars: d, w_0, w_a, w_m. That's the quantization I wanted, and it bakes in both monotonicity constraints automatically: w_a > 0 forces increasing widths, and the geometric staircase forces a sensible depth pattern.

Does this actually describe good models, or did I just invent a tidy formula? Test it: for each AnyNetX model, set d to its depth and grid-search w_0, w_a, w_m to minimize the fitting error e_fit, the mean log-ratio between predicted and observed per-block widths. Two things to check. First, do the *best* models fit well? Yes — the quantized line lands almost exactly on top of the top networks' staircases. Second, plot e_fit against actual network error across AnyNetX_C, D, E: the low-error models all cluster at small e_fit, and the bootstrap gives a tight band of e_fit near zero containing the best models in each space. And e_fit drops on average from C to E, confirming the linear parametrization is enforcing the same thing my monotonicity constraints were, only more tightly. So "is well-described by a quantized linear width function" is itself a marker of a good network.

Then make a space out of exactly that. A network is specified by six numbers: d, w_0, w_a, w_m, plus the shared b and g I already tied. Given them, generate the per-stage widths and depths by the linear-then-quantize procedure above, build the X-block body, done. This is the space of simple, *regular* networks — I'll call it RegNet. Sampling ranges, set from where e_fit was small on AnyNetX_E: d < 64, w_0 and w_a < 256, 1.5 ≤ w_m ≤ 3, b and g as before. Estimating its size by quantizing the continuous knobs gives ~3×10^8 configurations — about ten orders of magnitude below AnyNetX_A. And its EDF beats AnyNetX across the board: better average error while keeping the best models. Two further pinches confirm the structure isn't fragile: forcing w_m = 2 (exact doubling) slightly improves things, and even setting w_0 = w_a (so u_j = w_a·(j+1), the very line I first eyeballed) does a bit better still — but I'll leave both unrestricted to keep model diversity. The payoff: random search in RegNet is so much more efficient that sampling ~32 models reliably turns up a good one.

I designed this in one narrow regime — 400 MFLOPs, 10 epochs, the X block. The whole point is generalizable principles, so I have to check it transfers. Compare RegNetX's EDF to AnyNetX_A and AnyNetX_E at higher FLOPs, at more epochs, with five stages instead of four, and with different block types. In every case the ordering holds: RegNetX > AnyNetX_E > AnyNetX_A. No design-space overfitting. The five-stage result is especially reassuring — the regular linear structure extends to more stages where the unconstrained space has even more freedom to go wrong.

Now the part that pays off the whole "interpretability" promise: analyze RegNetX and read off principles, switching to 100 models trained 25 epochs to sharpen the trends. Bootstrap each of the six parameters against error across FLOP regimes. The depth of the best models is *stable* — about 20 blocks (~60 layers) — regardless of compute budget. That contradicts the reflex of using deeper models for bigger budgets; you should add width and groups, not depth, past ~20 blocks. The best bottleneck ratio is b = 1.0, which means *no* bottleneck at all — the reduce-then-expand that everyone uses isn't helping these structure-optimized nets. The width multiplier of good models is w_m ≈ 2.5, close to but not exactly the doubling heuristic. And w_a, w_0, g all grow with FLOPs. These are exactly the kind of transferable, surprising statements a single NAS instance could never give me.

One more complexity axis I should respect, because I care about real runtime, not just FLOPs. Define a network's *activations* as the total size of all conv output tensors. On memory-bound accelerators like GPUs and TPUs, runtime can track activations more closely than FLOPs. Fitting the best models across regimes: activations grow like the square root of FLOPs, parameters grow linearly with FLOPs, and runtime is best modeled with both a linear and a square-root term — it depends on both. So when I tighten RegNetX for the final comparison, I don't just cap FLOPs; I also cap parameters and activations, which yields fast, low-memory models at no accuracy cost. Combined with the trends, the constrained RegNetX uses b = 1, d ≤ 40 (later 12 ≤ d ≤ 28), w_m ≥ 2, and the activation/parameter limits.

I should also check the trendy block ingredients before declaring the X block sufficient. The inverted bottleneck (b < 1, à la MobileNetV2) actually *degrades* the EDF here, and pushing to depthwise convolution (g = 1) is worse still — so for these regular networks, plain grouped bottlenecks with b = 1 win. Varying input resolution (EfficientNet's compound scaling) also hurts; a fixed 224×224 is best even at higher FLOPs. The one addition that does help is Squeeze-and-Excitation: a cheap channel-attention op that global-average-pools the feature map, runs it through a small reduce–ReLU–expand–sigmoid MLP, and rescales each channel. Drop SE into the X block to get the Y block, giving the RegNetY space; its EDF improves cleanly over RegNetX. So I'll support an optional SE with reduction ratio 0.25.

Time to write the generator, since the whole method is "six scalars → per-stage width and depth lists, fed to a standard bottleneck-net builder." The continuous widths are arange(d)·w_a + w_0. The exponents are round(log(ws/w_0)/log(w_m)). The quantized per-block widths are w_0·w_m^exponent, then I round each to a multiple of q = 8 so channel counts are hardware-friendly. Counting equal consecutive widths gives the per-stage widths and depths (unique with counts). One subtlety from making it buildable: the bottleneck width w·b must be divisible by the group width g, so I adjust widths slightly for compatibility — take the inner width v = w·b, clamp g to v, round v to a multiple of (g, or lcm(g,b) when b>1), and recover w = v/b. Then I hand widths/depths/strides/bottleneck/group lists to the generic builder.

```python
import numpy as np

def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Six scalars -> per-stage widths and depths via the quantized linear function."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0                       # u_j = w_0 + w_a * j  (the line)
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))       # s_j = log(u_j/w_0)/log(w_m), rounded
    ws_all = w_0 * np.power(w_m, ks)                         # w_j = w_0 * w_m^round(s_j)
    ws_all = np.round(np.divide(ws_all, q)).astype(int) * q  # snap to multiple of q=8
    ws, ds = np.unique(ws_all, return_counts=True)           # equal widths -> per-stage w_i and d_i
    num_stages = len(ws)
    return ws.tolist(), ds.tolist(), num_stages

def adjust_block_compatibility(ws, bs, gs):
    """Make bottleneck width w*b divisible by group width g."""
    vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]        # inner (bottleneck) widths
    gs = [int(min(g, v)) for g, v in zip(gs, vs)]
    ms = [np.lcm(g, int(b)) if b > 1 else g for g, b in zip(gs, bs)]
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    ws = [int(v / b) for v, b in zip(vs, bs)]
    return ws, bs, gs
```

And the X/Y block and the builder, standard ResNeXt bottleneck with the optional SE:

```python
import torch.nn as nn

def conv2d(w_in, w_out, k, stride=1, groups=1):
    return nn.Conv2d(w_in, w_out, k, stride=stride, padding=(k - 1) // 2, groups=groups, bias=False)
def norm2d(w): return nn.BatchNorm2d(w)
def activation(): return nn.ReLU(inplace=True)
def gap2d(): return nn.AdaptiveAvgPool2d((1, 1))

class SE(nn.Module):
    """Squeeze-and-Excitation: pool -> reduce -> act -> expand -> sigmoid gate (the Y in RegNetY)."""
    def __init__(self, w_in, w_se):
        super().__init__()
        self.avg_pool = gap2d()
        self.f_ex = nn.Sequential(
            conv2d(w_in, w_se, 1), activation(),
            conv2d(w_se, w_in, 1), nn.Sigmoid())
    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))

class BottleneckTransform(nn.Module):
    """1x1 -> 3x3 group conv [+ SE] -> 1x1, the X (or Y) block body."""
    def __init__(self, w_in, w_out, stride, bot_mul, group_w, se_r):
        super().__init__()
        w_b = int(round(w_out * bot_mul))                    # bottleneck width (b=1 => no bottleneck)
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
    """x + f(x); 1x1 stride projection on the shortcut when shape changes."""
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
            stride, w_in = 1, w_out                          # first block downsamples; rest are identical

class RegNet(nn.Module):
    def __init__(self, w_a, w_0, w_m, depth, bot_mul=1.0, group_w=8, se_r=0.0,
                 stem_w=32, num_classes=1000):
        super().__init__()
        ws, ds, _ = generate_regnet(w_a, w_0, w_m, depth)    # six scalars -> per-stage structure
        bs = [bot_mul] * len(ws); gs = [group_w] * len(ws)
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
```

The causal chain: I wanted transferable principles plus a family of good models, which neither manual design nor NAS gives, so I made the design *space* the object — measured by error EDFs over sampled populations. Tying the bottleneck ratio and group width across stages cost nothing and exposed trends; *looking* at good models forced the monotonic-width and monotonic-depth constraints; overlaying their width staircases revealed an affine width-vs-block-index law, which I turned into a four-scalar quantized-linear parametrization (line → log-step exponents → round → power → snap to 8). The space of exactly those regular networks — RegNet, six knobs d, w_0, w_a, w_m, b, g — is ten orders of magnitude smaller than the unconstrained space, has a better-populated EDF, generalizes across regimes/stages/blocks, and on analysis hands back surprising principles: depth saturates near 20 blocks, the bottleneck should be removed (b=1), w_m≈2.5, and activations (∝√FLOPs) deserve to be constrained alongside FLOPs. Adding Squeeze-and-Excitation gives the Y variant.
