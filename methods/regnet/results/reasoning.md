Let me think about what I actually want out of architecture design, because I'm dissatisfied with both tools I have. Manual design — VGG, ResNet — gave me principles I can reuse: stack 3×3s, double width when you halve resolution, go deep, add residuals. That's gold, because a principle transfers. But once I'm juggling per-stage widths, depths, group widths, and bottleneck ratios all at once, hand-tuning gets hopeless. NAS, on the other hand, will happily search a fixed space and hand me one excellent network — but it's *one* network, tuned to one FLOP budget and one accelerator, and it tells me nothing about *why*, and nothing about what to do when I change the budget. I want the best of both: an automated procedure that nonetheless outputs a transferable principle and a whole family of good models, not a single point.

So flip the object of study. Instead of searching for the best model *in* a space, let me try to improve the *space itself*. If I can take an enormous, unconstrained space of architectures and progressively carve it down into a small, simple space where good models are everywhere, then the carving steps *are* the design principles — and sampling a handful of models from the final space gives me a good net for any budget.

For that I need a way to say one space is better than another without searching either of them. Comparing "best model found in A" vs "best model found in B" is fragile — it depends on how hard I searched. Better to sample a population from each space, train them, and compare the *distributions* of error. Concretely: sample n models, get errors e_1…e_n, and look at the error empirical distribution function F(e) = (1/n) Σ_i 1[e_i < e] — the fraction of sampled models below error e. A space whose EDF sits to the left (lower error across the whole population, not just at the tip) is the better space. This is robust: it's a statement about typical quality, and the mean error (related to the area under the curve) summarizes it. And it's affordable — if I work in a low-compute regime, say 400 MFLOPs and 10 epochs, training 100 such models costs about what one ResNet-50 at 4 GFLOPs for 100 epochs costs. Ten epochs is plenty to rank *populations* even if it's noisy per model. When I want to know which value of some knob the good models prefer, I'll bootstrap: resample the (knob, error) pairs, repeatedly take the lowest-error one, and read off a confidence interval for the best knob value.

Fix the parts I'm not studying so I can isolate structure. Stem: a single stride-2 3×3 conv, 32 channels. Head: global average pool then a linear classifier. Body: four stages at decreasing resolution, each a stack of identical blocks, the first block in a stage carrying the stride-2 downsample. The block itself I'll keep standard — the ResNeXt residual bottleneck: 1×1 to a bottleneck width, a 3×3 group convolution, 1×1 back out, BatchNorm and ReLU after each, residual add. I'll call it the X block. Now the only thing that varies is *structure*: for each of the 4 stages, the number of blocks d_i, the block width w_i, the bottleneck ratio b_i, and the group width g_i.

Call this initial space AnyNetX. Four stages times four per-stage parameters is 16 degrees of freedom. Sampling: depths d_i ≤ 16, widths w_i ≤ 1024 and divisible by 8, bottleneck ratios b_i ∈ {1,2,4}, group widths g_i ∈ {1,…,32}. That's on the order of (16·128·3·6)^4 ≈ 10^18 configurations. I am not going to search 10^18 models. I'm going to find principles that shrink this.

First simplification to try: do the four stages really each need their own bottleneck ratio? Tie them — one shared b for all stages. Call this AnyNetX_B. Sample 500, train, plot the EDF against AnyNetX_A's. The two EDFs sit right on top of each other, in both the average and the best case. I should be careful about what that does and doesn't say: it does not say a single b is *better*, only that I haven't paid for the constraint. That's exactly the trade I want — three fewer degrees of freedom at no cost to population quality. And there's a bonus I didn't go looking for: with a single b, I can bootstrap on it, and a trend appears — with 95% confidence b ≤ 2 is best in this regime. No such trend was visible in the four separate b_i, because the signal was smeared across four dimensions. So tying isn't only free, it sharpens the analysis.

Same move for group width: tie g across stages → AnyNetX_C. EDF essentially unchanged again. Each tie removes 3 of the 4 per-stage copies of that parameter, so I've gone from 16 dof to 10. How much smaller is the space? My count for AnyNetX_A was (16·128·3·6)^4; let me actually multiply that: 16·128·3·6 = 36864 per stage, so 36864^4 ≈ 1.85×10^18. Tying b and g moves two of the four factors from "per stage" to "shared," which roughly divides out 3·6 = 18 of the per-stage choices twice — between three and four orders of magnitude — and the EDF is unmoved. And tying g lets me read a preference for g > 1 — worth remembering.

So far I've only removed redundancy. To actually *improve* the space I have to stop simplifying blindly and *look* at the good models. Take AnyNetX_C, pull out the networks with the lowest error and the highest error, and plot, for each, the width of every block from the first to the last. A pattern shows up in the good ones: their widths tend to *increase* through the network, while the bad ones are all over the place — some flat, some non-monotone. The obvious thing to try is to impose that as a constraint — w_{i+1} ≥ w_i — giving AnyNetX_D, and see what the EDF does. It improves, and not marginally. I want to be honest that this is correlational evidence promoted to a constraint: I saw increasing widths in good models, I forced increasing widths, and the population got better, so for *this* space the constraint earns its place. Whether it's "causal" in any deeper sense I can't say from an EDF, but I don't need to — the space-level test is exactly the right one for a space-level decision.

Look again, same way. Beyond widths increasing, the stage *depths* of the best models tend to increase too (not always in the very last stage, but generally). Test d_{i+1} ≥ d_i → AnyNetX_E. The EDF improves again. There's a side benefit I can quantify: each monotonicity constraint keeps only one of the orderings of four stage values, and there are 4! = 24 orderings, so it shrinks the space by a factor of 24. Two such constraints, plus the two ties, plus the later reparametrization, are what compound into the many-orders-of-magnitude shrink — I'll add it all up once I have the final space, rather than claiming a round number now.

Here's where I want to push harder. "Widths increase, depths increase" is qualitative. Can I capture the actual *shape* of a good network's width profile? Take the top 20 models from AnyNetX_E and overlay their per-block width curves on one plot, log-scale on the y-axis. The individual curves are staircases — quantized, piecewise constant, because each stage holds a fixed width. But in aggregate, the staircases hug a straight line. I try fitting a line by eye and w_j ≈ 48·(j+1) tracks the population's width growth — i.e. roughly intercept 48, slope 48, with block index j. The reading I take from this is that the width of the j-th block of a good network is, on average, close to an affine function of j. I don't want to over-trust an eyeballed line, so I treat 48·(j+1) as a hypothesis to test, not a fact — the real test comes below when I fit the line to *each* model and correlate the fit error with the model's error.

So the underlying degrees of freedom might be far fewer than the staircase suggests. Let me parametrize the *continuous* per-block width as a line,

  u_j = w_0 + w_a · j,  for 0 ≤ j < d,

with three scalars: depth d, an initial width w_0 > 0, and a slope w_a > 0. That generates a distinct width for every block — but real networks need quantized, piecewise-constant widths (a stage shares one width). I need a principled way to snap this line to a staircase, and crucially the staircase should multiply width by a roughly constant factor each stage (that's the "double width per stage" heuristic generalized). So rewrite the line in terms of a multiplicative step. Introduce w_m > 1 and define, for each block, an exponent s_j by

  u_j = w_0 · w_m^{s_j},   so   s_j = log(u_j / w_0) / log(w_m).

s_j is generally non-integer. To quantize, round it to the nearest integer ⌊s_j⌉ and set the quantized per-block width

  w_j = w_0 · w_m^{⌊s_j⌉}.

Now all blocks sharing the same ⌊s_j⌉ get the same width — they form a stage. Reading off per-stage quantities: stage i has width w_i = w_0 · w_m^i, and its depth d_i is just the count of blocks with ⌊s_j⌉ = i. The whole staircase — every per-stage width and depth — is determined by four scalars: d, w_0, w_a, w_m. Before I trust the construction I should run it once by hand and see whether the line, the log-step, the rounding, and the power actually compose into a sane staircase. Take a concrete setting — say d = 13, w_0 = 24, w_a ≈ 36.4, w_m ≈ 2.49 — and crank it through. The continuous line u_j = 24 + 36.4·j gives [24, 60.4, 96.9, 133.3, 169.8, 206.2, 242.6, 279.1, 315.5, 352.0, 388.4, 424.8, 461.3]. The exponents s_j = log(u_j/24)/log(2.49) rounded are [0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]. Powering back, w_0·2.49^s_j and snapping to a multiple of 8, the per-block widths are [24, 56, 152, 152, 152, 152, 368, 368, 368, 368, 368, 368, 368]. Counting equal runs, that's four stages with widths [24, 56, 152, 368] and depths [1, 1, 4, 7], summing to 13 blocks. So the procedure really does turn 4 scalars into a clean monotone-width, monotone-depth, geometrically-stepping staircase — and I can see by construction that it bakes in both monotonicity constraints: w_a > 0 forces u_j increasing so the snapped widths are non-decreasing, and the geometric log-step gives the doubling-style stage pattern. Good — it's not just a formula on paper, it produces buildable per-stage lists.

Does this actually describe good models, or did I just invent a tidy formula that *can* produce staircases? Different claim, and it needs its own test. For each AnyNetX model, set d to its depth and grid-search w_0, w_a, w_m to minimize a fitting error e_fit = mean |log(predicted/observed)| over the per-block widths. Sanity-check the metric first on the staircase I just generated: it was produced by exactly this parametrization, so a perfect fit should read e_fit = 0, and computing mean|log(w/w)| over those 13 widths does give 0; against a deliberately non-monotone staircase like [200,200,100,100,50,50,368,...] the same metric reads ≈ 0.54. So e_fit is small only when the model's widths really do follow a quantized line, and large when they wander — it's measuring the right thing. Now the actual question: do the *best* models have small e_fit? Plotting e_fit against network error across AnyNetX_C, D, E, the low-error models cluster at small e_fit, and a bootstrap gives a tight band of e_fit near zero containing the best models in each space. And e_fit drops on average from C to E — which makes sense, since C→D→E added the very monotonicity constraints that a linear width law implies, so a population that satisfies those constraints should be easier to fit by a line. That correlation is the evidence I was missing for the eyeballed 48·(j+1): being well-described by a quantized linear width function is itself a marker of a good network, not just a convenient way to draw one.

Then make a space out of exactly that. A network is specified by six numbers: d, w_0, w_a, w_m, plus the shared b and g I already tied. Given them, generate the per-stage widths and depths by the linear-then-quantize procedure above, build the X-block body, done. This is the space of simple, *regular* networks — I'll call it RegNet. Sampling ranges, set from where e_fit was small on AnyNetX_E: d < 64, w_0 and w_a < 256, 1.5 ≤ w_m ≤ 3, b and g as before. Let me put a number on the size, quantizing the continuous knobs sensibly: d up to 64, w_0 in multiples of 8 below 256 (≈32 values), w_a up to 256, w_m over [1.5,3] at ≈16 steps, b ∈ {1,2,4}, g ≈ 6 values — that multiplies to 64·32·256·16·3·6 ≈ 1.5×10^8, call it order 10^8. Against AnyNetX_A's 1.85×10^18 that's a ratio of ~10^10, i.e. about ten orders of magnitude smaller. So the compounding I deferred earlier really does land in the 10-orders range — and it came from honest factors (two ties, two 24× orderings, the reparametrization), not a round number I assumed. And its EDF beats AnyNetX across the board: better average error while keeping the best models. Two further pinches probe whether the structure is fragile: forcing w_m = 2 (exact doubling) slightly improves things, and even setting w_0 = w_a (so u_j = w_a·(j+1), the very line I first eyeballed) does a bit better still — both *help* rather than hurt, which tells me the regular structure isn't a knife-edge, but I'll leave both unrestricted to keep model diversity. The payoff: random search in RegNet is efficient enough that sampling ~32 models reliably turns up a good one.

I designed this in one narrow regime — 400 MFLOPs, 10 epochs, the X block. The whole point is generalizable principles, so the obvious failure mode is that I've overfit the design space to that one setting, and the only way to know is to move the setting and re-measure. Compare RegNetX's EDF to AnyNetX_A and AnyNetX_E at higher FLOPs, at more epochs, with five stages instead of four, and with different block types. In every case the ordering holds: RegNetX > AnyNetX_E > AnyNetX_A. That's the evidence against design-space overfitting — not an assumption, a re-run in conditions the space was never tuned for. The five-stage result is the one I'd most expect to break, since the unconstrained space gains a whole extra stage of freedom there; that the regular linear structure still leads is the most reassuring of the four.

Now the part that pays off the whole "interpretability" promise: analyze RegNetX and read off principles, switching to 100 models trained 25 epochs to sharpen the trends. Bootstrap each of the six parameters against error across FLOP regimes. The depth of the best models comes out *stable* — about 20 blocks (~60 layers) — regardless of compute budget. I didn't expect that; the reflex is to use deeper models for bigger budgets, and this says instead add width and groups, not depth, past ~20 blocks. The best bottleneck ratio comes out b = 1.0, which means *no* bottleneck at all — the reduce-then-expand that everyone uses isn't earning its place in these structure-optimized nets. The width multiplier of good models is w_m ≈ 2.5, near but not equal to the doubling heuristic — consistent with the earlier pinch where forcing w_m = 2 only barely helped. And w_a, w_0, g all grow with FLOPs. The depth-saturation and b = 1 findings are the surprising ones — exactly the kind of transferable statement a single NAS instance could never hand me, and the reason the "design the space" detour was worth it.

One more complexity axis I should respect, because I care about real runtime, not just FLOPs. Define a network's *activations* as the total size of all conv output tensors. On memory-bound accelerators like GPUs and TPUs, runtime can track activations more closely than FLOPs. Fitting the best models across regimes: activations grow like the square root of FLOPs, parameters grow linearly with FLOPs, and runtime is best modeled with both a linear and a square-root term — it depends on both. So when I tighten RegNetX for the final comparison, I don't just cap FLOPs; I also cap parameters and activations, which yields fast, low-memory models at no accuracy cost. Combined with the trends, the constrained RegNetX uses b = 1, d ≤ 40 (later 12 ≤ d ≤ 28), w_m ≥ 2, and the activation/parameter limits.

I should also check the trendy block ingredients before declaring the X block sufficient. The inverted bottleneck (b < 1, à la MobileNetV2) actually *degrades* the EDF here, and pushing to depthwise convolution (g = 1) is worse still — so for these regular networks, plain grouped bottlenecks with b = 1 win. Varying input resolution (EfficientNet's compound scaling) also hurts; a fixed 224×224 is best even at higher FLOPs. The one addition that does help is Squeeze-and-Excitation: a cheap channel-attention op that global-average-pools the feature map, runs it through a small reduce–ReLU–expand–sigmoid MLP, and rescales each channel. Drop SE into the X block to get the Y block, giving the RegNetY space; its EDF improves cleanly over RegNetX. So I'll support an optional SE with reduction ratio 0.25.

Time to write the generator, since the whole method is "six scalars → per-stage width and depth lists, fed to a standard bottleneck-net builder." The continuous widths are arange(d)·w_a + w_0. The exponents are round(log(ws/w_0)/log(w_m)). The quantized per-block widths are w_0·w_m^exponent, then I round each to a multiple of q = 8 so channel counts are hardware-friendly. Counting equal consecutive widths gives the per-stage widths and depths (unique with counts) — this is exactly the hand-trace I ran above, now in code. One subtlety from making it buildable: the bottleneck width w·b must be divisible by the group width g (a grouped conv needs equal channels per group), so I adjust widths slightly for compatibility — take the inner width v = w·b, clamp g to v, round v to a multiple of (g, or lcm(g,b) when b>1), and recover w = v/b. I should check this adjustment doesn't quietly mangle the staircase I worked so hard to get. Run it on the four 200MF stages with b = 1, g = 8: widths [24, 56, 152, 368] go in, and since b = 1 the inner width is just w and every one of 24, 56, 152, 368 is already a multiple of 8, so they come back out unchanged — the adjustment is a no-op exactly when it should be. A case where it bites: b = 2, g = 24, w = 120 gives inner v = 240, and lcm(24,2) = 24, 240 is already a multiple of 24, so again no change; it only moves widths when v isn't a multiple of the group/lcm. Good — the compatibility fix is a small local correction, not a silent re-design. Then I hand widths/depths/strides/bottleneck/group lists to the generic builder.

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

Tracing the chain back: I wanted transferable principles plus a family of good models, which neither manual design nor NAS gives, so I made the design *space* the object — measured by error EDFs over sampled populations. Tying the bottleneck ratio and group width across stages cost nothing on the EDF and exposed trends; *looking* at good models motivated the monotonic-width and monotonic-depth constraints, each of which improved the EDF when imposed; overlaying their width staircases suggested an affine width-vs-block-index law, which the per-model e_fit correlation then upgraded from an eyeballed line to a real marker of quality, and which I turned into a four-scalar quantized-linear parametrization (line → log-step exponents → round → power → snap to 8) — verified by hand to emit a clean [24,56,152,368]/[1,1,4,7] staircase. The space of exactly those regular networks — RegNet, six knobs d, w_0, w_a, w_m, b, g — works out to ~10^8 configs against ~10^18 for the unconstrained space (the ~10-orders shrink I actually multiplied out), has a better-populated EDF, holds its ordering across regimes/stages/blocks when re-measured, and on analysis hands back surprising principles: depth saturates near 20 blocks, the bottleneck should be removed (b=1), w_m≈2.5, and activations (∝√FLOPs) deserve to be constrained alongside FLOPs. Adding Squeeze-and-Excitation gives the Y variant.
