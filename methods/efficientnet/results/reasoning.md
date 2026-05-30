OK, let me think this through from scratch. The situation I'm in is concrete: I have a convolutional network that works well at some modest budget, and someone has just handed me more compute. The expectation is that I "scale it up" for more accuracy, the way everyone does — a residual net goes from 18 layers to 200, a giant model goes to 480×480 input and hundreds of millions of parameters and sets a new record. So scaling up clearly works. What bugs me is that nobody can tell me *how* to do it well. The recipe is folklore. You add layers. Or you add channels. Or you feed bigger images. Pick one, push it, hope.

Let me first be honest about what knobs I even have. A conv net is a stack of stages; within a stage every layer is the same operator, and only the first layer of a stage changes the spatial size. So a network is really a list: stage i has an operator F_i repeated L_i times, acting on a tensor of shape ⟨H_i, W_i, C_i⟩, and as I go deeper the spatial size H×W shrinks while channels C grow. The things I can dial without touching the operators themselves are exactly three: how many times each stage repeats (depth, the L_i), how many channels each layer has (width, the C_i), and how big the input image is (resolution, the H_i, W_i). Three knobs. Every "scale it up" story is some setting of these three.

So what does each one actually buy me, and where does it stop paying? Let me reason through them one at a time, because that's the only way to see why single-knob scaling is unsatisfying.

Depth first, because it's the default. More layers means more composition, richer and more hierarchical features, and historically deeper has meant better and has transferred well. But two things. One, depth is hard to train — gradients vanish through a long stack — though I'll grant that skip connections and batch normalization mostly solved trainability. Two, and this is the part that actually limits me: even when training is fine, the *accuracy* return on extra depth flattens out fast. The blunt fact is that a roughly thousand-layer residual net is no more accurate than a roughly hundred-layer one. So past a point, adding depth is adding compute and parameters for nothing. Depth alone has a ceiling.

Width next. Cranking up the channel count is the usual move for small models — the "width multiplier" knob. Wider layers capture finer-grained features and, pleasantly, are *easier* to optimize than very deep stacks. But there's a complementary failure: a network that's very wide and very shallow can't form high-level abstractions — it has the capacity to see fine detail but not the depth to compose it into concepts. And empirically, if I just keep widening at fixed depth, accuracy saturates quickly. So width alone also has a ceiling, and a different-flavored one.

Resolution third. Bigger input images let the network resolve finer patterns; the field has been creeping up — 224, then 299, 331, 480, and detection runs at 600. Higher resolution does help. But, same story, the gain tapers: going from already-large to even-larger images buys less and less.

Let me write down the thing all three share, because it's the first real handle I have. Whatever single dimension I scale, accuracy goes up but the gain *diminishes* as the model gets bigger — and the plateau is roughly the same low-80s top-1 for each. Stare at that. Three different knobs, three saturating curves, all topping out around the same place. That's suspicious. If three independent levers all stall at the same accuracy, maybe the problem isn't any one lever being weak; maybe it's that pushing one lever while the other two sit still is the wrong thing to do.

So let me actually test the coupling in my head, because the intuition has to be made concrete before I trust it. Take width scaling. If I widen the network while keeping depth and resolution at baseline, accuracy saturates early — that's the curve I just described. Now widen it again, but this time on top of a network that's already deeper *and* run at higher resolution. If the dimensions were independent, the width curve should look the same — width is width. It doesn't. With more depth and bigger images underneath it, width scaling keeps paying off and lands at noticeably higher accuracy at the *same* compute cost. That's the tell. The three knobs are not independent.

And once I see it, the *why* is almost forced. A bigger input image has more pixels spread over each object. To make use of those extra pixels I need a larger receptive field — which means more layers, i.e. more depth — so a unit late in the network can actually see a whole object's worth of the bigger image. And I need more channels — more width — to represent the extra fine-grained patterns that a higher-resolution image now contains. So if I bump resolution, depth and width *want* to come along. Scaling resolution alone is like buying a bigger canvas and refusing to buy more paint or a longer brush. They have to move together.

That reframes the whole problem. The goal isn't "which single knob do I push." It's: given a fixed extra compute budget, how do I *distribute* it across depth, width, and resolution so they stay in balance? If I write the unrestricted version, it is ugly immediately. Fix a baseline list of operators F_i, then choose scaled repeats, channels, and spatial sizes:

  maximize   Accuracy(N(d, w, r))
  subject to Memory(N) <= target_memory
             FLOPS(N)  <= target_flops

where N is the stage composition built from those F_i. If each stage gets its own independent L_i, C_i, H_i, W_i, the search space is huge; worse, every larger budget would ask me to solve that huge search again. I want a rule, not a fishing expedition. People have balanced two or three dimensions before, but by hand, tuning each independently, which is tedious and lands on configurations that are mediocre for both accuracy and FLOPS. I need to collapse the search without ignoring the coupling.

Let me try the most aggressive simplification first and see if it survives scrutiny: scale *every* dimension by a *constant ratio*. One single user-facing knob — call it φ, "how much extra budget" — and three fixed exponents α, β, γ that say how that budget gets split:

  depth   d = α^φ
  width   w = β^φ
  resolution r = γ^φ

with α, β, γ ≥ 1 (I only ever want to grow). Now the entire scaling problem, the whole giant per-layer space, has been crushed down to: pick three constants α, β, γ once, then turn a single dial φ. That's the ambition. The question is whether constants like this can possibly be right — whether one fixed split works across budgets — and whether φ can be made to *mean* something clean. Let me work on the second part first, because it's pure arithmetic and it'll tell me how to constrain α, β, γ.

I want φ to be a budget dial. The natural budget is FLOPS. So I need to know: if I scale depth by d, width by w, resolution by r, what happens to total FLOPS? Let me derive it from the convolution, since convs dominate the compute in these networks and the rest is noise by comparison.

Take one convolutional layer producing an H×W output with C_in input channels and C_out output channels and a k×k kernel. Its cost is, per output pixel, k²·C_in·C_out multiply-adds, and there are H·W output pixels, so the layer costs k²·C_in·C_out·H·W. Now scale the three knobs and watch each factor.

Depth: scaling depth by d means d times as many layers. FLOPS is a sum over layers, so it scales linearly: ∝ d.

Width: scaling width by w means every layer's channel counts go up by w — both C_in and C_out. The per-layer cost has the *product* C_in·C_out, so each of those picks up a factor of w, and the layer cost scales by w·w = w². So ∝ w².

Resolution: scaling resolution by r means each spatial side grows by r, so the output map has H·W → (r·H)·(r·W), a factor of r². The per-layer cost is proportional to H·W, so it scales by r². So ∝ r².

Put them together. Total FLOPS scales by

  d · w² · r².

That asymmetry is worth pausing on. Depth is *linear* in FLOPS; width and resolution are *quadratic*. Doubling depth doubles cost; doubling width or resolution quadruples it. Depth is the cheap dimension per unit of scaling. I'll want to remember that when I pick the constants.

Now substitute the constant-ratio form d = α^φ, w = β^φ, r = γ^φ:

  FLOPS scales by (α^φ) · (β^φ)² · (γ^φ)² = α^φ · β^{2φ} · γ^{2φ} = (α · β² · γ²)^φ.

There it is. The total compute multiplier is (α·β²·γ²) raised to the φ. Which means the entire compute behavior of my knob is governed by the single number α·β²·γ². If I leave that number arbitrary, φ has no consistent meaning — "φ = 3" would imply different compute multiples depending on the constants. But if I *pin* that number, φ becomes a clean dial. The cleanest choice: make the per-φ multiplier exactly 2. Set

  α · β² · γ² ≈ 2.

Then FLOPS scales by 2^φ. Now φ means precisely "double the compute φ times": φ = 1 is 2×, φ = 2 is 4×, φ = 3 is 8×. That's the constraint, and it dropped straight out of the convolution FLOPS count — not imposed, *derived*. The "2" is a convention (I could have pinned it to anything), but pinning α·β²·γ² to a constant is what makes the single knob coherent, and 2 makes the dial read in powers of two. (It's approximate, because integer rounding of channels and layers nudges the real FLOPS off the ideal value.)

Notice the constraint also tells me something about how the split *should* lean. The constraint weights β and γ by their squares but α only to the first power — exactly mirroring the FLOPS asymmetry. So within a fixed compute budget, a unit of depth is "cheaper" than a unit of width or resolution. I'd expect the optimal α to come out the largest of the three, because depth gives you scaling at linear cost while width and resolution are taxed quadratically. I won't assume the values; I'll find them. But that's the shape I expect.

How do I find α, β, γ? They're just three constants, tied by one equation, so really two free numbers. The honest move is a small grid search: try a handful of (α, β, γ) triples that satisfy α·β²·γ²≈2, scale the baseline by each at φ=1, train, and keep the triple with the best accuracy. But *where* do I run that search? If I search directly on a large target model, each trial is enormously expensive; the search cost explodes with model size, which is the exact thing that made hand-balancing big models hopeless. So I do the search *once*, on the small baseline, at φ=1 (i.e. assuming just 2× resources, the smallest interesting step). I get α, β, γ from the small model, then I *freeze them* and just turn φ to generate the whole family of larger models. The search is paid once, at small-model prices, and amortized over every size — find the ratios cheaply on the small model, then reuse them for all scales.

There's an assumption hiding in "reuse the same ratios at every scale," and I should name it rather than pretend it isn't there. I'm betting that the *best balance* between depth, width, and resolution is roughly scale-invariant — that the optimal split found at 2× is still a good split at 8× or 32×. That's not obviously true. But it's plausible exactly because the coupling I found is structural (bigger images → bigger receptive field + more channels), and structural relationships shouldn't flip as I scale uniformly. And it's the only thing that makes the search tractable. So I'll adopt it, with the understanding that searching the ratios at every scale would be marginally better and ruinously expensive.

Now I have to confront something I've been quietly leaning on: the baseline. My scaling rule never touches the per-layer operator F_i — it only changes how many layers, how many channels, how big the image. So whatever accuracy ceiling the baseline's *building block* has, scaling inherits. If the block is weak, I'm efficiently scaling a weak thing. So a strong, efficient baseline isn't a side detail; it sets the whole ceiling. I should build the best small baseline I can and *then* scale it.

What should the building block be? I want maximum accuracy per FLOP at small size, because that's the regime I'm searching in and the seed I'm scaling from. Let me reason about the convolution cost again, because it points the way. A plain k×k conv from M to N channels over a D×D map costs k²·M·N·D². The expensive coupling is that single k×k filter mixing *all* M inputs into *all* N outputs at once — spatial filtering and channel mixing welded together and multiplied. Break them apart. First a *depthwise* conv: one k×k filter per channel, no cross-channel mixing, cost k²·M·D². Then a *pointwise* 1×1 conv that does only channel mixing, cost M·N·D². The sum is k²·M·D² + M·N·D², which is the original times (1/N + 1/k²) — for a 3×3 kernel and any reasonable N that's about an eighth or ninth of the cost. Depthwise separable convolution. That's my cheap primitive; the whole block should be built from it.

But there's a subtlety in how to arrange the channels around it, and getting it wrong loses accuracy. The instinct from residual nets is: keep a wide representation, squeeze it down inside the block, do the work, expand back — a bottleneck. Let me think about whether that's right *here*, with depthwise convs and a tight FLOP budget. The depthwise conv does its spatial filtering per-channel; it's cheap, so I can afford to run it on *many* channels. The pointwise convs are where the channel-mixing cost lives. So the efficient shape is actually the *inverse* of the classic bottleneck: keep the representation *thin* between blocks (few channels — cheap to carry, cheap to skip-connect), and inside the block *expand* up to a wide space, do the cheap depthwise filtering there, then *project* back down to thin. Expand → depthwise → project. The skip connection joins the thin endpoints, which is cheap to add and — importantly — means the wide expanded tensor never has to be carried from block to block, so memory stays low.

One more trap, and it's a sharp one. After I project from the wide space back down to the thin bottleneck, should I put a ReLU there? Reflexively yes — every conv gets an activation. But think about what ReLU does in a *low-dimensional* space. ReLU zeros out negatives; in a narrow representation, zeroing channels collapses the data onto a low-dimensional face and you cannot recover what was lost — the information is gone, not just folded. In the *wide* expanded space there's enough redundancy that ReLU's clipping is survivable, even useful. But on the *thin* projection output, a nonlinearity is destructive. So: keep the activation on the expanded depthwise stage, and leave the projection back to the bottleneck *linear* — no activation. A linear bottleneck. That's not an aesthetic choice; it's preventing information destruction exactly where the representation is too narrow to absorb it.

So the block is: 1×1 expand (by some factor t) with activation → k×k depthwise with activation → 1×1 project, linear → residual skip on the thin endpoints when shapes match. What's the expansion factor t? If t is too small, I am back to doing nonlinear spatial filtering in a cramped representation; if t is too large, the two pointwise convolutions dominate the FLOPS and I lose the whole efficiency argument. A mid-single-digit expansion gives the depthwise convolution enough room while keeping the pointwise cost controlled, and 6× is the practical balance inherited from the inverted-residual mobile block. The one exception is the very first block: it starts immediately after the stem, is already narrow, and does not need an expansion phase, so there I set t = 1 and skip the expansion conv entirely.

Now, channels in a feature map aren't equally useful for a given image, and I have a near-free way to exploit that. After the depthwise conv, collapse each channel's whole spatial map to one number by global average pooling — a per-channel global summary. Push that channel-descriptor vector through a tiny two-layer bottleneck (one fully-connected layer that squeezes the channel count down by a ratio, then one that brings it back up), end in a sigmoid, and use the result as a per-channel multiplicative gate on the feature map. The network learns, from global context, to turn channels up or down for this particular input. The compute is negligible — a couple of FC layers on a length-C vector — and it consistently helps. I'll squeeze relative to the block's *input* channel count, by a ratio of about a quarter, so the gating bottleneck stays small. Drop this squeeze-and-excite gate into every block, between the depthwise conv and the projection.

For the activation, a hard ReLU throws away all negative response at once. In these narrow efficient blocks, that is a little harsher than I want. The smooth x·σ(x) keeps a ReLU-like gate for large positives, lets small negatives fade rather than disappear abruptly, and has already proved useful as a drop-in convolutional activation, so I will use that throughout. In code I can call it SiLU or Swish; it is the same function.

Now the baseline body itself — the stage table. Rather than hand-draw it, I'll let a search find it, because searching is exactly how you find a good *small* network and I want the seed to be as strong as possible. Set up a multi-objective search that rewards accuracy but penalizes compute: maximize ACC(m)·(FLOPS(m)/T)^w, where T is a target FLOPS and w is a small negative exponent (around −0.07) that softly trades accuracy against cost. (I optimize FLOPS rather than measured latency because I'm not targeting one specific device.) Let the search pick, per stage, the kernel size (3×3 or 5×5), the expansion factor, the number of layers, the channel counts, whether to include the squeeze-and-excite gate. Aim the target at roughly 400M FLOPS — small, but a touch bigger than a phone-only model, since this is the seed I'll scale up. Out comes the baseline; call this φ=0 model B0.

What it produces, concretely: a 3×3 stride-2 stem to 32 channels, then seven stages of these expand-depthwise-project blocks. The first is the t=1 special case (3×3 kernel, 16 output channels, one layer). Then t=6 blocks: 3×3 to 24 channels (two layers, downsample), 5×5 to 40 (two layers, downsample), 3×3 to 80 (three layers, downsample), 5×5 to 112 (three layers, no downsample), 5×5 to 192 (four layers, downsample), 3×3 to 320 (one layer). Every block carries the squeeze-and-excite gate at ratio 0.25. Then a 1×1 conv up to 1280 channels, global average pool, and a fully-connected classifier. That's the thing I'm going to scale.

Now back to scaling, with a concrete baseline in hand. I fix φ=1, pretend I have 2× the budget, and grid-search α, β, γ under α·β²·γ²≈2 on B0. The search lands on α=1.2, β=1.1, γ=1.15. Let me sanity-check the constraint: β² = 1.21, γ² = 1.3225, and 1.2 · 1.21 · 1.3225 = 1.92. That's close to 2, and the small slack is useful because integer channels, integer repeats, and friendly image sizes will nudge the realized FLOPS anyway. α is the largest of the three, exactly as the FLOPS asymmetry predicted: depth is the linear-cost dimension, so the optimizer leans on it hardest; width and resolution, taxed quadratically, get smaller exponents. The structure I derived from the cost equation showed up in the search result.

Then I freeze α=1.2, β=1.1, γ=1.15 and use them as the balanced direction for the whole family. The exact powers are targets, not sacred decimals: α^φ, β^φ, and 224·γ^φ tell me the neighborhood, and the deployable model uses channel counts, repeats, and image sizes snapped to clean values. That matters most for the first scaled model, where the ideal φ=1 target would be depth 1.2, width 1.1, and resolution about 258, but a friendlier 240 input and a slightly lighter depth/width setting land closer to the intended real compute. The named family therefore uses these practical coefficients: B0 (1.0, 1.0, 224), B1 (1.0, 1.1, 240), B2 (1.1, 1.2, 260), B3 (1.2, 1.4, 300), B4 (1.4, 1.8, 380), B5 (1.6, 2.2, 456), B6 (1.8, 2.6, 528), B7 (2.0, 3.1, 600). They are still on the same balanced path: depth climbs fastest, width grows slowest, and resolution sits between them because the compute law taxes width and resolution quadratically.

Two implementation details I have to get right or the clean math turns ugly in practice. First, when I scale width I can't just multiply channel counts by β^φ and take whatever real number falls out — hardware wants channel counts that are multiples of a small number (8), and I don't want a rounding that drops more than ~10% of the intended width. So: multiply, round to the nearest multiple of 8, and if that rounded value fell more than 10% below the target, bump it up by one more multiple of 8. Second, when I scale depth I multiply the per-stage layer count by α^φ and round *up* (ceiling), so depth scaling never silently deletes a layer. Small things, but they're the difference between the formula and a network that actually builds.

One last regularization point follows from scaling itself. The bigger models in the family have far more capacity, so they need more regularization or they will overfit. I scale dropout before the classifier from 0.2 at B0 to 0.5 at B7, and I use stochastic depth on residual branches. In the implementation that is a drop-connect rate of 0.2 at the deepest end, multiplied by the block index divided by the total number of blocks, so early blocks are almost always kept and later blocks are dropped more often. The residual output is divided by its keep probability before masking, so the expected branch contribution stays unchanged. The rest of the training recipe is the efficient-network standard: RMSProp, weight decay, AutoAugment, and the same batch-norm convention as the TensorFlow-style implementation.

Let me now write the whole thing as code, because that's where the reasoning has to actually close. The block is expand→depthwise→squeeze-excite→project-linear with a thin-endpoint skip; the body is the searched stage table; and the scaler turns φ into width/depth/resolution and rounds them safely.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- safe rounding for the scaling rule ---------------------------------------
def round_filters(channels, width_coeff, divisor=8):
    # width scaling: multiply by beta^phi, then snap to a multiple of `divisor`
    # (hardware-friendly), never dropping more than ~10% of the intended width.
    if not width_coeff:
        return channels
    channels *= width_coeff
    new_c = max(divisor, int(channels + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * channels:          # rounded down too far -> bump up one step
        new_c += divisor
    return int(new_c)


def round_repeats(num_layers, depth_coeff):
    # depth scaling: multiply per-stage layer count by alpha^phi, round UP so
    # scaling never silently removes a layer.
    if not depth_coeff:
        return num_layers
    return int(math.ceil(depth_coeff * num_layers))


# --- the building block: inverted residual, linear bottleneck, SE gate --------
class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride,
                 expand_ratio=6, se_ratio=0.25):
        super().__init__()
        self.use_skip = (stride == 1 and in_ch == out_ch)   # thin-endpoint skip
        mid = in_ch * expand_ratio                           # the wide space
        layers = []

        # 1x1 expand to the wide space (skipped when expand_ratio == 1)
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, mid, 1, bias=False),
                       nn.BatchNorm2d(mid), nn.SiLU()]

        # kxk depthwise: cheap per-channel spatial filtering in the wide space
        layers += [nn.Conv2d(mid, mid, kernel_size, stride,
                             padding=kernel_size // 2, groups=mid, bias=False),
                   nn.BatchNorm2d(mid), nn.SiLU()]
        self.expand_dw = nn.Sequential(*layers)

        # squeeze-and-excite: global context -> per-channel gate (squeeze on the
        # block's INPUT channel count, ~1/4)
        se_dim = max(1, int(in_ch * se_ratio))
        self.se_reduce = nn.Conv2d(mid, se_dim, 1)
        self.se_expand = nn.Conv2d(se_dim, mid, 1)

        # 1x1 project back down to thin -- LEFT LINEAR (no activation) so the
        # narrow representation isn't destroyed by a nonlinearity.
        self.project = nn.Sequential(nn.Conv2d(mid, out_ch, 1, bias=False),
                                     nn.BatchNorm2d(out_ch))

    def forward(self, x):
        h = self.expand_dw(x)
        s = F.adaptive_avg_pool2d(h, 1)                 # squeeze
        s = self.se_expand(F.silu(self.se_reduce(s)))   # excite
        h = torch.sigmoid(s) * h                        # gate channels
        h = self.project(h)
        if self.use_skip:
            h = h + x                                   # residual on thin endpoints
        return h


# --- the searched baseline body (B0): (expand, kernel, stride, in, out, repeats)
BASE_STAGES = [
    (1, 3, 1,  32,  16, 1),
    (6, 3, 2,  16,  24, 2),
    (6, 5, 2,  24,  40, 2),
    (6, 3, 2,  40,  80, 3),
    (6, 5, 1,  80, 112, 3),
    (6, 5, 2, 112, 192, 4),
    (6, 3, 1, 192, 320, 1),
]


class EfficientNet(nn.Module):
    def __init__(self, width_coeff=1.0, depth_coeff=1.0,
                 dropout=0.2, num_classes=1000):
        super().__init__()
        stem = round_filters(32, width_coeff)
        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(stem), nn.SiLU())

        blocks, in_ch = [], stem
        for expand, k, stride, b_in, b_out, repeats in BASE_STAGES:
            out_ch = round_filters(b_out, width_coeff)        # scale width
            for i in range(round_repeats(repeats, depth_coeff)):  # scale depth
                blocks.append(MBConv(in_ch, out_ch, k,
                                     stride if i == 0 else 1, expand))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        head = round_filters(1280, width_coeff)
        self.head = nn.Sequential(nn.Conv2d(in_ch, head, 1, bias=False),
                                  nn.BatchNorm2d(head), nn.SiLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(head, num_classes)

    def forward(self, x):          # x fed at the resolution gamma^phi prescribes
        x = self.blocks(self.stem(x))
        x = self.pool(self.head(x)).flatten(1)
        return self.fc(self.dropout(x))


# --- the scaling rule: one knob phi -> (width, depth, resolution) -------------
# constants from the once-only grid search on the baseline, under alpha*beta^2*gamma^2 ~= 2
ALPHA, BETA, GAMMA, BASE_RES = 1.2, 1.1, 1.15, 224

def scale(phi, dropout=0.2):
    depth_coeff = ALPHA ** phi          # d = alpha^phi
    width_coeff = BETA ** phi           # w = beta^phi
    resolution = int(round(BASE_RES * (GAMMA ** phi)))  # r = gamma^phi
    model = EfficientNet(width_coeff, depth_coeff, dropout=dropout)
    return model, resolution            # FLOPS ~ (alpha*beta^2*gamma^2)^phi ~= 2^phi
```

So the causal chain, end to end: three saturating single-knob curves that all plateau in the same place tell me the knobs are coupled, and the width-at-(deeper, higher-res) test confirms it — bigger images demand more receptive field and more channels, so depth, width, and resolution must move together. To make "move together" a rule rather than a hand-search, I scale all three by constant ratios off one budget knob φ; counting convolution FLOPS gives d·w²·r², so the per-φ compute multiplier is α·β²·γ², and pinning that to 2 makes φ a clean doubling dial — and its quadratic weighting of width and resolution versus linear weighting of depth predicts (correctly) that depth should get the largest exponent. Because scaling can't fix a weak block, I build the strongest small baseline I can — depthwise-separable inverted-residual blocks with linear bottlenecks and squeeze-excite gates, found by a FLOPS-aware search — then grid-search α, β, γ once on that small model and reuse the ratios at every φ, paying the search cost once and amortizing it across the whole family.
