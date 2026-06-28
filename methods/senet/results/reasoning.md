OK, let me think this through from scratch. I keep coming back to one fact about the convolution, the unit everything in these networks is built out of. Write out what a single conv layer actually computes. The output has C channels; the c-th output map is

    u_c = v_c * X = sum over s of (v_c^s * x^s),

where X has C' input channels x^1..x^{C'} and v_c = [v_c^1, ..., v_c^{C'}] is the c-th filter. So to produce one output channel I convolve each input channel with its own 2D spatial kernel and sum the results across all input channels. Stare at that summation for a second. The "channel relationship" — how much output channel c listens to input channel s — is exactly the magnitude/shape of the kernel v_c^s. It's there, but it's two things tangled together: the same weights that decide *which input channels matter* also decide *what local spatial pattern to look for*. The channel mixing is entangled with the spatial filtering, and it's baked in at training time. Two consequences bug me.

First, it's local. v_c^s is a small kernel — 3x3, say. At every spatial position the output only ever integrates a small neighbourhood of the input. Yes, if I stack enough layers the theoretical receptive field grows until eventually, near the very top, a unit can in principle see the whole image. But in the bulk of the network, where most of the channel mixing actually happens, each unit is deciding how to combine channels while looking through a keyhole. It has no image-wide view to tell it "this is a picture of a dog, so the fur-texture channels matter and the sky-blue channels don't."

Second, it's static and instance-agnostic. Once trained, v_c is fixed. Every image gets the same channel mixing. There's no knob that says, for *this particular input*, turn channel 37 up and channel 112 down.

So here's the thought I want to chase: a conv filter mixes channels with fixed weights and has no global view. Could the network learn to emphasise the channels that are informative *for the current image*, using global context, cheaply? Let me be careful about what I'm even asking for, because it would be easy to reinvent something heavy. I want something that (a) sees beyond the local receptive field — global, image-wide; (b) is conditioned on the input, so it adapts per image rather than applying one fixed transform; (c) is cheap enough to drop into an existing strong backbone — ResNet, Inception — without re-tuning the whole architecture; and (d) doesn't fight the network — it should *modulate* the features the backbone already computes, not replace them.

Most of the effort in the field has gone the other way — into the spatial side. Inception throws several receptive-field sizes at each location and concatenates; spatial-transformer modules and trunk-and-mask attention learn *where* in the image to look. That's all "improve the spatial encoding." The channel axis has been treated mostly as something to make cheaper — grouped convolutions, 1x1 convolutions to remap channels, depthwise-separable factorisations — and crucially all of those are still instance-agnostic functions over local receptive fields. Nobody's giving the unit an explicit, dynamic, global handle on *which channels* matter. That gap is exactly (a)+(b). Let me push on it.

Suppose I have the output feature maps U of some transformation — for now think of it as the output of a conv layer, U in R^{H x W x C}, channels u_1..u_C, each u_c an H x W map. I want to produce, per channel, a number that says "how much should this channel be turned up or down for this image," and I want that number to depend on global context. Two sub-problems fall out immediately and I should solve them separately: how do I get global, image-wide information into a per-channel quantity at all; and then, given that, how do I turn it into modulation weights.

Take the first one. The problem with u_c is that it's spatial — H x W of it — and it's only ever been touched by local filters, so any single location in it carries local information. But I don't want a location; I want a single number per channel that summarises the channel over the *whole* spatial extent. The crudest possible way to get global reach is to collapse the entire H x W map into one scalar. And the simplest collapse that uses every spatial location equally is to average them. So define

    z_c = (1/(H·W)) · sum_{i=1..H} sum_{j=1..W} u_c(i, j).

That's just global average pooling of channel c. z is a length-C vector, one scalar per channel. Why does this earn its keep? Because that single scalar's "receptive field" is the entire feature map — it has, by construction, global spatial context, the thing the local convolutions could never give a unit in the middle of the network. It's a channel descriptor: an embedding of the global distribution of that channel's response over the image. And it costs nothing — no parameters, one pooling op. I'll call this the squeeze: squeeze each H x W channel down to its global summary scalar.

Should I worry I'm throwing away too much by averaging? I am throwing away spatial layout, sure. But that's fine — I'm not trying to localise anything; I'm trying to answer "to what extent is this channel active across the image," and the mean response is a clean answer. I could imagine fancier aggregations — max instead of mean, or higher-order statistics — but the mean is the natural first choice: it weights every location, it's smooth, it's differentiable, and it directly estimates the channel's average activation. The difference from max is worth being concrete about, because it changes what gets summarised. Take a tiny 2x2 channel that's strongly on in three locations and zero in one: u = [[4,4],[4,0]]. Its mean is (4+4+4+0)/4 = 3; its max is 4. Now a channel that spikes once and is otherwise dead: u = [[4,0],[0,0]], mean (4+0+0+0)/4 = 1, max 4. Max calls these two channels identical (both 4) — it can't tell a broadly-active channel from a single hot pixel — whereas the mean separates them, 3 vs 1, exactly tracking how much of the image the channel covers. Since the quantity I'm after is global *presence*, the mean is the descriptor that actually measures it; max would conflate presence with a single peak. I'll take the mean and note max as a variant to sweep empirically. And there's prior support for the idea that globally pooled statistics of local descriptors are expressive for whole-image recognition — that's the whole logic of spatial-pyramid and Fisher-vector features. So: squeeze = global average pool, z in R^C.

Now the second sub-problem, the interesting one. I have z, a global per-channel descriptor. I want from it a set of per-channel modulation weights s in R^C — one gate per channel, which I'll use to scale the channels. What should the map z -> s look like? Let me write down the requirements as constraints and let the form fall out.

It has to be a *learned* function of z, because "which channels matter given this global summary" is exactly the relationship I want the network to discover; a hand-fixed rule won't do. It has to be *nonlinear* — if it were linear, z -> s = Wz, then composed with the linear pooling and the linear scaling I'd basically be back to a fixed-ish linear reweighting, and I want it to capture genuinely nonlinear interactions between channels (channel A matters only when channel B is also active, that kind of thing). And it has to model *interactions across channels*, not treat each channel in isolation — the whole point is interdependencies — so s_c should be allowed to depend on all of z, not just z_c.

The simplest object that is learned, nonlinear, and lets every output coordinate depend on every input coordinate is a small multilayer perceptron on z. So: a fully-connected layer C -> something, a nonlinearity, a fully-connected layer back to C. That gives me s = g(z) with full cross-channel coupling and a nonlinearity in the middle. Good. Now I have to nail down three things: the hidden width, the inner nonlinearity, and the output nonlinearity. Let me take them in the order the constraints force.

Output nonlinearity first, because it interacts with how I'll *use* s. I'm going to use s to scale channels, and a scale wants to live in a bounded, well-behaved range — I don't want gates that can blow a channel up by 50x or flip its sign arbitrarily and destabilise the backbone I'm trying to gently improve. So squash each s_c into (0,1). The obvious candidates are sigmoid (per-coordinate logistic) and softmax (normalise across channels). Softmax is tempting because it's "the" attention nonlinearity, so before I default to sigmoid I should actually see what softmax does to gates in the regime I care about. The regime that matters is "several channels are informative at once" — edges *and* texture *and* colour all wanted. Model that as two channels whose excitation logits are both high and equal, say both 4.0. Softmax over [4, 4] gives [e^4/(e^4+e^4), e^4/(e^4+e^4)] = [0.5, 0.5]: the two gates are forced to sum to one, so two equally-informative channels each get *half* a gate, and if a third equally-informative channel joins they drop to 0.33 each. Softmax is dividing a fixed budget of attention; the more channels deserve to fire, the less each is allowed to. Now sigmoid on the same logits: sigma(4) = 1/(1+e^-4) = 0.982 for *each* channel, independently — both stay near fully open, and adding a third changes nothing for the first two. That's the behaviour I need: the channels aren't competing for a shared budget, so emphasising one must not require suppressing another. The computation makes the choice for me — softmax's normalisation is structurally wrong for "many channels useful simultaneously," and the per-channel sigmoid is right:

    s_c = sigma( g(z)_c ),  sigma the logistic sigmoid, independently per channel, each in (0,1).

That's a self-gating mechanism: the unit computes its own multiplicative gates from its own global summary, and nothing ties one gate to another.

Inner nonlinearity: ReLU. It's the standard, it's cheap, it gives the needed nonlinearity between the two FC layers without saturating on the positive side. I'll revisit whether a different inner nonlinearity matters, but ReLU is the default and I have no reason yet to deviate.

Now the hidden width, and this is where I have to think about cost, because requirement (c) was "cheap." Naively, the richest map z -> s is a single full C x C matrix (then sigmoid): that's C^2 parameters *per SE block*. In a ResNet-50 the later stages have C in the thousands, and there are many blocks; C^2 per block, summed over the network, is a lot of parameters — it would balloon the model and almost certainly overfit, defeating the "lightweight drop-in" goal. So I don't want a full C x C transform. Put a bottleneck in the middle: reduce C down to C/r with the first FC layer, apply ReLU, then expand C/r back to C with the second FC layer. Let r be a reduction ratio.

    s = sigma( W_2 · delta( W_1 z ) ),   delta = ReLU,
    W_1 in R^{(C/r) x C},  W_2 in R^{C x (C/r)}.

Count the parameters. W_1 has (C/r)·C entries, W_2 has C·(C/r) entries, so per block it's

    (C/r)·C + C·(C/r) = 2C^2 / r

(no biases — I'll drop the FC biases; empirically they get in the way of modelling the channel dependencies, and they're negligible in count anyway). Sum over the network: if stage s has N_s blocks operating at channel width C_s, the total extra parameters are

    (2/r) · sum_s N_s · C_s^2.

So r is a direct dial on cost: it cuts the bottleneck quadratic-in-C term by a factor r, while the ReLU in the middle keeps the map nonlinear and the two layers still let every output channel depend on every input channel through the shared C/r-dimensional code. There's a tension to feel out: too large an r and the bottleneck is so narrow it can't represent the channel interactions; too small an r and I'm back toward the full C x C blow-up and the overfitting/parameter cost I was avoiding — and I'd expect performance *not* to improve monotonically as I shrink r, because past a point I'm just adding capacity that doesn't help. So there's a sweet spot; something like r = 16 feels like the right order of magnitude for a good accuracy/cost balance, to be confirmed by sweeping it.

Before I trust the "lightweight" claim I should actually total the extra parameters, not just wave at "it's small." Take ResNet-50 with r = 16. The four stages run at expanded widths C = 256, 512, 1024, 2048 with N = 3, 4, 6, 3 blocks. Per block the SE adds 2C^2/r:

    stage C=256,  3 blocks: 3 · 2·256^2/16   = 3 · 8192    =    24,576
    stage C=512,  4 blocks: 4 · 2·512^2/16   = 4 · 32768   =   131,072
    stage C=1024, 6 blocks: 6 · 2·1024^2/16  = 6 · 131072  =   786,432
    stage C=2048, 3 blocks: 3 · 2·2048^2/16  = 3 · 524288  = 1,572,864

Summing: 24,576 + 131,072 + 786,432 + 1,572,864 = 2,514,944 ≈ 2.5M extra parameters. ResNet-50 itself is about 25.6M, so this is 2.51/25.6 ≈ 9.8% more — call it a ~10% bump. That lands where I'd hoped, "lightweight," and it confirms the math actually computes the number rather than my having assumed it. Two things jump out of the breakdown. The quadratic-in-C term means the cost is wildly concentrated: the final stage alone is 1,572,864 of 2,514,944, i.e. 62.5% of all the extra parameters sit in the last three blocks where C = 2048. So if parameters were tight, dropping SE from just that final stage cuts the extra down to 2,514,944 − 1,572,864 = 942,080 ≈ 0.94M, only ~3.7% over the base — a 60%+ saving in the SE overhead for losing recalibration in only the top stage, which I'd bet costs little accuracy (worth a direct test). And the FLOP increase is trivial regardless, because every one of these FC layers acts on a length-C vector, not on an H x W map — the pooling has already collapsed the spatial extent — so the added multiply-adds are negligible next to the conv FLOPs.

Now the third operation — actually *using* s. I have one gate s_c per channel; I want to recalibrate U with it. The simplest thing that respects "modulate, don't replace" is to scale each channel's whole feature map by its gate:

    x_tilde_c = s_c · u_c,

channel-wise multiplication of the scalar s_c into the H x W map u_c. Because s_c is in (0,1), this can only attenuate a channel (toward zero) or leave it near full strength — it emphasises the channels the global summary deemed informative and suppresses the rest, exactly the "selectively emphasise / suppress" behaviour I set out to get. Let me run the whole squeeze→excite→scale through on a toy 2-channel input to see it move actual numbers. Channel 0 broadly active, u_0 = [[4,4],[4,4]]; channel 1 nearly silent, u_1 = [[0.1,0],[0,0.1]]. Squeeze: z_0 = 4, z_1 = 0.05. Suppose the excitation has learned to roughly pass the descriptor through to its logit; then sigmoid gives gates s_0 = sigma(4) = 0.982 and s_1 = sigma(0.05) = 0.512. Scale: channel 0 becomes 0.982·4 = 3.93 per location (kept almost intact) while channel 1 becomes 0.512·0.1 = 0.051 (held down near its already-small value). The active channel survives, the dead one stays suppressed, and crucially both gates were computed *independently* — no budget shared between them — which is exactly what the sigmoid bought me over softmax. And note what this whole z -> s -> scale pipeline is: it's an input-dependent reweighting computed from global context, i.e. a self-attention over channels whose reach is the entire image, not the local receptive field the convolutions were stuck with. The three pieces — squeeze (global average pool to z), excitation (bottleneck MLP with ReLU then sigmoid to s), scale (channel-wise multiply) — answer requirements (a) global, (b) input-dependent, (c) cheap, (d) modulatory, in that order.

Let me re-examine the squeeze once more, because the global-context claim is load-bearing and I want to be sure the averaging is what's doing the work and not just the extra parameters of the MLP. Imagine a counterfactual unit that keeps the two FC layers but *skips the pooling* — implement them as 1x1 convolutions that preserve the H x W resolution and produce a spatial map of gates. Same parameter budget, but now the gate at each location is computed only from that location's channel vector — a purely local operator again. The point of constructing this twin is that it has *identical* capacity to the real block; the only thing it removes is the global averaging. So whatever accuracy gap appears between them cannot be attributed to parameter count — it can only be the value of the global embedding, the squeeze. I can't run the ImageNet comparison here, so I won't claim the outcome; what I can say is that this is the clean ablation to defend the squeeze, because it holds parameters fixed and varies only the global-vs-local aggregation. I'd expect the pooled version to win — that's the whole premise — but it's exactly the kind of claim I'd want to confirm empirically rather than assert, and the experiment is designed so its result actually isolates the question.

Now, where does this block go? I described it on top of "the output U of a transformation," and I kept that deliberately generic, because the nice thing is the block doesn't care what produced U. Let the transformation be F_tr: X -> U. Then squeeze-excite-scale acts on U and feeds the recalibrated x_tilde onward. If F_tr is a single conv, I insert the block right after the nonlinearity following that conv. If F_tr is a whole Inception module, I take U to be the module's concatenated output and recalibrate that — one block per module. The case I care most about is the residual network, because that's the strongest backbone. A residual block computes y = x + F(x), where F is the residual branch (for the deep nets, the 1x1-reduce -> 3x3 -> 1x1-expand bottleneck with BN+ReLU between). I take F_tr to be that non-identity branch: squeeze and excite its output, scale it, and *then* add the identity. So the recalibration happens on the residual branch before summation.

Why before the sum and not after? The identity branch is the highway that carries the input through and that makes deep training work — I don't want my (0,1) gates multiplying *that* and choking the gradient flowing along the skip path. It's the residual *contribution* F(x) that I want to recalibrate, decide channel-by-channel how much of it to add. So the gate belongs on F(x), before it merges with x. If I put it after the addition I'd be scaling the merged signal including the identity, which both attenuates the carried-through input and sits on the path the next block's gradient must traverse — I'd expect that placement to be worse, and applying it before the branches aggregate to be the safe choice. (Putting it on the identity branch, or before the whole residual unit, are conceivable too and probably comparably fine, since they also leave it before aggregation — but on the residual branch before the add is the clean default.) One more efficiency thought: I could even tuck the block inside the bottleneck, right after the 3x3 conv, where the channel count is smaller (the reduced width, not the expanded width) — fewer channels means a cheaper SE for similar effect. Worth keeping in mind as a parameter-saving variant.

Let me also pin down the gate function once more now that the structure is fixed. "Use a sigmoid at the end" was forced by the non-mutually-exclusive requirement, but I should check it against the two obvious alternatives by asking what each does to the *scaled feature*, since that's what actually feeds the addition. The scale is x_tilde_c = s_c · u_c, so the gate's range is the range of the multiplier. With sigmoid, s_c in (0,1): the worst it can do to a channel is push it toward zero, and the best is leave it near unchanged — pure attenuation, and the residual branch's contribution can only shrink, never explode. Swap in tanh and s_c in (-1,1): now a gate of, say, -0.7 turns u_c into -0.7·u_c, *flipping its sign*. The block is no longer "decide how much of this channel to keep"; it can invert channels, which is a different and stronger operation that I have no reason to want and which muddies the clean attenuate-or-keep semantics — I'd want to verify it on ImageNet, but the arithmetic alone says it changes the job description. Swap in ReLU and s_c in [0, ∞): a gate of 5 turns u_c into 5·u_c. Now the residual contribution can be amplified without bound, and on a path I'm adding straight into the identity sum, an unbounded multiplier is a direct route to activation blow-up during training; meanwhile any channel whose logit goes negative is hard-zeroed and dead. Both failure modes — explode and kill — come for free from the unbounded-above range, so I'd expect ReLU gates to be the worst of the three, plausibly unstable enough to underperform the plain backbone, though that's the empirical claim I'd actually test. The common thread the arithmetic exposes: only sigmoid keeps the multiplier in a bounded sub-unity range, which is exactly the "modulate, don't replace / don't destabilise" requirement I wrote down at the start. So the sigmoid wasn't an arbitrary pick; it's the one output nonlinearity whose range matches how the gate is used.

So the final shape of one block, for a residual bottleneck: run the residual branch to get U with C channels; squeeze with global average pooling to z in R^C; excite with FC(C -> C/r), ReLU, FC(C/r -> C), sigmoid to get gates s in (0,1)^C; scale each channel of U by its gate; add the identity; final ReLU. Stack these everywhere and the recalibrations compound through depth. What I'd want to validate next is whether the gains actually accumulate across stages and how they trade against r — but the mechanism is settled. Now the code.

```python
import torch
import torch.nn as nn


class SELayer(nn.Module):
    """Squeeze-and-excitation: recalibrate channels using global context."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        # squeeze: collapse each HxW map to one global-average scalar -> z in R^C
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # excitation: bottleneck MLP z -> s, ReLU inside, sigmoid gates out.
        # 2*C^2/r params; biases dropped (they hinder the channel modelling).
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # C -> C/r
            nn.ReLU(inplace=True),                                 # delta
            nn.Linear(channel // reduction, channel, bias=False),  # C/r -> C
            nn.Sigmoid()                                           # sigma: gates in (0,1), non-competing
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        z = self.avg_pool(x).view(b, c)            # squeeze -> (b, C)
        s = self.fc(z).view(b, c, 1, 1)            # excitation -> per-channel gates
        return x * s.expand_as(x)                  # scale: channel-wise multiply


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SEBottleneck(nn.Module):
    """ResNet bottleneck with SE on the residual branch, before the add."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # SE acts on the expanded width (planes*4) = the residual-branch output
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))   # 1x1 reduce
        out = self.relu(self.bn2(self.conv2(out)))  # 3x3
        out = self.bn3(self.conv3(out))            # 1x1 expand -> U
        out = self.se(out)                         # squeeze-excite-scale on F(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual                            # add identity (ungated)
        return self.relu(out)
```

Chain it back together: the convolution mixes channels with fixed, local, instance-agnostic weights and never sees global context where the mixing happens, so I wanted a cheap way to reweight channels per image using image-wide information. Averaging each channel's H x W map to a scalar (squeeze) gives a global per-channel descriptor for free; a bottleneck MLP with a ReLU and a final sigmoid (excitation) turns that descriptor into per-channel gates that are learned, nonlinear, cross-channel-coupled, cheap (2C^2/r per block, tuned by r), and — because sigmoid not softmax — non-competing so many channels can fire at once; multiplying each channel by its gate (scale) attenuates the uninformative channels and keeps the informative ones; and dropping the whole thing onto the residual branch before the identity add lets the recalibrations stack through the network without touching the skip-connection highway.
