Stochastic depth is the strongest rung so far, and the *way* it is strong tells me what it costs and where
the next move has to come from. On the deep CIFAR-100 nets it did exactly what the ensemble argument
promised: ResNet-110 jumped from the pre-activation 74.08 to 75.70 — the biggest lift any rung has produced
on any setting — and ResNet-56 recovered from its 71.78 stall all the way to 74.56, decisively. The benefit
sorted by depth, just as predicted: largest where `L` is largest. But read the shallow number and the trade
is laid bare. ResNet-20/CIFAR-10 fell to 91.37 — the lowest CIFAR-10 result of the entire ladder, below the
gated 92.96 and the pre-activation 92.62. That is the honest cost I flagged: at depth 20 there are few
blocks to drop, the depth-ensemble is small, and dropping branches on a net already near its CIFAR-10
capacity ceiling just *removes* capacity for too much of training. So I now have a ladder where every rung
trades one setting against another, and — more importantly — every rung so far has pulled the *same* lever:
the residual *flow and depth* axis. The gate rescaled the branch; pre-activation reordered the path;
stochastic depth dropped the branch. Three different ways of asking "how much, how cleanly, how often does
the residual branch contribute." None of them ever asks a different question: *which features* the branch
should emphasize for a given image. That axis is completely untouched, and it is orthogonal to depth, which
is exactly the property I want — something that can lift the deep nets without paying the shallow tax that
dropping does.

Let me go back to the convolution itself, because the gap is in the primitive every block is built from.
A 3×3 conv produces each output channel by convolving each input channel with its own small kernel and
summing across channels. Two facts about that fall out immediately. First, the channel mixing — how much
output channel `c` listens to input channel `s` — is baked into the kernel weights at training time and is
then *identical for every image*. The block has no way to say "this particular image is a picture of a dog,
so the fur-texture channels matter and the sky-blue channels do not." Second, the mixing is *local*: a 3×3
kernel integrates a tiny spatial neighborhood, so even though stacking blocks grows the receptive field,
each block decides how to combine channels while looking through a keyhole, with no image-wide context. So
the block's channel responses are static and locally-decided. The deep CIFAR-100 nets, fitting 100
fine-grained classes, are precisely where input-dependent channel emphasis should pay: many classes share
low-level features and differ in which high-level channels are salient, and a block that could *recalibrate*
its channels per image — turn up the discriminative ones, turn down the irrelevant ones, using global
context — would add a kind of capacity none of the flow-axis rungs can.

So the design goal: give the block a lightweight, *input-dependent* gate over its channels, informed by
*global* (image-wide) context, applied to the branch output before it joins the identity stream. Three
moves make that concrete. First, *squeeze* the spatial dimension away to get a global per-channel summary:
global average pooling over `H×W` turns the branch's `C×H×W` feature map into a length-`C` descriptor whose
every entry has a full-image receptive field — the global context the conv lacked. Second, *excite*: feed
that descriptor through a tiny bottleneck MLP — a linear `C → C/r`, a ReLU, a linear `C/r → C`, a sigmoid —
to produce one gate per channel in `(0,1)`. The bottleneck (reduction ratio `r`) keeps the added capacity
tiny and forces the gate to summarize channel interactions rather than memorize them, which aids
generalization; the sigmoid (not a softmax) is deliberate, because channels are *not* mutually exclusive —
several should be able to fire at once, so I want independent per-channel gates, not a competition for a
fixed budget. Third, *scale*: multiply each channel of the branch output by its gate. Because the gates live
in `(0,1)` the block can attenuate uninformative channels and preserve informative ones, image by image — a
self-attention over channels with global reach, costing only the two small FC layers.

The placement decision is the one that ties this back to everything the ladder has learned, and I want to be
precise about it. The recalibration must act on the **branch output, before the identity addition** — never
on the shortcut. This is the same principle that ran through every rung: the identity highway stays open and
ungated, so signal and gradient pass through it untouched; only the residual branch is modulated. Gating the
sum, or the shortcut, would reintroduce exactly the Highway failure (a gate that suppresses the through-line)
and would undo the clean additive path the whole ladder depends on. So the SE gate sits at the end of the
branch, between the second conv's BN and the addition: `H = ReLU(shortcut(x) + SE(F(x)))`. The shortcut is
the unchanged dimension-matching identity-or-projection. And note what this buys at the shallow end where
stochastic depth hurt: SE adds capacity *without removing any* — there is no dropping, the full net trains
every step, so ResNet-20/CIFAR-10 should recover from the 91.37 dip rather than pay it, while the
per-image channel gate still gives the deep CIFAR-100 nets a lever orthogonal to the depth-ensemble they
already benefit from.

Now the substrate-specific care, because the block I have to fill is the task's basic block and the SE
geometry has to fit its narrow channel counts. The block is post-activation here:
`out = ReLU(BN(conv1(x)))`, `out = BN(conv2(out))`, then I insert the SE recalibration on `out`, then add
the shortcut, then the final ReLU. The squeeze is `AdaptiveAvgPool2d(1)` to a length-`planes` vector; the
excitation is the two-layer bottleneck. The reduction ratio needs care: the CIFAR ResNet stages are 16, 32,
64 channels wide, and a textbook `r = 16` would make the first stage's bottleneck `16/16 = 1` channel — a
degenerate one-dimensional squeeze that throws away almost all channel structure. So the bottleneck width is
floored: `mid = max(planes // 16, 4)`, guaranteeing at least 4 hidden units even in the 16-channel stage.
That floor is the right adaptation of the SE recipe to these thin CIFAR widths; without it the shallow
stages would get no usable recalibration. The two FC layers carry biases (the default `nn.Linear`), the
hidden activation is ReLU, the output activation is sigmoid, and the gate is reshaped to `(B, C, 1, 1)` and
broadcast-multiplied against `out`. The added cost per block is `2·planes²/r` parameters — for a 64-channel
block at `r = 16` that is a few hundred weights, utterly negligible against the conv weights, and the FLOPs
rise is on length-`C` vectors not `H×W` maps, so it is essentially free. The full scaffold module is in the
answer.

So the edit relative to stochastic depth is a *different axis* entirely: keep the proven post-activation
block at full depth (no dropping, no reordering, no branch scalar), and insert a squeeze-excite channel gate
on the branch output before the addition. It does not compete with the flow-axis tricks; it composes with
the plain block by adding the one capability none of them had — per-image, global-context channel emphasis.

This is the endpoint, so let me state the bar it must clear and exactly what I would validate, against the
strongest baseline's measured numbers. Stochastic depth set the marks at 91.37 (ResNet-20/CIFAR-10), 74.56
(ResNet-56/CIFAR-100), and 75.70 (ResNet-110/CIFAR-100). For SE to be a genuine improvement and not just a
sideways trade, the falsifiable claims are setting-specific. First and most diagnostic: ResNet-20/CIFAR-10
should *recover* well above 91.37 — back toward or past the gated 92.96 — because SE adds capacity without
the dropping that cost stochastic depth the shallow case; if SE does *not* fix the shallow number, then its
channel gate is not buying real capacity and the whole orthogonal-axis premise is wrong. Second, on the deep
CIFAR-100 nets SE must at least *hold* stochastic depth's strong 74.56 and 75.70 and ideally edge them,
since channel recalibration is orthogonal to the depth-ensemble and a 100-class problem is where per-image
channel emphasis should help most; if SE *underperforms* stochastic depth on the deep nets, then on
CIFAR-100 the binding constraint really was depth-regularization (which SE does not provide) rather than
channel selectivity, and the right endpoint would be to *combine* SE's gate with stochastic depth's
dropping rather than substitute it. The decisive test of the endpoint, then, is whether one block design can
simultaneously clear 91.37 on the shallow easy case *and* hold ~74.5/~75.7 on the deep CIFAR-100 cases — a
net positive across the whole depth sweep that no single flow-axis rung managed, because each of those
traded one end of the sweep against the other. That balance — recover the shallow capacity that dropping
removed while keeping the deep-net strength — is precisely what an orthogonal, capacity-adding,
depth-agnostic channel gate is built to deliver, and it is the claim the endpoint stands or falls on.
