Stochastic depth is the strongest step, and the way it is strong tells me where the finale has to come from.
On the deep CIFAR-100 nets it did what the ensemble argument promised: ResNet-110 `74.08 → 75.70` (+1.62, the
biggest lift any step produced anywhere) and ResNet-56 `71.78 → 74.56` (+2.78, erasing the pre-activation dip
and clearing the gated 71.98 by more than two and a half points). The benefit sorted by depth, as ensemble
size predicted. But the shallow number lays the trade bare: ResNet-20/CIFAR-10 fell to 91.37 — the lowest
CIFAR-10 result of the ladder, −1.25 below pre-activation and −1.59 below the gated 92.96 — the honest cost
of dropping branches on a net already near its CIFAR-10 ceiling.

Lay the ladder out as a table and the pattern decides the finale. No single block design has held the top of
all three columns at once: the best ResNet-20/CIFAR-10 belongs to the gated block (92.96), the best ResNet-56
and ResNet-110 both to stochastic depth (74.56, 75.70). Each step wins where its mechanism is strongest and
pays where that mechanism costs — the gate matched the baseline and so topped only the easy case,
pre-activation bought deep flow at a shallow warm-up cost, stochastic depth bought the deep ensemble at a
shallow capacity cost. On CIFAR-10 the three numbers march monotonically *down* — 92.96, 92.62, 91.37 — every
structural move has quietly eroded the shallow case even as it lifted the deep ones. So I have built a
sequence of one-sided trades and never once a block net-positive across the whole depth sweep at the same
time. That is the gap the finale must close, and it will not close by pulling harder on the axis every prior
step pulled — pulling that axis is *what created the trade*.

One read of the deep pair before I leave the flow axis, because the *gap between* the CIFAR-100 nets is
cleaner than either number. Gated: `73.46 − 71.98 = 1.48`; pre-activation widened it to 2.30 (the reorder
helped the deeper net more); stochastic depth *narrowed* it to `75.70 − 74.56 = 1.14`. A narrowing looks
wrong for an ensemble until I remember 56 was partly *recovering* the pre-activation dip. Measure both from
the common gated baseline: ResNet-56 `+2.58`, ResNet-110 `+2.24` — close, the residual difference just the
dip 56 climbed out of. So the depth-ensemble lifted *both* CIFAR-100 nets strongly and comparably; they are
no longer starved for regularization. The next increment on CIFAR-100 is unlikely to come from *more*
ensemble — it has to come from a lever the ensemble does not pull.

And every step so far pulled the *same* lever: the residual flow-and-depth axis. The gate rescaled the
branch, pre-activation reordered the path, stochastic depth dropped the branch — three ways of asking "how
much, how cleanly, how often does the branch contribute." None asks *which features* the branch should
emphasize for a given image. That axis is untouched, and it is orthogonal to depth — which is exactly the
property I want, because a move orthogonal to depth can lift the deep nets without paying the shallow tax
dropping does.

The gap is in the primitive every block is built from. A 3×3 conv produces each output channel by convolving
each input channel with its own kernel and summing across channels, and two facts fall out. The channel
mixing — how much output channel `c` listens to input channel `s` — is baked into the weights at training
time and is then *identical for every image*: the block cannot say "this is a dog, so fur-texture channels
matter and sky-blue channels do not." And the mixing is *local*: a 3×3 kernel integrates a tiny neighborhood,
so each block combines channels through a keyhole with no image-wide context. Static and locally-decided
channel responses — two limitations none of the flow-axis steps touched, because all three operate on the
branch as a whole, never its channels. The deep CIFAR-100 nets are precisely where per-image channel emphasis
should pay: many of the 100 fine-grained classes share low-level features and differ mainly in which
high-level channels are salient.

There are two ways to add capacity, and naming them separates the one I want. *Static* capacity — widen the
block, or switch `expansion` to a bottleneck 4 — computes a richer *fixed* function, which does nothing about
image-independence and is expensive: a 3×3 conv costs `9·C_in·C_out`, so doubling widths roughly quadruples
conv parameters, and on a 50k-image, 100-class set that much static capacity is as likely to overfit as help,
and it lifts shallow and deep together — the lockstep signature that would mean I missed the real bottleneck.
*Conditional* capacity keeps the parameter count almost fixed but lets the function *depend on the input*: a
per-image channel gate leaves the conv weights static and decides which output channels are amplified or
muted per image from global context. It is cheap, attacks image-independence head-on, and because it
modulates rather than enlarges the static machinery it does not carry the overfitting risk. So I want the
smallest module that turns a static, locally-decided channel mixing into a dynamic, globally-informed one — a
controller over the convolution, not more convolution.

Three moves make that concrete: *squeeze* the spatial dimension to a global per-channel summary, *excite* it
into one gate per channel, *scale* each branch channel by its gate. For the squeeze, global average pooling
gives a length-`C` descriptor whose entry `c` answers "how strongly and broadly does channel `c` respond to
this image" — every entry has a full-image receptive field, the global context the conv lacked. Global *max*
pooling keys on the single strongest location, high-variance and blind to the *extent* of a response, so I use
average, not max.

The excite is a bottleneck: `Linear(C → C/r)`, a nonlinearity, `Linear(C/r → C)`, then a gate nonlinearity.
The bottleneck matters: a full `C → C` map is `C²` weights that would let the gate *memorize* channel
co-activations rather than summarize them, an invitation to overfit on 50k images; the narrow neck forces it
to encode only the dominant modes of channel interaction, which is the generalization I want. The
nonlinearity *between* the linears matters too: without it the two collapse into a single rank-`C/r` linear
map that can only produce gates that are linear combinations of the channel means, never a conditional gate
like "raise channel `c` only when `s` and `t` are both active." A ReLU in the neck makes the excite genuinely
nonlinear at almost no cost.

The output nonlinearity is a **sigmoid, not a softmax**, and arithmetic settles it. Softmax normalizes the
gates to sum to 1, so over `C = 64` channels the average gate is `1/64 ≈ 0.016` and over the 16-channel
stage-1 blocks `1/16 ≈ 0.0625` — it would attenuate the whole branch to a couple of percent of its strength.
For a residual branch whose job is to *add* a correction, scaling it to ~2% nearly zeros the residual and
hands the net back to the shortcut. Channels are also not mutually exclusive — a dog image should light up fur
*and* ear *and* eye channels at once — so softmax's forced competition is the wrong prior. Sigmoid gives
independent gates in `(0,1)`, each set on its own, so the block can hold several channels near 1 and push
others toward 0 — exactly "amplify the discriminative, attenuate the irrelevant."

Placement ties back to the whole ladder: the recalibration acts on the *branch output, before the addition*,
never on the shortcut — the same principle every step kept, the identity highway open and ungated, only the
branch modulated. Gating the sum or the shortcut would reintroduce the Highway failure and undo the clean
additive path. So the SE gate sits between the second conv's BN and the addition: `H = ReLU(shortcut(x) +
SE(F(x)))`. And unlike stochastic depth, SE *adds* capacity without *removing* any — the full net trains
every step — so ResNet-20/CIFAR-10 should recover from the 91.37 dip rather than pay it, while the per-image
gate still gives the deep nets a lever orthogonal to the depth-ensemble they already have. Help the deep nets
without hurting the shallow one is the whole reason to reach orthogonal now.

A fresh SE block does not start as the plain baseline the way the gated `alpha = 1` did, so its init matters.
The FC weights are small (`1/√fan_in`), so each pre-sigmoid logit is dominated by the second linear's bias,
~`U(−0.5, 0.5)` (fan-in `mid = 4`), and `sigmoid` of that range centers on 0.5. So every channel gate starts
near 0.5 and the branch begins at half amplitude, injecting ~`0.25` of full-strength variance — a *mild*
soft-start beside the two I know (the gate's variance-1 exact baseline, pre-activation's `alpha² = 0.01`),
four times gentler than pre-activation's and per-channel learnable from step one. So the recovery on ResNet-20
will come not from the starting point but from adding real per-image capacity every step.

The SE geometry has to fit the block's narrow channel counts, and working the reduction ratio out reveals
something I did not expect. The stages are 16, 32, 64 wide, and a textbook `r = 16` would make stage 1's
bottleneck `16/16 = 1` channel — a degenerate one-dimensional squeeze with no conditional gating at all. So I
floor it: `mid = max(planes // 16, 4)`. But then `mid` is 4 at *every* stage — `max(1,4)`, `max(2,4)`,
`max(4,4)` — the floor fires on stages 1 and 2 and stage 3 lands on 4 exactly. So on these thin widths the
"reduction ratio 16" is nearly a misnomer: the real bottleneck is a constant 4-wide neck everywhere, and the
floor, not the ratio, shapes the recalibration. The *effective* ratio `planes/mid` is then `4, 8, 16` across
the stages — widest at the thinnest stage (few interactions to summarize, spend a wider code) and tightening
toward 16 only at the deepest — which is the right direction. And the 4-wide neck is a genuine expressiveness
constraint I read as a feature: at stage 3 the excite emits 64 gates from a 4-dimensional code, so they cannot
be set independently — only patterns factoring through a handful of shared modes of channel co-variation
("this image is high-texture, lift the texture group") rather than arbitrary per-channel values. On 50k images
that low-rank constraint is exactly the regularization that keeps a per-image gate from overfitting.

The cost is lightweight: each SE module is `2·planes·mid + mid + planes` parameters — 148 at 16 channels, 292
at 32, 580 at 64 — so ResNet-110 totals ~18.4k, about 1.1% of the net's ~1.7M weights, and the FLOPs are
negligible since SE runs on length-`C` vectors after pooling (a stage-3 `conv2` is ~2.4M multiply-adds
against the excite's ~512). The gate is per-image and per-channel, constant over space, and leaves the
shortcut path untouched — real added capacity of the right kind, with the identity highway open.

The block stays post-activation, full depth — no dropping, no reordering, no branch scalar — so this step
isolates the channel axis the way each prior one isolated its own. I considered *combining* SE with
stochastic depth, but dropping is what cost the shallow case 1.59 points and reaching orthogonal is precisely
to *stop* paying that tax; I hold the combine in reserve as the explicit fallback if the deep nets turn out to
need the ensemble *on top of* the gate. Spatial attention instead of channel would miss the diagnosed gap
(channel mixing is static, not spatial pooling wrong) and CIFAR maps are already tiny — `8×8` at stage 3. A
full `C → C` excite is 8× the parameters and exactly the memorization the bottleneck exists to prevent. The
full module is in the answer.

This is the endpoint, so the bar is setting-specific, against stochastic depth's 91.37 / 74.56 / 75.70. Most
diagnostic: ResNet-20/CIFAR-10 should *recover* well above 91.37, back toward the gated 92.96, because SE adds
capacity without the dropping that cost the shallow case and its init soft-start is gentle — if it does not,
the channel gate buys no real capacity and the orthogonal-axis premise is wrong. On the deep CIFAR-100 nets SE
must at least *hold* 74.56 and 75.70 and ideally edge them, since channel recalibration is orthogonal to the
depth-ensemble and a 100-class fit is where per-image emphasis should help most; if it *underperforms* there,
the CIFAR-100 limit really was depth-regularization SE does not provide, and the fallback is to combine SE
with dropping rather than substitute it. The endpoint stands on clearing the shallow case *and* holding the
deep cases at once — the net-positive-across-the-sweep no single flow-axis step managed.
