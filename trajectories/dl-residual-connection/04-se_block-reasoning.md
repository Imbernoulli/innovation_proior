Stochastic depth is the strongest rung so far, and the *way* it is strong tells me what it costs and where
the next move has to come from. On the deep CIFAR-100 nets it did exactly what the ensemble argument promised:
ResNet-110 went from the pre-activation 74.08 to 75.70 — a lift of +1.62, the biggest any rung has produced on
any setting — and ResNet-56 recovered from its 71.78 stall all the way to 74.56, a swing of +2.78 that not
only erased the pre-activation dip but cleared the gated 71.98 by more than two and a half points. The benefit
sorted by depth, just as the ensemble size predicted: largest where `L` is largest. But read the shallow
number and the trade is laid bare. ResNet-20/CIFAR-10 fell to 91.37 — the lowest CIFAR-10 result of the entire
ladder, `−1.25` below the pre-activation 92.62 and `−1.59` below the gated 92.96. That is the honest cost I
flagged: at depth 20 there are only 9 blocks to drop, the ensemble is a mere 512 members, and dropping
branches on a net already near its CIFAR-10 capacity ceiling just *removes* capacity for too much of training.

Lay the whole ladder out as a mental table, one column per setting, and the pattern that jumps out decides the
finale for me. No single block design has ever held the top of all three columns at once. The best
ResNet-20/CIFAR-10 number belongs to the gated block (92.96); the best ResNet-56 and ResNet-110 numbers both
belong to stochastic depth (74.56 and 75.70). Each rung wins exactly where its mechanism is strongest and
loses where that mechanism has to be paid for — the gate matched the baseline everywhere and so topped only the
easy case, pre-activation bought the deep flow at a shallow warm-up cost, stochastic depth bought the deep
ensemble at a shallow capacity cost. And on CIFAR-10 specifically the three numbers march monotonically *down*
the ladder — 92.96, 92.62, 91.37 — every structural move I have made has quietly eroded the shallow easy case
even as it lifted the deep ones. So what I have built is a sequence of one-sided trades, and the one thing I
have never once produced is a block that is net-positive across the whole depth sweep at the same time. That is
the gap the finale has to close, and I already suspect it will not close by pulling harder on the axis every
prior rung pulled, because pulling that axis is *what created the trade*.

One more read of the deep pair before I leave the flow axis, because the *gap between* the two CIFAR-100 nets is
a cleaner diagnostic than either number alone. Under the gated block the 110-net led the 56-net by
`73.46 − 71.98 = 1.48`; pre-activation widened that to `74.08 − 71.78 = 2.30` (the reorder helped the deeper net
more, its depth-scaling signature); stochastic depth *narrowed* it back to `75.70 − 74.56 = 1.14`. Naively the
ensemble should have *widened* the gap — 110 has an astronomically larger sub-network family than 56 — so a
narrowing looks wrong until I remember that 56 was partly *recovering* rather than purely gaining. Measure both
from the common gated baseline instead of from the pre-activation marks: ResNet-56 went `71.98 → 74.56 = +2.58`,
ResNet-110 went `73.46 → 75.70 = +2.24`, which is much closer, and the residual difference is just the
pre-activation dip 56 had to climb back out of. So the honest read is that the depth-ensemble lifted *both*
CIFAR-100 nets strongly and comparably, and both now sit far above where any flow-axis move left them. The deep
nets are no longer starved for regularization. That matters for the finale: it says the next increment on
CIFAR-100 is unlikely to come from *more* ensemble, because the ensemble I have is already doing its job — it
has to come from a lever the ensemble does not pull at all.

So I now have a ladder where every rung trades one setting against another, and — more tellingly — every rung so
far has pulled the *same* lever: the residual flow-and-depth axis. The gate rescaled the branch; pre-activation
reordered the path; stochastic depth dropped the branch. Three different ways of asking "how much, how cleanly,
how often does the residual branch contribute." None of them ever asks a different question: *which features*
the branch should emphasize for a given image. That axis is completely untouched, and — crucially — it is
orthogonal to depth, which is exactly the property I want. A move that is orthogonal to depth can lift the deep
nets without paying the shallow tax that dropping does, because it is not manipulating depth at all. That is the
first time on this ladder I can see a way off the one-sided trade.

Let me go back to the convolution itself, because the gap is in the primitive every block is built from. A 3×3
conv produces each output channel by convolving each input channel with its own small kernel and summing
across channels, and two facts about that fall out immediately. First, the channel mixing — how much output
channel `c` listens to input channel `s` — is baked into the kernel weights at training time and is then
*identical for every image*. The block has no way to say "this particular image is a picture of a dog, so the
fur-texture channels matter and the sky-blue channels do not"; the same fixed mixing runs on every input.
Second, the mixing is *local*: a 3×3 kernel integrates a tiny spatial neighborhood, so even though stacking
blocks grows the receptive field, each block decides how to combine channels while looking through a keyhole,
with no image-wide context. So the block's channel responses are static and locally decided — two limitations
that neither the gate, nor the reorder, nor the dropping ever touched, because all three operate on the branch
*as a whole*, never on its channels. The deep CIFAR-100 nets, fitting 100 fine-grained classes, are precisely
where input-dependent channel emphasis should pay: many classes share low-level features and differ mainly in
which high-level channels are salient, and a block that could *recalibrate* its channels per image — turn up
the discriminative ones, turn down the irrelevant ones, using global context — would add a kind of capacity
none of the flow-axis rungs can.

There are two ways to add capacity here, and naming them separates the one I want from the one that would
fail. The blunt way is *static* capacity: widen the block — more channels, or switch `expansion` from 1 to a
bottleneck 4 — so each layer computes a richer fixed function. But static capacity is the same function for
every image, which does nothing about the image-independence I just diagnosed, and it is expensive on exactly
this substrate. A 3×3 conv costs `9·C_in·C_out`, quadratic in width, so doubling the stage widths roughly
quadruples the conv parameters, and a bottleneck `expansion = 4` would also push the linear head from `64` to
`64·4 = 256` inputs; on a 50k-image, 100-class training set that much extra static capacity is as likely to
overfit as to help, and it lifts shallow and deep nets *together* — the all-settings-move-in-lockstep signature
that would mean I had not touched the real bottleneck. The other way is *conditional* capacity: keep the
parameter count almost fixed but let the function the block applies *depend on the input*. A per-image channel
gate is precisely that — the conv weights stay static, but which of their output channels are amplified or
muted is decided per image from global context. Conditional capacity is cheap (a gate, not a new conv), it
attacks the image-independence gap head-on, and because it modulates rather than enlarges the static machinery
it does not carry the overfitting risk of quadrupling the convs. So I want the smallest module that turns a
static, locally-decided channel mixing into a dynamic, globally-informed one — not more convolution, but a
controller over the convolution I already have.

So the design goal: give the block a lightweight, *input-dependent* gate over its channels, informed by
*global* (image-wide) context, applied to the branch output before it joins the identity stream. Three moves
make that concrete. First, *squeeze* the spatial dimension away to get a global per-channel summary. Second,
*excite*: turn that summary into one gate per channel. Third, *scale*: multiply each channel of the branch
output by its gate. Take the squeeze first and be deliberate about the pooling, because it is a real choice and
not a reflex. I want a length-`C` descriptor whose entry `c` answers "how strongly and how broadly does
channel `c` respond to *this* image." Global average pooling over `H×W` does exactly that — it averages the
whole feature map, so a channel that fires weakly but everywhere and a channel that fires strongly in one
corner are told apart, and every entry has a full-image receptive field, the global context the conv lacked.
The tempting alternative, global *max* pooling, keys on the single strongest spatial location, which is
high-variance (it moves with whichever pixel happens to peak), sensitive to a lone outlier activation, and
throws away the *extent* of the response — precisely the information "is this channel salient for the whole
image" needs. So average pooling, not max, and the descriptor is the mean over the `C×H×W` branch map, a clean
`C`-vector.

Now the excite stage, and two of its three ingredients are load-bearing decisions I can pin down rather than
copy. The excite is a tiny bottleneck: a linear `C → C/r`, a nonlinearity, a linear `C/r → C`, then a gate
nonlinearity. Why the bottleneck at all, when a full `C → C` map would be more expressive? Because a full map
is `C²` weights per layer and would let the gate *memorize* channel co-activations rather than *summarize*
them — on a 50k-image set that is an invitation to overfit, and the whole point of a gate is that it should
generalize the "which channels matter" decision across images. The reduction ratio forces the gate to pass the
descriptor through a narrow neck, so it can only encode the dominant modes of channel interaction, which is the
generalization-friendly constraint I want. Why the nonlinearity *between* the two linears? Because without it,
`Linear(C/r, C) ∘ Linear(C, C/r)` collapses into a single linear map `C → C` of rank at most `C/r` — a
rank-limited linear reprojection of the descriptor, which can only produce gates that are linear combinations
of the channel means and cannot express a conditional gate like "raise channel `c` only when channels `s` *and*
`t` are both active." A ReLU in the neck makes the excite a genuine nonlinear function of the descriptor at
almost no cost, and that conditional expressiveness is the difference between a fixed reweighting and an
input-dependent one. So ReLU in the hidden layer.

The output nonlinearity is the third ingredient and it is a **sigmoid, not a softmax**, and this one I can
settle with arithmetic rather than taste. A softmax normalizes the gates to sum to 1 across channels, so over
`C = 64` channels the *average* gate would be `1/64 ≈ 0.016`, and over the 16-channel stage-1 blocks it would
be `1/16 = 0.0625` — the gate would attenuate the entire branch to a couple of percent of its strength on
average. For a residual branch whose whole job is to *add* a correction to the identity stream, scaling it to
~2% of its magnitude is catastrophic; it would nearly zero the residual and hand the whole net back to the
shortcut. Channels are also not mutually exclusive — a dog image should be able to light up fur *and* ear *and*
eye channels at once — so a softmax's forced competition over a fixed budget is the wrong prior entirely. A
sigmoid gives independent per-channel gates in `(0,1)`, each set on its own with no cross-channel trade-off:
the block can hold several channels near 1 and push others toward 0 simultaneously, which is exactly the
"amplify the discriminative, attenuate the irrelevant" behavior I am after.

The placement decision ties this back to everything the ladder has learned. The recalibration must act on the
**branch output, before the identity addition** — never on the shortcut. This is the same principle that ran
through every rung: the identity highway stays open and ungated, so signal and gradient pass through it
untouched, and only the residual branch is modulated. Gating the sum, or the shortcut, would reintroduce
exactly the Highway failure (a gate that suppresses the through-line) and undo the clean additive path the
whole ladder depends on. So the SE gate sits at the end of the branch, between the second conv's BN and the
addition: `H = ReLU(shortcut(x) + SE(F(x)))`. And note what this buys at the shallow end where stochastic
depth hurt: SE *adds* capacity without *removing* any — there is no dropping, the full net trains every step,
so ResNet-20/CIFAR-10 should recover from the 91.37 dip rather than pay it, while the per-image channel gate
still gives the deep CIFAR-100 nets a lever orthogonal to the depth-ensemble they already benefit from. That
asymmetry — help the deep nets without hurting the shallow one — is the whole reason to reach for an orthogonal
axis now rather than a fourth variation on flow.

Before I trust that shallow-recovery claim I should work out what the gate does to the branch *at
initialization*, because a fresh SE block does not start as the plain baseline the way the gated `alpha = 1`
did, and if it started badly it could eat the recovery I am counting on. At init the two FC weights are small
(default init scales them by `1/√fan_in`), so each channel's pre-sigmoid logit is dominated by the second
linear's bias, drawn from roughly `U(−0.5, 0.5)` (its fan-in is `mid = 4`, so `1/√4 = 0.5`). Sigmoid of that
range is `(sigmoid(−0.5), sigmoid(0.5)) = (0.38, 0.62)`, centered on `sigmoid(0) = 0.5`. So every channel gate
starts at about 0.5, and the branch begins life scaled to *half amplitude* per channel — which, since the
branch after `bn2` carries roughly unit per-channel variance, means it injects about `0.5² = 0.25` of the
variance it would at full strength. Put that beside the two soft-starts I already know: the gated block started
its branch at variance 1 (the exact baseline), the pre-activation block at `alpha² = 0.01`. SE's 0.25 sits
between them — a *mild* soft-start, four times gentler than pre-activation's, and unlike either prior scalar it
is *per-channel* and *immediately learnable* in whatever direction each channel wants. So SE begins at roughly
quarter-strength residual and grows in, but far less aggressively than the 0.1 warm-up that cost ResNet-20 a
fraction last time, and the recovery I am counting on comes not from the starting point but from adding real
per-image capacity every step of training — capacity that more than offsets a half-strength start. If ResNet-20
fails to recover, this init scaling is the first suspect; but the arithmetic says it is a gentle start, not a
crippling one.

Now the substrate-specific care, because the block I have to fill is the task's basic block and the SE geometry
has to fit its narrow channel counts. The block stays post-activation here — full depth, no dropping, no
reordering, no branch scalar, so this rung isolates the channel axis cleanly the way each prior rung isolated
its own: `out = ReLU(BN(conv1(x)))`, `out = BN(conv2(out))`, then I insert the SE recalibration on `out`, then
add the shortcut, then the final ReLU. The reduction ratio needs care, and working it out reveals something I
did not expect. The CIFAR ResNet stages are 16, 32, 64 channels wide, and a textbook `r = 16` would make the
first stage's bottleneck `16/16 = 1` channel — a degenerate one-dimensional squeeze that throws away almost all
channel structure and leaves the excite unable to encode any conditional gating at all. So I floor the
bottleneck width: `mid = max(planes // 16, 4)`, guaranteeing at least 4 hidden units. But now compute `mid` at
each stage: stage 1 gives `max(16//16, 4) = max(1, 4) = 4`; stage 2 gives `max(32//16, 4) = max(2, 4) = 4`;
stage 3 gives `max(64//16, 4) = max(4, 4) = 4`. It is 4 at *every* stage — the floor fires on stages 1 and 2,
and stage 3 lands on 4 exactly. So on these thin CIFAR widths the "reduction ratio 16" is almost a misnomer:
the real bottleneck is a *constant 4 hidden units* everywhere, and `r = 16` is only nominally the ratio because
the floor overrides it in two of three stages and coincides with it in the third. That is worth knowing,
because it means every SE block in the net summarizes its channels through the same 4-wide neck, and the floor
— not the ratio — is the parameter actually shaping the recalibration. Without it the 16-channel stage would
get a 1-wide neck and no usable channel selectivity at all.

Two consequences of that floor change how I read the module's capacity, so I want them explicit. First, the
*effective* reduction ratio is not 16 anywhere but stage 3. With `mid = 4` fixed, the real ratio `planes/mid`
is `16/4 = 4` at stage 1, `32/4 = 8` at stage 2, and `64/4 = 16` at stage 3 — the neck is proportionally
*widest* at the thinnest stage and tightens toward the nominal 16 only at the deepest. That is the right
direction: a 16-channel stage has few channel interactions to summarize and can spend a relatively wide code,
while a 64-channel stage has many and benefits from the harder squeeze. Second, the bottleneck is a genuine
expressiveness constraint, and I want to read it as a feature rather than an accident. At stage 3 the excite
emits 64 channel gates, but from a 4-dimensional hidden code (and after the ReLU, from at most 4 active
coordinates). So the 64 gates cannot be set independently — they are pinned to the image of a 4-dimensional
code passed through one linear map and a sigmoid. The module cannot memorize an arbitrary per-channel gate
pattern; it can only express patterns that factor through a handful of shared modes of channel co-variation —
"this image is high-texture, lift the texture group" rather than "set channel 37 to 0.8." On a 50k-image,
100-class set that low-rank constraint is exactly the regularization that keeps a per-image gate from
overfitting: the capacity is dynamic but deliberately narrow. A full `C → C` excite would remove the constraint
and hand back 64 free gates — the very thing the bottleneck exists to forbid.

Count the cost, because "lightweight" has to be checked, not asserted. Each SE module is two linears with
biases: going down, `planes·mid + mid` parameters; coming back up, `mid·planes + planes`; total
`2·planes·mid + mid + planes`. With `mid = 4` a 16-channel block carries `2·16·4 + 4 + 16 = 148` parameters, a
32-channel block `2·32·4 + 4 + 32 = 292`, a 64-channel block `2·64·4 + 4 + 64 = 580`. For ResNet-110, 18 blocks
per stage, that totals `18·(148 + 292 + 580) = 18·1020 ≈ 18.4k` parameters — about 1.1% of the net's ~1.7M
weights. That is real added capacity, unlike the gated rung's 54 scalars, but still cheap, and it is capacity
of the *right kind*: a per-image gate, not more static convolution. The FLOP cost is smaller still, because SE
operates on length-`C` vectors after the pooling, not on `H×W` maps: a stage-3 block's `conv2` is on the order
of `64·64·9·8·8 ≈ 2.4M` multiply-adds, while its SE excite is `2·64·4 ≈ 512` — a ratio near `2×10⁻⁴`,
essentially free. So SE adds a per-image, global-context channel gate at ~1% of the parameters and a fraction
of a per-mille of the compute.

Let me verify the shapes end to end, because a broadcast error here would silently corrupt the recalibration
and I would never see it in the loss. Take the tightest stage first, a stage-1 block where `out` is
`(B, 16, 32, 32)`. `AdaptiveAvgPool2d(1)` → `(B, 16, 1, 1)`; `Flatten` → `(B, 16)`; `Linear(16, 4)` →
`(B, 4)`; `ReLU` → `(B, 4)`; `Linear(4, 16)` → `(B, 16)`; `Sigmoid` → `(B, 16)`. Then
`w = se(out).unsqueeze(-1).unsqueeze(-1)` → `(B, 16, 1, 1)`, and `out * w` broadcasts the per-channel gate
across all `32×32` spatial positions → `(B, 16, 32, 32)`, matching `out` and the shortcut for the addition. Now
the widest stage, a stage-3 block where `out` is `(B, 64, 8, 8)`: pool → `(B, 64, 1, 1)`, flatten → `(B, 64)`,
`Linear(64, 4)` → `(B, 4)`, ReLU, `Linear(4, 64)` → `(B, 64)`, sigmoid → `(B, 64)`, unsqueeze twice →
`(B, 64, 1, 1)`, and `out * w` broadcasts over the `8×8` map → `(B, 64, 8, 8)`. Both trace clean, and both
confirm the object I built is exactly what I set out to build: the gate is per-image (the `B` axis is preserved
untouched through the whole MLP), per-channel (length 16 or 64), and constant over space (the `1×1` broadcast).
Global-context, per-channel, per-image — confirmed by the shapes, not assumed.

The backward direction deserves the same quick check, because the whole ladder has lived on keeping the
identity path open and I should be sure the gate does not silently close it. Differentiate
`H = ReLU(shortcut(x) + out·w)` where `w = se(out)`. The gradient to the block input `x` splits, as always,
into a shortcut path and a branch path, and the shortcut path is `∂/∂x` of `shortcut(x)` unmodified by
anything the SE did — the gate never touches the highway, so a gradient at the top still reaches the bottom
along the identity term exactly as in the plain post-activation block. The branch path is where the gate lives,
and it is worth noticing it is *two* terms, not one: `d(out·w)/d(out) = w + out · (∂w/∂out)`, the direct
scaling by the gate plus the gate's own dependence on `out` (the descriptor is a function of `out`). So the SE
participates in its own branch gradient, but everything it does is confined to the branch — the through-line
Jacobian is untouched, the identity highway stays open, and the same safety property every rung protected holds
here too. The gate can mute its branch's contribution and its branch's gradient; it can never throttle the
shared through-line.

Three alternatives I weighed before settling here, because the finale should be the right axis and not just an
available one. I could *combine* SE with stochastic depth — keep the dropping and add the gate — but dropping
is exactly what cost the shallow case its 1.59 points, and my whole reason for reaching orthogonal is to *stop*
paying that tax; combining would carry the shallow cost forward, so I keep the full-depth block and hold the
combine in reserve as the explicit fallback if the deep nets turn out to need the ensemble *on top of* the
gate. I could reach for *spatial* attention instead of channel attention, but the gap I diagnosed is
specifically that channel mixing is static and image-independent, not that spatial pooling is wrong, and CIFAR
feature maps are already tiny — `8×8 = 64` positions at stage 3 — so a spatial gate has little global structure
to exploit and does not address the channel-selectivity gap at all. I could drop the bottleneck and use a full
`C → C` excite, but at stage 3 that is `64² = 4096` weights against the bottleneck's `2·64·4 = 512` — eight
times the parameters and exactly the memorization the reduction ratio exists to prevent. So: channel attention,
squeezed by global average pooling, excited through a ReLU-nonlinear bottleneck floored to a width of 4,
sigmoid-gated, on the branch output of the full-depth post-activation block. The full module is in the answer.

This is the endpoint, so let me state the bar it must clear and exactly what I would check, against the
strongest baseline's measured numbers. Stochastic depth set the marks at 91.37 (ResNet-20/CIFAR-10), 74.56
(ResNet-56/CIFAR-100), and 75.70 (ResNet-110/CIFAR-100). For SE to be a genuine improvement and not a sideways
trade, the falsifiable claims are setting-specific. First and most diagnostic: ResNet-20/CIFAR-10 should
*recover* well above 91.37 — back toward or past the gated 92.96 — because SE adds capacity without the
dropping that cost stochastic depth the shallow case, and because I just showed its init soft-start is far
gentler than the pre-activation warm-up; if SE does *not* fix the shallow number, then its channel gate is not
buying real capacity and the whole orthogonal-axis premise is wrong. Second, on the deep CIFAR-100 nets SE must
at least *hold* stochastic depth's 74.56 and 75.70 and ideally edge them, since channel recalibration is
orthogonal to the depth-ensemble and a 100-class problem is where per-image channel emphasis should help most;
if SE *underperforms* stochastic depth on the deep nets, then on CIFAR-100 the binding constraint really was
depth-regularization (which SE does not provide) rather than channel selectivity, and the right endpoint would
be to *combine* SE's gate with stochastic depth's dropping rather than substitute it — the fallback I kept in
reserve above. The decisive test of the endpoint, then, is whether one block design can simultaneously clear
91.37 on the shallow easy case *and* hold ~74.5 / ~75.7 on the deep CIFAR-100 cases — a net positive across the
whole depth sweep that no single flow-axis rung ever managed, because each of those, as the ladder table
showed, traded one end of the sweep against the other. That balance — recover the shallow capacity that
dropping removed while keeping the deep-net strength — is precisely what an orthogonal, capacity-adding,
depth-agnostic channel gate is built to deliver, and it is the claim the endpoint stands or falls on.
