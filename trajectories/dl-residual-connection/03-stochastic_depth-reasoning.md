The pre-activation reordering did almost exactly what I predicted on the deep end, and the *shape* of where
it succeeded and where it didn't is what hands me the next move. Line the three deltas up against the gated
rung: ResNet-20/CIFAR-10 went `92.96 → 92.62`, a slip of −0.34; ResNet-56/CIFAR-100 went `71.98 → 71.78`, a
dip of −0.20; ResNet-110/CIFAR-100 went `73.46 → 74.08`, a lift of +0.62. Those three deltas are *monotone in
depth* — −0.34, −0.20, +0.62 — which is the cleanest possible confirmation that the reorder acted on a
depth-scaling mechanism exactly as the clamp-count argument said it would: the clamps hurt in proportion to
how many blocks stack, so removing them helps in that same proportion, and only the deepest net had enough
clamps for the fix to overcome its costs. And the very deep net finally pulled clear of the merely-deep one:
the 110-minus-56 gap widened from 1.48 to `74.08 − 71.78 = 2.30`, the "pull decisively above" I was hoping
for. So the wiring diagnosis is confirmed. But read the other two numbers honestly. ResNet-20 slipped —
exactly the small shallow cost I flagged, the 0.1 residual warm-up buying nothing at depth 20 and losing a
fraction to the soft start. And ResNet-56 actually went *down*. That −0.20 dip is the important signal:
pre-activation fixed gradient *flow*, and yet on a 100-class problem the merely-deep net got no better and
the very deep net, at 74.08 on 54 blocks, is still not a number that says "this net is using its depth." I am
now in exactly the regime I warned about closing the last rung — flow is no longer the limit, and what
remains is that the deep net does not *use* or *regularize* its depth. A reordering cannot give me that. I
need a different kind of intervention.

The ResNet-56 dip is worth one more moment, because it is not noise — it is the clamp asymmetry from last
rung reading back exactly. The clamp damage the reorder repairs scales as `2^{−k}` per stage in the number of
blocks `k`, so at 9 blocks per stage (ResNet-56) the damage was about `2^{−9} ≈ 2×10⁻³` while at 18 per stage
(ResNet-110) it was `2^{−18} ≈ 4×10⁻⁶`, roughly 512× worse. But the *cost* of the pre-activation fix — the
0.1 residual warm-up, the few early epochs spent near the identity — is the *same* at both depths. So at
ResNet-56 the reorder paid a fixed warm-up cost to repair a mild clamp problem, and the arithmetic came out
slightly negative (−0.20); at ResNet-110 it paid the same cost to repair a 512× worse clamp problem, and came
out clearly positive (+0.62). The dip is therefore not evidence that flow was 56's problem — it is evidence
that flow was *not* 56's binding problem, that 56's clamps were too mild for a flow fix to be worth its
warm-up. Which is precisely why the merely-deep net, like the very-deep one, is waiting on a *different* lever
than flow, and reinforces that the remaining CIFAR-100 gap is about using and regularizing depth, not moving
signal through it.

Let me state the new tension precisely, because it is genuinely contradictory and the contradiction is the
whole idea. Depth helps — pre-activation just proved 110 beats 56 by 2.30 once the gradients flow. But depth
also hurts on CIFAR-100: more blocks means more parameters fitting a 50k-image, 100-class training set,
longer gradient and forward chains even with a clean highway, and more opportunity for late blocks to overfit
or to sit idle passing their input along — the identity path makes "learn nothing" a perfectly comfortable
equilibrium for a block. So I want two opposite things from the same network: the *expressiveness* of a deep
net at test time, and the *optimization and regularization behavior* of a shorter net during training. Stated
like that it sounds like I have to pick. But notice *when* each property is needed. I need the capacity at
test time, when the model has to represent 100 fine-grained classes; I need easy optimization and
regularization during training. Those are different phases — so the resolution is a network that is
effectively *short while I train it* and *deep when I deploy it*.

That reframing narrows the tool, but several regularizers could deliver it and I should say why block-dropping
specifically. The recipe freezes most of the usual knobs — optimizer, weight decay, schedule, and data
augmentation are all fixed, and only the block is editable — so anything living in the training loop or the
input pipeline is off the table, which already rules out simply turning up weight decay or augmentation to
fight the deep-net overfit. Inside the block I could add plain *feature dropout*, zeroing random activations of
the branch; but that regularizes by injecting noise into *what* each block computes, and it neither shortens
the training-time gradient chain nor builds an ensemble over *depth* — it does not touch the specific complaint
that the deep net's extra blocks are idle or redundant. I could add an auxiliary loss at intermediate depths to
force early blocks to earn their place, but that changes the training objective, which the frozen loop does not
let me edit cleanly. Block-dropping is the one intervention that is both inside the block *and* exploits the
structure already in front of me: because the shortcut carries the input across for free, deleting a block's
branch on a step turns that block into a pass-through at zero cost, which simultaneously shortens the chain and
— since each block is independently present or absent — turns one weight set into an ensemble of sub-networks
of different depths. It is the only candidate that converts "short while training, deep at test" from a wish
into a mechanism using nothing but the residual wiring I already have.

For a fixed architecture the depth is the depth, unless I can make some blocks *not count* on a given
training step. And the residual structure I have been building on hands me the tool directly: a block computes
`H = ReLU(F(x) + shortcut(x))`, and the shortcut path is already there, already carrying the input across. So
if, on a given step, I simply *delete* `F` for some block and keep only the shortcut, that block becomes a
pass-through and the network behaves, for that step, as if the block were not there. The mechanism is to gate
the branch with a per-mini-batch Bernoulli `b ∈ {0,1}`: `H = ReLU(b·F(x) + shortcut(x))`. When `b = 1` it is
exactly the block I already have; when `b = 0` the branch vanishes and the block is `ReLU(shortcut(x))`.

I should check the "as if it were not there" claim rather than assume it, because it is the load-bearing
property. On a within-stage block the shortcut is a bare identity, so a dropped block returns `ReLU(x)`. Is
that `x`? The input `x` to this block is the output of the *previous* block, and every block ends in a final
ReLU, so `x ≥ 0` elementwise — and `ReLU` of a non-negative tensor is that tensor unchanged. So `ReLU(x) = x`
*exactly*: a dropped within-stage block is a true identity, signal and gradient flow through it untouched, and
there is no forward or backward compute for the dropped branch at all. A dropped block is free and clean. This
is the property I have to protect, and it will drive the reversal decision below.

That delivers two things at once, and they are the two things the pre-activation numbers said were missing.
First, the *effective* training depth shrinks: the number of surviving blocks `L̃` is a sum of independent
Bernoullis, `E(L̃) = Σ p_ℓ`, so the gradient and forward chains are shorter during training even though the
deployed net is full-depth — the optimization-behavior-of-a-shorter-net I wanted. Second, and this is what
makes test error *fall* rather than just training go faster: because each of the `L` blocks is independently
on or off, one set of shared weights now defines `2^L` different sub-networks of *varying depth*, each
minibatch samples and updates one of them, and at test time the full net combines them as an implicit
ensemble over depth-diverse members. Put the numbers on that ensemble, because its size is the whole
regularization argument. ResNet-20 has 9 blocks, `2^9 = 512` sub-nets; ResNet-56 has 27, `2^27 ≈ 1.3×10⁸`;
ResNet-110 has 54, `2^54 ≈ 1.8×10¹⁶`. The ensemble the regularizer averages over is astronomically richer at
110 than at 20 — sixteen quadrillion members versus five hundred — so this is regularization that gets
*stronger* with depth, which is exactly the right direction given that the deep nets are where I am stuck.

It is worth separating those two benefits, because the metric — best test accuracy over a *fixed* 200-epoch
budget — responds to them differently and I do not want to credit the wrong one. The shorter-chain benefit is
an *optimization* effect: with `E(L̃) ≈ 40` for ResNet-110, the average backprop path is 40 blocks rather than
54, and the per-step gradient traverses fewer nonlinearities, which lowers gradient variance and eases the
early optimization of a very deep net. But that alone would only make training *faster*, and against a fixed
200-epoch schedule "faster" does not by itself raise the best accuracy reached — a full-depth net that trains
more slowly could still catch up within 200 epochs. The benefit that actually moves the fixed-budget number is
the *regularization* one: the `2^L`-member depth-ensemble changes what the converged net generalizes to, not
just how fast it gets there. So the honest mechanism for a higher best-test-accuracy is the ensemble, with the
shorter chains as a supporting effect that makes the deep net easier to optimize along the way. The ~25%
compute saving per step is a real by-product but not something the metric rewards directly, since the epoch
count is fixed; I keep it in mind as a bonus, not as the reason this should work.

Now the design decisions that pin down the actual fill, and the first is a deliberate reversal I want to be
honest about. The instinct is to stack this on top of last rung's pre-activation block, keeping its flow fix.
But two concrete problems make that the wrong substrate. The clean, free identity when a block is dropped —
`ReLU(x) = x` — is a property of the *post-activation* block, the most-studied substrate for block-dropping,
and it is what lets a dropped block cost nothing. More decisively, stacking dropping on the pre-activation
block would put *two* multiplicative scales on the same branch: the learned residual scale `alpha` (init 0.1)
and the survival scaling `p` I derive below for test time. At test the branch contribution would be
`p·alpha·F`, two factors calibrated under different assumptions tangled into one, and the 0.1 soft-start
already reduces the *effective* init depth in a way that *overlaps* with what dropping does — so I could not
read whether a gain came from the ensemble or from the soft start. To isolate the ensemble axis cleanly I
revert to the proven post-activation block (with the final ReLU after the add) and take the block-dropping on
it. This is not throwing away last rung's win for nothing: pre-activation fixed how gradients flow *through* a
full-depth net (worth +0.62 at 110), while stochastic depth makes the net *train shallow and regularize like
an ensemble*, a property of the dropping schedule and not the activation order. The two do not compose for
free, so I take the trade — and the trade is quantifiable and falsifiable: I am giving up the +0.62 the
reorder bought at 110, so for this rung to be worth it the ensemble has to add *more* than 0.62 back at 110.
The numbers will tell me which axis the deep CIFAR-100 nets cared about more.

Second decision: the survival schedule. Each block needs a survival probability `p_ℓ = Pr(b_ℓ = 1)`, and
uniform is wrong. Early blocks extract the low-level features that *every* later block builds on; drop an
early block and I corrupt the foundation the whole rest of the net depends on for that step, whereas a late
block's transformation is more specialized and less universally relied on. So survival should *decrease* with
depth, and the gentlest schedule with that property is a straight line anchored so the first block essentially
always survives and the last survives with probability `p_L`: `p_ℓ = 1 − (ℓ/L)(1 − p_L)`. One free knob,
`p_L`, and training is known to be insensitive to it, so I fix `p_L = 0.5` — the deepest block survives half
the time, the earliest nearly always. Concretely for ResNet-110's 54 blocks that gives `p_1 = 1 − (1/54)(0.5)
≈ 0.99`, `p_27 = 0.75`, `p_54 = 0.50`: a smooth ramp from near-certain survival at the bottom to a coin flip
at the top. Work out the effect on effective depth: `E(L̃) = Σ_{ℓ=1}^{L} [1 − (ℓ/L)(1 − p_L)] =
L − (1−p_L)·(L+1)/2`, and with `p_L = 0.5` this is `L − (L+1)/4 = (3L − 1)/4`. For `L = 54` that is
`161/4 = 40.25`; I train a net that is on average ~40 blocks deep and deploy all 54, saving `1 − 40.25/54 ≈
25%` of the forward/backward compute. The same formula gives `E(L̃) = 20` for ResNet-56 and `6.5` for
ResNet-20 — every net trains at about three-quarters of its deployed depth, but the *absolute* number of
dropped blocks, and hence the ensemble richness, is far larger where `L` is larger. Let me separate those two
quantities cleanly, because it is the crux of why this rung sorts by depth. The *fraction* of depth trained is
`E(L̃)/L = (3L − 1)/(4L)`, which is `40.25/54 ≈ 0.745` at 110, `20/27 ≈ 0.741` at 56, `6.5/9 ≈ 0.722` at 20 —
essentially flat, all three near `3/4`, tending to exactly `3/4` as `L → ∞`. So on a fractional basis every net
gets the *same* soft treatment; the per-step compute saving is ~25% everywhere, not a depth-dependent bonus.
What *does* scale with depth is the *absolute* number of blocks dropped per step, `L − E(L̃) = (L + 1)/4`, which
is `55/4 = 13.75` at 110, `28/4 = 7` at 56, `10/4 = 2.5` at 20 — the 110-net drops on average five to six times
as many blocks per step as the 20-net. And it is the absolute count, not the fraction, that drives the
regularizer: the sub-network family the ensemble averages over has `2^L` members, so a handful more droppable
blocks multiplies the diversity astronomically. That split — flat compute saving, sharply depth-scaling
ensemble richness — is precisely why the "short while training, deep at test" wish pays off most at the largest
`L`, and it is a quantitative claim, not a slogan. The same linear ramp at `L = 27` reads `p_1 ≈ 0.98`,
`p_14 ≈ 0.74`, `p_27 = 0.50`, a gentler version of the ResNet-110 schedule with fewer coins to flip.

The mean tells me how deep each step's sub-network is; its step-to-step *spread* is what makes this an ensemble
rather than one fixed shorter net, so let me size that too. With one Bernoulli per block per step,
`L̃ = Σ b_ℓ` has variance `Σ p_ℓ(1 − p_ℓ)`. Along the ramp `p` runs from ~0.99 at the first block to 0.5 at the
last, and `p(1 − p)` is near zero at the top (`0.99·0.01 ≈ 0.01`) and maximal at the bottom (`0.5·0.5 = 0.25`);
averaged over the ramp it is about 0.17, so for ResNet-110 the variance is roughly `54·0.17 ≈ 9.2` and the
standard deviation about 3. So each step draws a sub-network of depth `40 ± 3` blocks — not the same 40 every
step, but sometimes 37, sometimes 43, and with a *different* subset of the 54 blocks present each time. That
spread is the diversity the ensemble argument actually needs: if every step trained the identical 40-block
sub-net I would just have a shorter fixed network, but a depth fluctuating by ±3 around 40 over a different
subset each step is what makes the `2^54` members genuinely distinct and their test-time average a real
ensemble. The spread collapses at shallow depth — for ResNet-20's 9 blocks the same estimate gives variance
near `9·0.17 ≈ 1.5`, a standard deviation barely above 1 — so the shallow net not only has fewer members (512)
but samples them across a far narrower band of depths, one more reason the regularizer bites hardest exactly
where `L` is large.

The two endpoints of `p_L` confirm 0.5 is a genuine interior choice rather than an arbitrary one. At
`p_L = 1` nothing is ever dropped, every block survives with probability 1, and the block reduces *exactly* to
the vanilla post-activation baseline — so the floor of this rung, as `p_L → 1`, is the proven baseline, and I
am departing from it in a controlled direction. At `p_L = 0` the deepest block would survive with probability
0 — it would never train, its weights left at initialization — which is degenerate; the whole point is to
drop blocks *sometimes*, not to delete them. `p_L = 0.5` sits squarely between: the deepest block trains on
half its steps, enough to learn while still being absent often enough to force the rest of the net to cope
without it. That the two limits bracket a sensible interior, and that training is known to be insensitive
across a wide band of `p_L`, is why I fix it at 0.5 and spend no search budget on it.

Third, the test-time rule, where I have to be careful. At test I want the full net — every branch active, all
the capacity. But during training block ℓ's branch was present only a fraction `p_ℓ` of the time, and
everything downstream calibrated to that intermittent presence. Turn it on for *every* test example and its
contribution is, on average, `1/p_ℓ` larger than what the downstream weights expect. The fix is to match the
expectation: during training the branch's mean contribution is `E[b·F] = p_ℓ·F`, so at test I scale the branch
by its survival probability, `H_test = ReLU(p_ℓ·F(x) + shortcut(x))`, which makes the test-time contribution
equal the training-time *expected* contribution exactly. The identity passes through at full strength; only
the recalibrated branch is weighted. That equality, `p_ℓ·F = E[b·F]`, is the whole justification, and it is
an equality I can check by inspection rather than a heuristic.

I should also check the test rule composes with the BN inside the branch, because that is where a naive
scaling could silently break. The branch is `bn2(conv2(...))`, and `bn2`'s running mean and variance are
updated only on steps where the block survives — on dropped steps the branch is never computed, so those
statistics accumulate over exactly the distribution of `F` the surviving steps produced. At test, `bn2` uses
those running statistics on the always-computed branch, and the `p_ℓ` factor is applied *outside* BN, after
`bn2`. So BN normalizes `F` to the same distribution it saw in training, and only then is the whole normalized
branch scaled by `p_ℓ` to match its expected training contribution — the two operations do not interfere. Had
I folded `p_ℓ` inside, before BN, the running statistics would be inconsistent between train and test. Placing
the scale after the branch is thus not incidental; it is what keeps the recalibration exact.

Now the substrate-specific care, because the harness implements the counting in a way I have to derive rather
than assume. There is no global notion of "block index" handed to a `CustomBlock` — the constructor only sees
`(in_planes, planes, stride)` — so the block has to count itself. The fill uses a class-level counter: each
block, when constructed, increments a shared `CustomBlock` counter and records its own index, and the counter
is *reset* at the first block of stage 1, detected by the signature `in_planes == 16 and planes == 16 and
stride == 1`, which is unique to the first block of a CIFAR ResNet's first stage, so building a fresh model
starts the indexing over. The total `L` is read from the same class counter at forward time — by then every
block has been constructed, so `L` is the true total block count and `block_idx` runs `1..L`. Then
`p = 1 − (block_idx / L)(1 − p_L)` exactly as derived, the training forward draws a fresh `torch.rand(1)` per
block per step and keeps the branch iff it is below `p`, and the eval forward uses the `p`-scaled branch. That
single draw per block per step is worth reading precisely: it is *one* Bernoulli for the whole minibatch, so
every example in a batch traverses the *same* sampled sub-network on that step, rather than each example
getting its own drop pattern. That is coarser than per-example dropping and buys less within-batch diversity,
but it is the cheaper and cleaner choice on this substrate — each SGD step becomes a single, low-variance
update of one well-defined sub-network of the `2^L` family, and the ensemble diversity is recovered *across*
steps as the sampler visits different sub-networks over the ~78k steps of training. Given a fixed schedule,
sampling one member per step and cycling through the family is exactly enough to build the depth-ensemble I am
after. One further detail I will note rather than fight: when a *transition* block is dropped, the returned value is
`ReLU(shortcut(x))` with `shortcut` the Conv-BN projection, so the dropped transition is *not* a literal
identity — it is the projected, rectified input. That is unavoidable on the dimension-changing blocks (there
is no identity to fall back on when the shapes change), and there are only two such blocks per net, so 52 of
ResNet-110's 54 blocks are the clean free identity when dropped and only 2 are the projected fallback; the
clean-identity argument holds for the overwhelming majority.

Two alternatives I considered and rejected on the way here, to be sure the fill is the right one. A *uniform*
survival probability is simpler but wrong for the reason above — it would drop foundational early blocks as
readily as specialized late ones, corrupting the features the whole net depends on; the linear decay costs
nothing extra and puts the dropping where it is safe. A *learnable* per-block drop rate would be more
expressive, but training is known to be insensitive to `p_L`, so the extra knobs would buy noise, not signal,
and they would reintroduce learnable branch scaling of the kind I just reverted away from to keep the ensemble
axis clean. So: fixed linear decay, `p_L = 0.5`, no new learnable parameters — the regularization comes
entirely from the sampling.

So the edit relative to the pre-activation rung is: revert to the post-activation Conv-BN-ReLU block (final
ReLU after the add), give the class a self-counter and `_p_last = 0.5`, compute the linear-decay survival `p`
per block, drop the branch with probability `1 − p` per minibatch in training (returning the rectified
shortcut alone when dropped), and scale the branch by `p` at test. The full module is in the answer.

The falsifiable expectations against the pre-activation numbers sort by depth in the *opposite* way from last
rung on the shallow end, which is the cleanest possible test of *which* axis matters. Stochastic depth's
benefit — both the shorter training chains and the ensemble regularization — grows with `L`, so it should help
the very deep CIFAR-100 nets most and the shallow CIFAR-10 net least or not at all. Concretely: I expect
ResNet-110/CIFAR-100 to clear the pre-activation 74.08 by the largest margin, and by more than the 0.62 I gave
up (the `1.8×10¹⁶`-member ensemble is exactly the regularizer a 110-layer 100-class fit was missing), and
ResNet-56 to recover and pass its 71.78 stall, because dropping blocks regularizes the merely-deep net too.
ResNet-20 is the risk and the honest cost: at depth 20 there are only 9 blocks to drop, the ensemble is a mere
512 members, and dropping any branch on a net already near its CIFAR-10 capacity ceiling just *removes*
capacity for too much of training — so I expect ResNet-20/CIFAR-10 to *fall*, possibly below both prior rungs,
and that drop is acceptable because the shallow easy case was never the binding objective and the task rewards
a block that lifts the deep nets. The falsifiable failure mode is explicit and symmetric to last rung's: if
stochastic depth does *not* sort by depth — if it fails to lift ResNet-110 above 74.08, or if it helps the
shallow net as much as the deep ones — then the deep nets' remaining gap was not a depth-regularization
problem and the ensemble story is wrong. What I am betting is that the deep CIFAR-100 nets were
under-regularized, not under-flowing, and that trading the pre-activation flow fix for the depth-ensemble fix
is the right trade *because* the room left was at the deep end where the ensemble is largest.
