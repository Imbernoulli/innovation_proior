Pre-activation did almost exactly what I predicted on the deep end, and the *shape* of where it succeeded
hands me the next move. Against the gated block: ResNet-20/CIFAR-10 `92.96 → 92.62` (−0.34),
ResNet-56/CIFAR-100 `71.98 → 71.78` (−0.20), ResNet-110/CIFAR-100 `73.46 → 74.08` (+0.62). Those deltas are
*monotone in depth* — the cleanest confirmation that the reorder acted on a depth-scaling mechanism exactly
as the clamp-count argument said: the clamps hurt in proportion to how many blocks stack, so removing them
helps in that proportion, and only the deepest net had enough clamps for the fix to overcome its warm-up
cost. The very deep net finally pulled clear of the merely-deep one — the 110-minus-56 gap widened from 1.48
to `74.08 − 71.78 = 2.30`. So the wiring diagnosis is confirmed. But ResNet-56 went *down*, and that −0.20 is
the important signal: the same fixed 0.1-warm-up cost bought a repair of a mild clamp problem (9 blocks/stage,
`2^{−9}` damage) versus a 512× worse one at 110 (18 blocks/stage, `2^{−18}`), so the arithmetic came out
slightly negative at 56 and clearly positive at 110. The dip is not evidence that flow was 56's problem — it
is evidence flow was *not* 56's binding problem. And even at 74.08 on 54 blocks the 110-net is not a number
that says "this net is using its depth." Flow is no longer the limit; what remains is that the deep net does
not *use or regularize* its depth, and a reordering cannot give me that.

State the new tension precisely, because the contradiction is the whole idea. Depth helps — pre-activation
just proved 110 beats 56 by 2.30 once gradients flow. But depth also hurts on CIFAR-100: more blocks means
more parameters fitting a 50k-image, 100-class set, longer chains even with a clean highway, and more
opportunity for late blocks to overfit or sit idle passing their input along — the identity path makes "learn
nothing" a comfortable equilibrium. So I want two opposite things from one network: the *expressiveness* of a
deep net at test time and the *optimization and regularization* of a shorter net during training. But those
are needed in different *phases* — capacity at test, easy optimization during training — so the resolution is
a net that is effectively short while I train it and deep when I deploy it.

Several regularizers could deliver that; why block-dropping. The recipe freezes the optimizer, weight decay,
schedule, and augmentation — only the block is editable — so anything in the training loop or input pipeline
is off the table, ruling out simply turning up weight decay or augmentation. Plain feature dropout inside the
block regularizes *what* each block computes but neither shortens the training-time chain nor builds an
ensemble over *depth* — it does not touch the complaint that the extra blocks are idle. An auxiliary loss
would change the objective the frozen loop owns. Block-dropping is the one intervention inside the block that
also exploits the structure already here: because the shortcut carries the input across for free, deleting a
block's branch on a step turns that block into a pass-through at zero cost, which simultaneously shortens the
chain and — since each block is independently present or absent — turns one weight set into an ensemble of
sub-networks of different depths.

The mechanism: gate the branch with a per-mini-batch Bernoulli `b ∈ {0,1}`, `H = ReLU(b·F(x) + shortcut(x))`.
At `b = 1` it is the block I have; at `b = 0` the branch vanishes and the block is `ReLU(shortcut(x))`. Check
the load-bearing "as if it were not there" claim: on a within-stage block the shortcut is a bare identity, so
a dropped block returns `ReLU(x)`, and `x` is the previous block's post-ReLU output, non-negative
elementwise, so `ReLU(x) = x` *exactly* — a dropped within-stage block is a true identity, no forward or
backward compute for the branch at all. That cleanliness is a property of the *post-activation* block, and it
will drive the reversal decision below.

That delivers the two things the pre-activation numbers said were missing, and I want to keep them separate
because the metric — best test accuracy over a *fixed* 200-epoch budget — responds to them differently. The
number of surviving blocks `L̃ = Σ b_ℓ` shortens the forward and backward chains, an *optimization* effect
that lowers gradient variance and eases early training — but against a fixed schedule "trains faster" does not
by itself raise the best accuracy reached. The benefit that actually moves the fixed-budget number is
*regularization*: because each of `L` blocks is independently on or off, one weight set defines `2^L`
sub-networks of varying depth, each minibatch updates one, and at test the full net combines them as an
implicit ensemble over depth. Those ensembles are `2^9 = 512` at ResNet-20, `2^27 ≈ 1.3×10⁸` at ResNet-56,
`2^54 ≈ 1.8×10¹⁶` at ResNet-110 — astronomically richer where depth is largest, so this regularization gets
*stronger* with depth, exactly the direction I need. The ~25% per-step compute saving is a real by-product
but not something the fixed epoch count rewards, so I credit the ensemble, with shorter chains as support.

The survival schedule. Uniform is wrong: early blocks extract the low-level features every later block builds
on, so dropping one corrupts the foundation, whereas a late block's transformation is more specialized. So
survival should *decrease* with depth, and the gentlest such schedule is linear:
`p_ℓ = 1 − (ℓ/L)(1 − p_L)`, anchored so the first block nearly always survives and the last survives with
`p_L`. Training is known insensitive to `p_L`, so I fix `p_L = 0.5`. Then
`E(L̃) = Σ p_ℓ = L − (1−p_L)(L+1)/2 = (3L − 1)/4`: ResNet-110 trains ~40 blocks deep and deploys 54 (~25%
compute saved), ResNet-56 ~20, ResNet-20 ~6.5. The *fraction* trained, `(3L−1)/(4L)`, is ~3/4 at all three
depths — the same soft treatment everywhere — but the *absolute* number dropped per step, `(L+1)/4`, is 13.75
at 110, 7 at 56, 2.5 at 20, and it is the absolute count that drives the `2^L`-member ensemble diversity.
Flat compute saving, sharply depth-scaling ensemble richness — that split is precisely why "short while
training, deep at test" pays off most at the largest `L`. And the step-to-step spread
(`Var L̃ = Σ p_ℓ(1−p_ℓ) ≈ 54·0.17 ≈ 9`, sd ~3 at ResNet-110) means each step draws a *different* ~40-block
subset, which is what makes the members genuinely distinct rather than one fixed shorter net; at ResNet-20
that spread collapses to ~1, one more reason the regularizer bites hardest where `L` is large. The endpoints
confirm 0.5 is a real interior choice: at `p_L = 1` nothing drops and the block reduces exactly to the
baseline (the floor here); at `p_L = 0` the deepest block would never train.

The test rule needs care. At test I want the full net, every branch active — but block ℓ's branch was present
only a fraction `p_ℓ` of the time and everything downstream calibrated to that. Turning it on for every
example makes its contribution `1/p_ℓ` too large. So scale the branch by its survival probability,
`H_test = ReLU(p_ℓ·F(x) + shortcut(x))`, making the test contribution equal the training *expected*
contribution `E[b·F] = p_ℓ·F` exactly — an equality by inspection, not a heuristic. The scale must go
*outside* BN, after `bn2`: `bn2`'s running statistics accumulate only over surviving steps, so at test BN
normalizes `F` to the distribution it saw and only then is the whole normalized branch scaled by `p_ℓ`.
Folding `p_ℓ` inside, before BN, would make the running statistics inconsistent between train and test.

The substrate forces the counting to be derived, not assumed. A `CustomBlock` only sees
`(in_planes, planes, stride)` — no global block index — so the block self-counts: a class-level counter
incremented per construction, *reset* at the first block of stage 1 (detected by the signature
`in_planes == 16 and planes == 16 and stride == 1`, unique to a CIFAR ResNet's first block) so a fresh model
reindexes, with `L` read from the same counter at forward time (by then all blocks are built). Then
`p = 1 − (block_idx/L)(1 − p_L)`, the training forward draws one `torch.rand(1)` per block per step and keeps
the branch iff below `p`, the eval forward uses the `p`-scaled branch. That one draw is per minibatch, not per
example, so every example in a batch traverses the same sampled sub-network — coarser than per-example
dropping, but a single low-variance update of one well-defined member per step, with ensemble diversity
recovered *across* the ~78k steps as the sampler visits different members. One detail: when a *transition*
block is dropped the value is `ReLU(shortcut(x))` with `shortcut` the Conv-BN projection — not a literal
identity, unavoidable where the shapes change, but only 2 of ResNet-110's 54 blocks; the other 52 are the
clean free identity.

Now the reversal I want to be honest about. The instinct is to stack dropping on last step's pre-activation
block and keep its flow fix, but two things make that the wrong substrate. The clean free identity when a
block drops — `ReLU(x) = x` — is a post-activation property. And more decisively, stacking dropping on
pre-activation would put *two* multiplicative scales on the branch: the learned `alpha` (init 0.1) and the
survival `p`. At test the branch would be `p·alpha·F`, two factors calibrated under different assumptions,
and the 0.1 soft-start already reduces effective init depth in a way that *overlaps* with what dropping does
— I could not read whether a gain came from the ensemble or the soft start. To isolate the ensemble axis I
revert to the post-activation block (final ReLU after the add) and drop on it. This is not discarding the
last step's win for nothing: pre-activation fixed how gradients flow through a full-depth net (+0.62 at 110),
while stochastic depth makes the net train shallow and regularize like an ensemble — a property of the
dropping schedule, not the activation order. The two do not compose for free, and the trade is quantifiable:
I give up the +0.62 the reorder bought at 110, so to be worth it the ensemble must add more than 0.62 back
there.

I keep the schedule fixed rather than learnable: a uniform `p` would drop foundational early blocks as
readily as late ones, and a learnable per-block drop rate would buy noise (training is insensitive to `p_L`)
and reintroduce the learnable branch scaling I just reverted away from. So fixed linear decay, `p_L = 0.5`,
no new learnable parameters. The full module is in the answer.

The falsifiable shape sorts by depth in the *opposite* way from last time on the shallow end. Stochastic
depth's benefit grows with `L`, so I expect ResNet-110/CIFAR-100 to clear 74.08 by the largest margin — and
by more than the 0.62 I gave up — and ResNet-56 to recover past its 71.78 stall. ResNet-20 is the honest
cost: only 9 blocks to drop, a mere 512-member ensemble, and dropping branches on a net already near its
CIFAR-10 ceiling removes capacity for too much of training, so I expect ResNet-20/CIFAR-10 to *fall*,
possibly below both prior steps — acceptable, since the shallow easy case was never the binding objective.
If the benefit does *not* sort by depth — ResNet-110 fails to clear 74.08, or the shallow net gains as much
as the deep ones — the remaining gap was not depth-regularization and the ensemble story is wrong.
