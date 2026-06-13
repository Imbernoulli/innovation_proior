The pre-activation reordering did almost exactly what I predicted on the deep end, and the *shape* of where
it succeeded and where it didn't is what hands me the next move. ResNet-110/CIFAR-100 went from the gated
73.46 to 74.08 — the largest absolute lift of the three settings, and the very deep net finally pulled
clearly above the merely-deep one, which is the depth-dependent payoff the clean additive highway was
supposed to buy. That confirms the wiring diagnosis: at 110 layers the binding constraint really was the
obstructed identity path, and unclamping it helped most where there were most clamps. But look at the other
two numbers and the story is not clean. ResNet-20/CIFAR-10 slipped to 92.62 from the gated 92.96 — exactly
the small shallow cost I flagged, the 0.1 residual warm-up buying nothing at depth 20 and losing a fraction
to the soft start. And ResNet-56/CIFAR-100 actually went *down*, 71.98 → 71.78. That ResNet-56 dip is the
important signal. Pre-activation fixed gradient *flow*, and yet the deep CIFAR-100 nets are still leaving
points on the table — 74.08 at 110 layers on a 100-class problem is not a number that says "this net is
using its depth." So I am now in exactly the regime I warned about closing the last rung: flow is no longer
the limit, and what's left is that the deep net does not *use* or *regularize* its depth. A reordering
cannot give me that. I need a different kind of intervention.

Let me state the new tension precisely, because it is genuinely contradictory and the contradiction is the
whole idea. Depth helps — pre-activation just proved 110 beats 56 once the gradients flow. But depth also
hurts on CIFAR-100: more blocks means more parameters fitting a 50k-image, 100-class training set, longer
gradient and forward chains even with a clean highway, and more opportunity for the late blocks to overfit
or to sit idle passing their input along (the identity path makes "learn nothing" a perfectly comfortable
equilibrium for a block). So I want two opposite things from the same network: the *expressiveness* of a
deep net at test time, and the *optimization and regularization behavior* of a shorter net during training.
Stated like that it sounds like I have to pick. But notice *when* each property is needed. I need the
capacity at test time, when the model has to represent 100 fine-grained classes. I need easy optimization
and regularization during training. Those are different phases. What if the network could be effectively
*short while I train it* and *deep when I deploy it*?

For a fixed architecture the depth is the depth — unless I can make some blocks *not count* on a given
training step. And here the residual structure I have been building on hands me the tool directly. A block
computes `H = ReLU(F(x) + shortcut(x))`; the shortcut path is already there, already carrying the input
across. So if, on a given step, I simply *delete* `F` for some block and keep only the shortcut, that block
becomes a pass-through and the network behaves, for that step, as if the block were not there. The depth I
actually backpropagate through shrinks. The mechanism is to gate the branch with a per-mini-batch Bernoulli
`b ∈ {0,1}`: `H = ReLU(b · F(x) + shortcut(x))`. When `b = 1` it is exactly the block I already have; when
`b = 0` the branch vanishes and the block is `ReLU(shortcut(x))`. On the within-stage blocks the shortcut is
a bare identity and the input is non-negative (it is the output of a prior ReLU), so `ReLU(x) = x` and a
dropped block is an *exact* identity — the signal and gradient flow through it untouched, as if it were
removed, and there is no forward or backward compute for the dropped branch at all. A dropped block is free
and clean.

That immediately delivers two things at once, and they are the two things the pre-activation numbers said
were missing. First, the *effective* training depth shrinks, so the gradient and forward chains are shorter
during training even though the deployed net is full-depth — the optimization-behavior-of-a-shorter-net I
wanted. Second, and this is the part that makes test error *fall* rather than just training go faster:
because each of the `L` blocks is independently on or off, one set of shared weights now defines `2^L`
different sub-networks of *varying depth*, each minibatch samples and updates one of them, and at test time
combining them is an implicit ensemble over depth-diverse members. That is regularization of exactly the
kind a deep CIFAR-100 net needs — and it is regularization that gets *stronger* with depth, since `L` is
larger, which is the right direction given that the deep nets are where I am stuck.

Now the design decisions that pin down the actual fill, and the first one is a deliberate reversal I want
to be honest about. The natural instinct is to stack this on top of last rung's pre-activation block. But
the dropping argument depends on a *clean exact identity* when the branch is off, and it is cleanest on the
plain post-activation block: there `b = 0` gives `ReLU(shortcut(x))`, and on the bare-identity shortcuts
that is exactly `x`. So for this rung I drop the pre-activation reordering and the 0.1 residual scale and go
back to the proven post-activation block as the thing-being-gated. Why is that not throwing away the last
rung's win? Because stochastic depth attacks a *different* axis: pre-activation fixed how gradients flow
*through* a full-depth net, and that mattered most at 110; stochastic depth makes the net *train shallow and
regularize like an ensemble*, which is a property of the dropping schedule, not of the activation order.
The two do not compose for free — the residual scale would interact with the survival scaling at test, and
the cleanest, most-studied version of block-dropping is on the post-activation block — so I take the
block-dropping on the vanilla block and accept that this rung trades the pre-activation flow fix for the
ensemble fix. The numbers will tell me which axis the deep CIFAR-100 nets cared about more.

Second decision: the survival schedule. Each block needs a survival probability `p_ℓ = Pr(b_ℓ = 1)`, and
uniform is wrong. Early blocks extract low-level features that *every* later block builds on; drop an early
block and I corrupt the foundation the whole rest of the net depends on for that step. A late block's
transformation is more specialized and less universally relied on. So survival should *decrease* with depth.
The gentlest schedule with that property is a straight line, anchored so the first block essentially always
survives and the last survives with probability `p_L`: `p_ℓ = 1 − (ℓ/L)(1 − p_L)`. One free knob, `p_L`,
and training is known to be insensitive to it, so I fix `p_L = 0.5` — the deepest block survives half the
time, the earliest nearly always. Work out the effect on effective depth: the number of surviving blocks
`L̃` is a sum of independent Bernoullis, so `E(L̃) = Σ p_ℓ`, and under linear decay with `p_L = 0.5` this is
`Σ [1 − ℓ/(2L)] = L − (L+1)/4 = (3L − 1)/4 ≈ 3L/4`. For ResNet-110, which is 54 residual blocks, that is
about 40 — I train a net that is on average ~40 blocks deep and deploy all 54, with roughly a quarter of the
forward/backward compute saved. That is the "short while training, deep at test" wish made concrete, and it
is strongest precisely at the largest `L`, i.e. ResNet-110.

Third, the test-time rule, where I have to be careful. At test I want the full net — every branch active,
all the capacity. But during training, block ℓ's branch was present only a fraction `p_ℓ` of the time, and
everything downstream calibrated to that intermittent presence. Turn it on for *every* test example and its
contribution to the sum is, on average, `1/p_ℓ` larger than what the downstream weights expect. This is the
Dropout situation and the fix is the same: scale the branch by its survival probability at test,
`H_test = ReLU(p_ℓ · F(x) + shortcut(x))`, so the expected contribution matches training. The identity
passes through at full strength; only the recalibrated branch is weighted.

Now the substrate-specific care, because the task's harness implements the counting in a way I have to
derive exactly rather than assume the textbook form. There is no global notion of "block index" handed to a
`CustomBlock` — the constructor only sees `(in_planes, planes, stride)`. So the block has to count itself.
The fill uses a class-level counter: each block, when constructed, increments a shared `CustomBlock`
counter and records its own index; the counter is *reset* at the first block of stage 1 (detected by the
signature `in_planes == 16 and planes == 16 and stride == 1`, which is unique to the first block of a CIFAR
ResNet's first stage) so that building a fresh model starts the indexing over. The total `L` is read from
the same class counter at forward time — by then every block has been constructed, so `L` is the true total
block count and `block_idx` runs `1..L`. Then `p = 1 − (block_idx / L)(1 − p_L)` exactly as derived, the
training forward draws a fresh `torch.rand(1)` per block per step and keeps the branch iff it is below `p`,
and the eval forward uses the `p`-scaled branch. One detail of the harness I will note rather than fight:
when a transition block is dropped, the returned value is `ReLU(shortcut(x))` where `shortcut` is the
Conv-BN projection, so the dropped transition block is *not* a literal identity — it is the projected,
rectified input. That is unavoidable on the dimension-changing blocks (there is no identity to fall back to
when the shapes change) and there are only two such blocks per net, so the clean-identity argument holds for
the overwhelming majority of blocks; the two transitions just pass their projected input when dropped, which
is the closest thing to a pass-through the shape change allows. The full scaffold module is in the answer.

So the edit relative to the pre-activation rung is: revert to the post-activation Conv-BN-ReLU block (with
the final ReLU after the add), give the class a self-counter and `_p_last = 0.5`, compute the linear-decay
survival `p` per block, drop the branch with probability `1 − p` per minibatch in training (returning the
rectified shortcut alone when dropped), and scale the branch by `p` at test. No new parameters, no learned
gate — the regularization comes entirely from the sampling.

The falsifiable expectations against the pre-activation numbers, and they sort by depth in the opposite way
from last rung, which is the cleanest possible test of *which* axis matters. Stochastic depth's benefit —
both the shorter training chains and the ensemble regularization — *grows with `L`*, so it should help the
very deep CIFAR-100 nets most and the shallow CIFAR-10 net least or not at all. Concretely: I expect
ResNet-110/CIFAR-100 to clear the pre-activation 74.08 by the largest margin (the ensemble over `2^54`
depth-varying sub-nets is exactly the regularizer a 110-layer 100-class fit was missing), and ResNet-56 to
recover and pass its 71.78 stall, because dropping blocks regularizes the merely-deep net too. ResNet-20 is
the risk and the honest cost: at depth 20 there are few blocks to drop, the ensemble is small, and dropping
any block on a net that is already near its capacity ceiling for CIFAR-10 may simply *remove* useful
capacity for too much of training — so I expect ResNet-20/CIFAR-10 to *fall*, possibly below both prior
rungs, and that drop is acceptable because the shallow easy case was never the binding objective and the
task rewards a block that lifts the deep nets. The falsifiable failure mode is explicit and symmetric to
last rung's: if stochastic depth does *not* sort by depth — if it fails to lift ResNet-110 above 74.08, or
if it helps the shallow net as much as the deep ones — then the deep nets' remaining gap was not a
depth-regularization problem and the ensemble story is wrong. What I am betting is that the deep CIFAR-100
nets were under-regularized, not under-flowing, and that trading the pre-activation flow fix for the
depth-ensemble fix is the right trade *because* the room left was at the deep end where the ensemble is
largest.
