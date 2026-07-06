The Kaiming numbers landed almost exactly on the three predictions, and the one that matters most is the
ResNet wash, so I want to read the table with a pencil before I decide anything. On MobileNetV2 He jumped to
94.49 from orthogonal's 93.88 — a swing of `+0.61` — the depthwise convs got their coherent per-filter init
and the improvement was real, confirming the spectrum was never the thing earning points there. On VGG-16-BN
He came in at 73.38 against orthogonal's 72.83 — `+0.55` — so the plain stack's off-diagonal spectral
conditioning, which I thought *might* hold a slim edge, did not survive contact; the cheap correct second
moment beat the expensive pinned spectrum even on its home turf, which settles the open question from last
rung in favor of "the spectrum was redundant." But ResNet-56 came in at 72.07 against orthogonal's 72.08 — a
difference of `0.01`, one part in about seven thousand, a dead wash. Put the three deltas side by side:
`+0.61`, `+0.55`, `+0.01`. The residual net moved by roughly a *fiftieth* of what the other two moved when I
swapped the entire per-layer scheme. Two initializers that disagree *maximally* about the per-layer map — one
pinning every singular value to a single value, the other scattering them across Marchenko–Pastur — agree on
the residual net to within noise. That is not a small fact dressed up; it is a clean falsification. When the
per-layer spectrum can be changed from one extreme to the other and the residual net does not budge, per-layer
scale is provably *not* what limits the residual net. The ResNet ceiling is set by something neither rung
touched, and the structure of the network tells me exactly what it is: residual accumulation.

Make the accumulation problem precise, because the fix has to come out of its structure and not out of an
analogy. A ResNet main path is a running sum `x_{l+1} = x_l + F_l(x_l)`, where `F_l` is a residual branch —
for `BasicBlock`, two 3×3 convs each with BN, ending `conv2 → bn2`. I can pin the branch's output scale using
the BN I have been keeping at `(γ=1, β=0)`: with `γ=1`, `bn2` standardizes its channel outputs to unit
variance, so each `F_l(x_l)` enters the sum at roughly unit scale regardless of what the two convs inside did.
The branch outputs at init are roughly independent, and independent unit-variance contributions *add* in
variance, so `Var(x_{l+1}) ≈ Var(x_l) + 1`, and after `L` blocks `Var(x_L) ≈ Var(x_0) + L`. ResNet-56 on
CIFAR has 27 `BasicBlock`s, so the signal entering the head is inflated by roughly a factor of `28` in
variance — `√28 ≈ 5.3×` in standard deviation. (The true figure is milder because the three stages insert a
projection shortcut at each boundary that partially resets the running sum, but the mechanism is what matters,
and it grows with depth no matter the exact count.) Three consequences follow at the start of training, and
each is a real cost. The logits are over-scaled by that factor, so the softmax starts saturated and the early
cross-entropy gradient is distorted. BN does re-standardize inside each branch — that is why the network
trains at all — but the *additive main path itself* carries the growing sum, and each branch enters that sum
at unit scale, i.e. on *equal footing with the entire accumulated identity path* rather than as a small
correction; a 27-block-deep residual function whose branches all shout at full volume is not the "identity
plus small refinements" that makes deep residual learning easy. And the deepest branches, whose unit
contribution is buried in a sum of variance 28, get a gradient signal that is a small fraction of the whole,
so they train slowest. That is the accumulation tax, and no per-layer second-moment choice can pay it, because
the problem is *between* layers — in how the branches sum — not *within* any one. Which is exactly why the two
per-layer schemes tied.

The remedy the residual structure itself suggests is to make each branch *start small* so the running sum does
not inflate, and to start it from the *identity* rather than from a random function so SGD does not have to
first unlearn a bad random branch before it can use it. I want the "start small" to have the right depth
dependence, and I can derive the exponent instead of guessing it. If I want each branch's contribution to the
network function to be `Θ(1/L)` in variance rather than `Θ(1)` — so that the sum of `L` branches is `Θ(1)`
again, depth-independent — then each branch's output magnitude (standard deviation) must be `Θ(L^{-1/2})`.
Now, the residual-scaling analysis of Zhang, Dauphin & Ma (2019) distributes that target across a branch's `m`
weight layers by zero-initializing the *last* layer (so the branch starts at exactly zero) and scaling the
other `m-1` layers each by `L^{-1/(2m-2)}`; then the `m-1` scaled layers compose multiplicatively to
`(L^{-1/(2m-2)})^{m-1} = L^{-(m-1)/(2m-2)} = L^{-1/2}`, precisely the per-branch magnitude I wanted. For the
`BasicBlock` here, `m = 2` (two convs per branch), so the exponent is `-1/(2·2-2) = -1/2`: the scale is
`L^{-1/2}`, i.e. `n_blocks^{-1/2}`. For `n_blocks = 27` that is `27^{-1/2} ≈ 0.192` — each scaled branch conv
pulled down to about a fifth of its He magnitude. That exponent is the depth-corrected factor that controls
the accumulation directly, which is exactly the lever the ResNet wash exposed as untouched.

But here I have to be careful, because the substrate I am editing is *not* the normalization-free setting that
analysis was built for, and importing the wrong machinery wholesale would either break the graph or duplicate
what BN already does. The original residual-scaling recipe was designed to train res-nets *without any
normalization at all*: it removes BatchNorm entirely, replaces it with learnable scalar biases inserted before
every conv and activation, adds one learnable scalar multiplier per branch, scales the *non-zero* branch
weights by `L^{-1/(2m-2)}` while zero-initializing the *last* conv of each branch, and zero-initializes the
classifier. Every one of those pieces exists to *substitute* for what BN would have done — the scalar biases
stand in for BN's shift, the multiplier for BN's scale, the zeroed last conv for a controlled start. But this
task's networks **keep BatchNorm** — it is part of the frozen model graph I am forbidden to touch — and I
cannot add scalar biases or multipliers without altering the graph, which the contract prohibits. So I must
*not* transplant that story. I cannot add learnable scalars (graph change), I cannot remove BN (graph change),
and I should not zero the classifier: with BN present and the head's logits already controlled by the `fan_in`
scaling I kept from step 2, zeroing the head would throw away a good start and force SGD to rebuild it from
nothing. What I *can* do, and what is the right translation of the residual-scaling idea into a BN-equipped
network, is two edits, both strictly inside the contract, and it is worth tracing exactly what each one does
because the BN changes their roles from what they play in the normalization-free recipe.

The first edit is the depth-corrected down-scale: run the full step-2 He pass first — every conv `fan_out` He,
every Linear `fan_in` He, every BN neutral, zero biases, so I keep the second moment that just proved itself
on VGG and Mobile — and then, for ResNets only, multiply each block's *last* conv (`conv2`) weight by
`n_blocks^{-1/2}`. The second edit exploits BatchNorm itself to start the branch from identity, for free, with
no graph change: in a `BasicBlock` the branch ends `conv2 → bn2`, and `bn2`'s affine `weight` (γ) multiplies
the *entire* standardized branch output before it is added to the shortcut, so setting `bn2.weight = 0` at
initialization makes the branch output exactly zero and the block compute exactly `relu(shortcut(x))` — the
residual branch starts as the *zero function* and the block starts as a clean identity-plus-shortcut. This is
the zero-γ trick (Goyal et al. 2017): zero the last BN's scale in each residual block so branches begin inert
and SGD grows them from zero.

Now I have to be honest about the interaction between these two edits, because BN makes it more subtle than
"scale the branch down and start it near zero," and getting it wrong would be exactly the kind of empty
assertion I want to avoid. Trace the forward pass at init. With `bn2.weight = 0` (and `bn2.bias = 0` from the
neutral pass), the branch output is `0 · normalized(conv2 output) + 0 = 0`, *identically*, regardless of what
`conv2`'s weights are. So the accumulation is killed at init *entirely by the zero-γ*: every branch
contributes zero, the running sum stays `Var(x_0)`, and `√28` collapses to `√1`. The `conv2` down-scale does
*nothing* to the initial forward pass — the branch is already zero. Where does the down-scale then act? Notice
that `bn2` standardizes `conv2`'s output, and BN is scale-invariant in the weight that precedes it: for any
positive scalar `s`, `BN(s·z) = (s·z − s·μ)/(s·σ) = (z − μ)/σ = BN(z)` — the batch mean and standard deviation
both scale by `s` and cancel. So scaling `conv2` down by `n_blocks^{-1/2}` leaves the branch's forward output
unchanged *at every point in training*, not just at init: whatever `γ` has become, the branch output is
`γ · BN(conv2 output)` and `BN` has erased `conv2`'s scale. The down-scale is *forward-inert* under BN. Its
effect is on the *gradient*: BN's backward pass carries a `1/σ` factor, and shrinking `conv2`'s output `σ` by
`n_blocks^{-1/2}` rescales the gradient that `conv2`'s weights receive, which is a depth-aware conditioning of
how fast each branch *grows* once `γ` lifts off zero. So the honest division of labor is: the zero-γ delivers
the identity start and the accumulation kill at init; the `conv2` down-scale is the residual-scaling rule's
depth correction acting through the gradient, shaping the branches' growth so that as they wake up the
reassembled sum stays `Θ(1)` across depth rather than re-inflating. It costs nothing, it matches the `m=2`
accounting, and I keep it as the depth-aware companion to the zero-γ start.

It is worth tracing how a branch that starts at exactly zero ever *wakes up*, because if it could not, the
whole scheme would be a clever way to permanently disable the residual branches. Look at what receives
gradient on the first step. The branch output is `y = γ · BN(conv2(...))` with `γ = 0`, so
`∂y/∂conv2 = γ · ∂BN/∂conv2 = 0` — and by the chain rule everything upstream of `conv2` in the branch
(`bn1`, `conv1`) is gated by that same zero, so it too gets zero gradient at init. The *one* parameter in the
branch with a nonzero gradient is `γ` itself, because `∂y/∂γ = BN(conv2(...))`, the normalized branch output,
which is not zero — `conv2` still holds its (down-scaled) He weights, so `BN(conv2(...))` is a well-defined
random-but-unit-variance signal. So on step one only `γ` moves, and it moves by the correlation between that
normalized branch signal and the upstream loss gradient — generically nonzero, so `γ` lifts off zero. The
instant `γ ≠ 0`, on the next step `conv2` and then `conv1` begin to receive gradient and refine the map. The
branch therefore emerges from the identity in the right order: `γ` opens the gate first, and the convs — held
ready at their He scale, already a decently-conditioned random map that BN standardizes — follow immediately
behind. This is precisely why I zero the *γ* rather than zeroing `conv2` itself, and it reconciles the "small
but live correction" intent with the exact-zero init: zeroing `γ` starts the branch's *output* at zero while
leaving `conv2`'s weights at a good He map, so the branch is *inert but loaded* — the moment the gate opens
there is a usable random function behind it. Zeroing `conv2` instead would start the branch's *map* degenerate,
and the branch would have to grow a whole conv from nothing before it could contribute; the zero-γ route gets
the identity start without paying that price.

And starting every branch inert buys a second thing beyond killing the accumulation, which is why I expect it
to specifically help the *deep* blocks that the accumulation start punished. With all 27 branches zero, the
forward and backward signals both flow along the pure identity path (plus the projection shortcuts at the
three stage boundaries), so at step one *every* block — including the deepest — sees an unattenuated gradient
routed straight down the identity, instead of the deep branches' contribution being a `1/28` sliver buried in
a variance-28 sum. The network at init is effectively *shallow* — a stack of identities — and it deepens as
SGD grows the branches it needs, block by block, from a clean-gradient starting point. That is the mechanism I
am really buying: not just a smaller logit scale, but a residual net that begins as an easy shallow function
and earns its depth, which is exactly the regime a 56-layer net could not reach when both per-layer rungs
started it at full accumulated depth. Weight decay (`5e-4`) sits quietly consistent with this: it never pushes
`γ` below zero from zero (a zero parameter has zero decay force), so the branches grow only as far as the data
pulls them and no further, which is the behavior I want from a correction term.

This BN scale-invariance also explains, incidentally, why I am free to deviate from the canonical recipe's
choice of *which* conv to scale. The normalization-free recipe scales the first conv and zero-inits the last;
I scale the last conv (`conv2`) and zero the last BN's γ. Because BN erases `conv2`'s forward scale, it is
forward-immaterial which conv carries the `L^{-1/2}` factor — so scaling `conv2`, the layer whose output sits
directly against the additive sum and the most natural place to think about "the branch's contribution," is as
valid as scaling `conv1` would be, and it pairs cleanly with zeroing `bn2` on the same tail of the branch. So
the task's residual init is a hybrid the normalization-free recipe never was: keep BN, keep the He second
moment, then for ResNets (a) down-scale the last branch conv by `n_blocks^{-1/2}` to carry the depth-corrected
`Θ(1/L)` growth, and (b) zero the last branch BN's γ to start each branch from identity and kill the
accumulation at init. I am explicitly *not* adding scalar biases or multipliers, *not* removing BN, *not*
zeroing the classifier — those belong to the no-normalization world and would violate the frozen-graph
contract here.

For the non-residual architectures the answer is "do nothing extra," and it is worth saying why rather than
treating it as a default. **VGG-16-BN** has no shortcuts and no additive accumulation — it is a plain chain,
so there is no branch to scale and no last-branch-BN to zero; the He pass *is* the right init, and it already
won there at 73.38, so I leave it untouched. **MobileNetV2** does have additive shortcuts in its
`InvertedResidual` blocks, and I considered extending the fix to them, but three things argue against it and
they are worth spelling out. Its block is a three-conv branch (expand-1×1 → depthwise-3×3 → project-1×1), so
`m = 3` and the exponent would be `-1/(2·3-2) = -1/4`, a *different* scaling than the `BasicBlock`'s `-1/2` —
the `n_blocks`/`BasicBlock` accounting does not transfer. The shortcut exists *only* when stride is 1 and the
channel count matches, so many `InvertedResidual` blocks have no additive path at all and therefore no
accumulation to control — the depth-`L` sum argument does not even apply uniformly. And the leaderboard's
strongest MobileNetV2 result already lives at He's 94.49, with no evidence the accumulation binds there. So
rather than guess an exponent for a block type the derivation was not built for, on a network where the
accumulation may not even be the constraint, I gate the residual scaling on `arch.startswith('resnet')` and let
Mobile and VGG keep the plain He init that is already best for them. The honest scope of the residual fix is
the architecture whose accumulation I can actually count: ResNet.

So the step-3 edit, literally: phase one, the full He pass from step 2 (convs `fan_out`, Linear `fan_in`, BN
neutral, zero biases); phase two, *if* `arch` is a ResNet, count `n_blocks` = number of `BasicBlock`s,
multiply each block's `conv2.weight` by `n_blocks^{-1/2}`, and set each block's `bn2.weight` to zero. No graph
change, no added parameters, no data, no calibration. The distilled module and its exact code live in the
answer.

The falsifiable expectations against the Kaiming rung, where this either earns the top of the ladder or does
not. The decisive test is **ResNet-56**: it sat in a dead wash at 72.07/72.08 under the two per-layer schemes
precisely because neither addressed accumulation, and this rung is the *only* one that does — the zero-γ start
turns `√28` of main-path inflation into `√1` and lets SGD grow the branches from identity — so if my reading
of the wash is right, ResNet-56 should finally move *up* off that 72.07 ceiling. If it does not, then
accumulation was not the binding constraint after all, and my whole reading of the tie is wrong: BN would have
to be neutralizing the accumulation well enough on its own that starting the branches at zero buys nothing,
and the next move would have to look past initialization entirely. On **VGG-16-BN** I expect essentially no
change from He's 73.38, because the residual phase never fires there — any movement is plain seed-level noise,
and a clean tie-or-better is the success condition; a *drop* would mean my phase-one pass somehow diverged
from step 2, which it must not, since phase one *is* step 2 byte for byte. On **MobileNetV2** the residual
phase is gated off too, so I expect to hold He's 94.49 — again, holding is the success condition, not
improving. The shape of the win I am predicting is therefore narrow and specific: the top of this ladder is
won *on the residual net*, by paying the accumulation tax that the per-layer rungs left on the table, while
matching the best per-layer result everywhere the residual machinery correctly stays out of the way.
