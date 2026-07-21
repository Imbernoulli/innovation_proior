The Kaiming numbers landed almost exactly on the three predictions, and the one that matters is the ResNet
wash. On MobileNetV2 He jumped to 94.49 from orthogonal's 93.88 — `+0.61` — the depthwise convs got their
coherent per-filter init, confirming the spectrum was never earning points there. On VGG-16-BN He came in at
73.38 against 72.83 — `+0.55` — so the plain stack's off-diagonal conditioning, which I thought *might* hold a
slim edge, did not survive contact: the cheap second moment beat the pinned spectrum even on its home turf,
settling the earlier open question in favor of "the spectrum was redundant." But ResNet-56 came in at 72.07
against orthogonal's 72.08 — `0.01`, one part in seven thousand, a dead wash. Put the deltas side by side:
`+0.61`, `+0.55`, `+0.01`. The residual net moved by a fiftieth of what the other two moved when I swapped the
entire per-layer scheme. Two initializers that disagree *maximally* about the per-layer map — one pinning every
singular value, the other scattering them across Marchenko–Pastur — agree on the residual net to within noise.
That is a clean falsification: when the per-layer spectrum can be driven from one extreme to the other and the
residual net does not budge, per-layer scale is provably not what limits it. The ceiling is set by something
neither attempt touched, and the structure names it: residual accumulation.

Make the accumulation precise, because the fix has to come out of its structure. The main path is a running
sum `x_{l+1} = x_l + F_l(x_l)`, `F_l` for a `BasicBlock` two 3×3 convs ending `conv2 → bn2`. With `bn2` at
`γ=1` each branch is standardized to unit variance, so `F_l(x_l)` enters the sum at unit scale regardless of
the convs inside; roughly-independent unit contributions add in variance, so `Var(x_{l+1}) ≈ Var(x_l) + 1` and
after `L` blocks `Var(x_L) ≈ Var(x_0) + L`. ResNet-56's 27 `BasicBlock`s inflate the signal into the head by
~`28` in variance, `√28 ≈ 5.3×` in std (milder in truth — the three stage-boundary projection shortcuts
partially reset the sum — but it grows with depth regardless). Three costs follow at init: the logits are
over-scaled so the softmax starts saturated and the early cross-entropy gradient is distorted; each branch
enters the additive path *on equal footing with the whole accumulated identity path* rather than as a small
correction, so a 27-deep stack of full-volume branches is not the "identity plus small refinements" that makes
deep residual learning easy; and the deepest branches, their unit contribution buried in a variance-28 sum, get
a gradient that is a small fraction of the whole and train slowest. No per-layer second moment can pay that
tax, because the problem is *between* layers, in how branches sum. Which is exactly why the two per-layer
schemes tied.

The remedy the structure suggests is to make each branch *start small* so the sum doesn't inflate, and start it
from the *identity* rather than a random function so SGD doesn't first have to unlearn a bad branch. I can
derive the depth exponent rather than guess it: to make each branch's contribution `Θ(1/L)` in variance — so
the sum of `L` stays `Θ(1)` — each branch's output std must be `Θ(L^{-1/2})`. The residual-scaling analysis of
Zhang, Dauphin & Ma (2019) distributes that across a branch's `m` weight layers by zero-initializing the *last*
and scaling the other `m-1` each by `L^{-1/(2m-2)}`, so they compose to `L^{-1/2}`. For the `BasicBlock`,
`m = 2`, exponent `-1/(2·2-2) = -1/2`: scale `n_blocks^{-1/2}`, `27^{-1/2} ≈ 0.192`, each scaled branch conv
pulled to a fifth of its He magnitude.

But the substrate I edit is *not* the normalization-free setting that analysis was built for, and importing it
wholesale would break the graph or duplicate BN. The original recipe trains res-nets *without any
normalization*: it removes BatchNorm, replaces it with learnable scalar biases before every conv and
activation, adds one learnable scalar multiplier per branch, scales the non-zero branch weights by
`L^{-1/(2m-2)}` while zero-initializing the last conv, and zero-initializes the classifier — every piece a
substitute for what BN would do. These networks *keep* BatchNorm (frozen graph, forbidden to touch), and I
cannot add scalar biases or multipliers without altering the graph. So I do not transplant it: no learnable
scalars, no removing BN, and I do not zero the classifier either — with BN present and the head's logits
already controlled by the `fan_in` scaling from step 2, zeroing the head would throw away a good start and force
SGD to rebuild it. The right translation into a BN-equipped network is two edits, both strictly inside the
contract.

First, the depth-corrected down-scale: run the full step-2 He pass (convs `fan_out`, Linear `fan_in`, BN
neutral, zero biases — the second moment that just proved itself on VGG and Mobile), then for ResNets multiply
each block's *last* conv (`conv2`) weight by `n_blocks^{-1/2}`. Second, exploit BN itself to start the branch
from identity for free: the branch ends `conv2 → bn2`, and `bn2`'s affine `γ` multiplies the entire
standardized branch output before it is added to the shortcut, so `bn2.weight = 0` at init makes the branch
output exactly zero and the block compute `relu(shortcut(x))` — the residual branch starts as the zero function
(the zero-γ trick, Goyal et al. 2017).

The interaction between these two edits is more subtle than "scale down and start near zero." Trace the forward
pass at init. With `bn2.weight = 0` (and `bn2.bias = 0`),
the branch output is identically zero regardless of `conv2`'s weights — so the accumulation is killed at init
*entirely by the zero-γ*: every branch contributes zero, the sum stays `Var(x_0)`, `√28` collapses to `√1`. The
`conv2` down-scale does nothing to the initial forward pass. Where does it act? BN is scale-invariant in the
weight preceding it: for any `s > 0`, `BN(s·z) = BN(z)` since batch mean and std both scale by `s` and cancel.
So down-scaling `conv2` leaves the branch's forward output unchanged at *every* point in training, not just init
— the branch output is `γ · BN(conv2 output)` and BN has erased `conv2`'s scale. The down-scale is
forward-inert; its effect is on the *gradient*, because BN's backward pass carries a `1/σ` factor and shrinking
`conv2`'s output `σ` by `n_blocks^{-1/2}` rescales the gradient `conv2` receives — a depth-aware conditioning of
how fast each branch grows once `γ` lifts off zero. So the division of labor: the zero-γ delivers the identity
start and kills accumulation at init; the `conv2` down-scale is the residual-scaling rule's depth correction
acting through the gradient, shaping branch growth so the reassembled sum stays `Θ(1)` across depth as branches
wake.

That a branch starting at exactly zero can ever wake up is worth checking, or the scheme would permanently
disable the branches. The branch output is `y = γ · BN(conv2(...))` with `γ = 0`, so
`∂y/∂conv2 = γ · ∂BN/∂conv2 = 0`, and by the chain rule everything upstream in the branch (`bn1`, `conv1`) is
gated by that same zero. The *one* parameter with a nonzero gradient is `γ`: `∂y/∂γ = BN(conv2(...))`, the
normalized branch output, which is not zero because `conv2` still holds its down-scaled He weights. So on step
one only `γ` moves — by the correlation between that normalized signal and the upstream loss gradient,
generically nonzero — and the instant `γ ≠ 0`, `conv2` then `conv1` begin to receive gradient. The branch
emerges from the identity in the right order: `γ` opens the gate, the convs — held at their He scale, a
decently-conditioned random map BN standardizes — follow immediately behind. This is why I zero *γ* rather than
`conv2` itself: zeroing `γ` starts the branch's *output* at zero while leaving `conv2` a good He map, so the
branch is inert but *loaded* — the moment the gate opens there is a usable random function behind it. Zeroing
`conv2` would start the branch's *map* degenerate, forcing it to grow a whole conv from nothing.

Starting every branch inert buys a second thing that specifically helps the deep blocks the accumulation start
punished: with all 27 branches zero, forward and backward signals flow along the pure identity path (plus the
three projection shortcuts), so at step one every block — including the deepest — sees an unattenuated gradient
down the identity, instead of a deep branch's `1/28` sliver buried in a variance-28 sum. The net begins
effectively *shallow*, a stack of identities, and deepens as SGD grows the branches it needs from a
clean-gradient start — an easy shallow function earning its depth, the regime a 56-layer net could not reach
when both per-layer attempts started it at full accumulated depth. Weight decay (`5e-4`) sits consistent: a zero
`γ` has zero decay force, so branches grow only as far as the data pulls them. And BN's scale-invariance is also
why I am free to scale `conv2` rather than the recipe's `conv1` — forward-immaterial which conv carries the
`L^{-1/2}`, so scaling the layer that sits directly against the additive sum pairs cleanly with zeroing `bn2` on
the same tail.

For the non-residual architectures the answer is "do nothing extra," and the reasoning matters. VGG-16-BN has
no shortcuts and no accumulation — a plain chain, no branch to scale, no last-branch-BN to zero — so the He
pass *is* the right init and it already won at 73.38; leave it. MobileNetV2 does have additive shortcuts in its
`InvertedResidual` blocks, and I considered extending the fix, but three things argue against it: its branch is
three convs (expand-1×1 → depthwise-3×3 → project-1×1), so `m = 3` and the exponent would be `-1/4`, a
different scaling the `BasicBlock` accounting does not transfer; the shortcut exists only when stride is 1 and
channels match, so many blocks have no additive path and no accumulation to control; and He's 94.49 is already
best there with no sign the accumulation binds. Rather than guess an exponent for a block type the derivation
wasn't built for on a network where accumulation may not even be the constraint, I gate the residual scaling on
`arch.startswith('resnet')` and let Mobile and VGG keep plain He.

So the step-3 edit: phase one, the full step-2 He pass; phase two, if `arch` is a ResNet, count `n_blocks`
`BasicBlock`s, multiply each `conv2.weight` by `n_blocks^{-1/2}`, set each `bn2.weight` to zero. No graph
change, no added parameters, no data. The decisive test is ResNet-56: it sat at the 72.07/72.08 wash precisely
because neither per-layer attempt addressed accumulation, and this is the only one that does, so if my reading is
right it should finally move up off that ceiling — while VGG and Mobile, where the residual phase never fires,
should simply hold their best per-layer results. If ResNet does *not* move, accumulation was not the binding
constraint and the next move must look past initialization entirely. The full module is in the answer.
