The gated block landed exactly where its own low bandwidth said it would, and the numbers point past it
cleanly. On ResNet-20/CIFAR-10 it hit 92.96 — a strong, well-tuned vanilla ResNet-20 number, which is the
tell that the learnable scalars settled near 1 and the block behaved as the proven baseline; there was
little structural slack at depth 20 on a 10-class problem and the gate found none to exploit, exactly the
"match the baseline, best on the easy case" outcome I predicted. But the deeper CIFAR-100 settings are
where the gate ran out of road: 71.98 on ResNet-56 and 73.46 on ResNet-110. Read those two together. The
*very deep* net (110) only edges the merely-deep one (56) by ~1.5 points, and on an absolute scale 73.46
at 110 layers is unimpressive — a 110-layer net should be pulling clear of a 56-layer one, not crawling
past it. That is the signature of a depth problem the scalar gate could not touch: a single per-block
multiplier rescales how *loud* each block is, but it cannot change how cleanly signal and gradient pass
*through* the stack, and at 110 layers "through the stack" is the binding constraint. So the diagnosis is
sharp and it is structural: the limit is not the residual *magnitude* (which the gate already let SGD tune)
but the residual *wiring* — what sits on the identity path and in what order the block applies its
normalization and nonlinearity. That is the lever I have not yet pulled.

So let me look hard at the one piece of the block I deliberately held fixed on the last rung: the ordering.
The default — and the gated block, which kept the default ordering — is *post-activation*:
`out = ReLU(BN(conv1(x)))`, `out = BN(conv2(out))`, then the addition, then a **final ReLU on the sum**,
`H = ReLU(shortcut(x) + alpha·F(x))`. Two things on that identity path bother me when I imagine stacking it
110 deep. First, that last ReLU sits *after* the addition, which means the value carried from one block's
output to the next block's input is `ReLU(...)` — every block's output is forced non-negative before the
next block (and the next shortcut) sees it. Second, follow the shortcut across many blocks. Within a stage
the shortcut is a bare identity, so the "highway" from the input of block ℓ to the output of block ℓ is
`x ↦ ReLU(x + F(x))`. Chain that across the whole stage and the clean additive identity I wanted —
`x_{out} = x_{in} + Σ F` — is interrupted at *every* block by a ReLU. The identity path is not actually
clean; it is a sequence of additions each clamped through a rectifier. At depth 20 that barely matters; at
depth 110 those repeated clamps are precisely the kind of thing that throttles how a perturbation at the
top propagates back to the bottom, and how the input's information survives forward to the top — the
"signal through the stack" problem the gated numbers exposed.

What I want, then, is a block where the identity path is *genuinely* unobstructed: the output of block ℓ
should be addable to block ℓ+1's branch without a nonlinearity sitting on the through-line. State it as a
goal: make the shortcut carry its signal from the input of the first block in a stage to the output of the
last with no BN and no ReLU in the way, so that across the whole stage the map is a pure sum
`x_{out} = x_{in} + Σ_ℓ F_ℓ(x_ℓ)`. If I can arrange that, a gradient at the top reaches the bottom along a
path of all 1's — no rectifier zeroing half of it at each block, no BN rescaling it — which is exactly the
property a 110-layer net needs and a 56-layer net needs less and a 20-layer net hardly needs at all,
matching the shape of where the gated block left headroom.

The obstruction is the activation order. The post-activation block computes the nonlinearity *after* the
addition because the convs come first and BN/ReLU follow them; the only way to get the ReLU off the
through-line is to move it *inside* the branch, ahead of the convolutions. So reorder the block to
**pre-activation**: each conv is preceded by its own BN and ReLU rather than followed by them. The branch
becomes `F(x) = conv2( ReLU(BN( conv1( ReLU(BN(x)) ) )) )` — BN-ReLU-Conv, twice — and the block output is
just `shortcut(x) + F(x)`, with **no activation after the addition**. Now trace the through-line again. The
branch consumes a *pre-activated* copy of `x` (it applies BN then ReLU to `x` at its entrance), does its
work, and the result is added to the shortcut; nothing rectifies the sum. The next block, in turn, applies
*its own* entrance BN-ReLU to whatever it receives. So the signal that travels block-to-block along the
shortcut is the raw additive stream, never clamped; each block's nonlinearity acts only on the *copy* it
pulls into its branch, not on the highway. Across a stage the identity path is now the clean
`x_{out} = x_{in} + Σ F_ℓ` I was after — the rectifiers have all moved off the through-line and into the
branches where they belong.

I want to double-check this is a real improvement and not just a cosmetic shuffle, because reordering BN
and ReLU around a conv feels like it shouldn't matter. It matters for two concrete reasons. Backward: with
no post-addition ReLU, the derivative of `x_{out}` with respect to the stage input is exactly
`I + Σ ∂F_ℓ/∂x`, an identity plus corrections — the gradient at the top is *guaranteed* to reach the bottom
undiminished along the `I` term, no matter how small the `∂F_ℓ` factors get. In the post-activation block
that `I` is gated by a ReLU's 0/1 mask at every block, so the clean additive path is repeatedly
multiplied by an indicator that can zero it. Forward: putting BN at the *entrance* of each branch means the
input to every convolution in the network is normalized, which is a strictly better-conditioned input than
a conv that consumes a raw, possibly large additive sum. So pre-activation both unclogs the backward
highway and normalizes every conv's input — and both effects scale with depth, which is exactly why I
expect this to help ResNet-110 most, ResNet-56 next, and ResNet-20 barely.

Now the substrate-specific care, because this is where the task's block differs from the textbook
pre-activation idea and I have to derive the *actual* fill, not the generic one. Two deviations the harness
bakes in, and both are deliberate.

First, the shortcut on the dimension-changing blocks. In a fully "pure" identity-mapping block the
shortcut would also be a clean, parameter-free path. But the stride-2, channel-doubling transitions
(the first block of stages 2 and 3) cannot be a bare identity — the shapes don't match — and this scaffold
resolves that with a *projection*: a 1×1 stride-2 conv followed by BN, applied to the **raw input `x`**, not
to the pre-activated input. So the through-line is only the clean unobstructed identity on the *within-stage*
blocks; at the two transition blocks per net the shortcut is a small Conv-BN projection of `x`. That is a
conscious choice and the right one here: putting BN on that projection keeps the two transition points
well-scaled, and there are only two of them per net, so the great majority of blocks still enjoy the clean
additive highway. I keep the branch's own entrance pre-activation (BN-ReLU) computed once and shared — the
branch reads the pre-activated `x`, the projection reads the raw `x` — which is the natural way to wire it
when the shortcut needs its own normalization anyway.

Second, and this is the deviation I would *not* have reached for on my own but which the substrate's depth
demands: a fixed-small **residual scale**. Even with the clean pre-activation highway, a freshly
initialized 110-layer net adds 54 branch outputs onto the identity stream, and at init each `F_ℓ` is a
random small-but-nonzero perturbation; 54 of them summed onto the identity can push the stream's variance
up enough to make the very first epochs jittery before the convs settle. The fix that fits the
"don't touch the recipe" constraint is to multiply each branch by a small constant at the start so the net
*begins* close to the identity and the branches grow into their contribution — but, having learned the
lesson from the gated rung, I do **not** init this scale at zero (that would waste the early budget on an
effectively-shallow net against a fixed 200-epoch schedule) and I do not even need it to be learnable in
the same loud way. The scaffold's choice is a learnable scalar `alpha` initialized to **0.1**: each branch
starts at one-tenth strength, so a deep net begins as roughly the identity plus a gentle residual, the
through-line is dominated by the clean path through the early epochs, and SGD is free to grow `alpha` toward
where each block wants it. This is a different role than the gated rung's scalar — there the scalar started
at 1 to *match* the baseline and re-weight around it; here it starts at 0.1 to *stabilize* the much deeper
pre-activation stack at init while still being depth-one-block-capacity from the first step. The two scalars
look alike and mean opposite things: the gate at 1 says "behave like the baseline, adjust"; the
pre-activation scale at 0.1 says "start near identity, grow." I keep it because at 110 layers the
unscaled clean highway is helped, not hurt, by a soft start. The full scaffold module is in the answer.

So the edit relative to the gated block is genuinely structural, not a rescaling: move both BNs and ReLUs
to *precede* their convs (BN-ReLU-Conv ordering), delete the post-addition ReLU so the through-line is a
pure sum, route the dimension-change shortcut through a Conv-BN projection of the raw input, and scale the
branch by a 0.1-initialized learnable `alpha`. The block's two convs, the kernel sizes, the
dimension-matching logic, and the parameter budget are otherwise the same; what changed is the *order*, and
the order is the thing the gated numbers said was limiting the deep nets.

Now the falsifiable expectations against the gated rung's measured numbers, because the whole point of
reordering is a depth-dependent payoff and the three settings are a depth sweep. On ResNet-20/CIFAR-10 the
clean highway buys little — 20 layers barely feel the repeated post-activation clamps — so I expect to land
*around* the gated 92.96, possibly a hair under, since at shallow depth the 0.1 residual init costs a few
epochs of warm-up the gated block didn't pay. That is fine; the shallow case was never where the room was.
The real test is the deep CIFAR-100 nets, where the gate stalled at 71.98 (56) and 73.46 (110). If the
diagnosis is right — that the limit there was the obstructed identity path, not the residual magnitude —
then pre-activation should lift the *very deep* net most: I expect ResNet-110 to clear the gated 73.46 by
the largest margin, ideally pulling decisively above the ResNet-56 number, because a clean additive highway
is exactly what 110 layers were missing. ResNet-56 should improve too but by less, since 56 layers suffered
the clamps less. The falsifiable failure mode is explicit: if the gate's problem had really been residual
magnitude rather than wiring, pre-activation would *not* differentiate the deep nets from the shallow one,
and all three would move together — if I see that, the reordering story is wrong. And one risk I am
flagging now for the next rung: pre-activation fixes how signal flows *through* the depth, but it does
nothing to make the deep net *use* its depth as an ensemble or to regularize the 100-class fit — so if the
deep CIFAR-100 numbers improve on gradient flow but still leave the very deep net under-generalizing, the
next lever is not flow but *implicit regularization across depth*, which a per-block scalar reordering can
never provide.
