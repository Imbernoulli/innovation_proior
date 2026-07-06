The gated block landed exactly where its own low bandwidth said it would, and the numbers point past it
cleanly. On ResNet-20/CIFAR-10 it hit 92.96 — a strong, well-tuned vanilla ResNet-20 number, which is the
tell that the learnable scalars settled near 1 and the block behaved as the proven baseline; there was
little structural slack at depth 20 on a 10-class problem and the gate found none to exploit, exactly the
"match the baseline, best on the easy case" outcome I predicted. But the deeper CIFAR-100 settings are where
the gate ran out of road: 71.98 on ResNet-56 and 73.46 on ResNet-110. Read those two together, because
their *relationship* is the diagnostic, not either number alone. The very deep net (110 layers, 54 residual
blocks) beats the merely-deep one (56 layers, 27 blocks) by only `73.46 − 71.98 = 1.48` points. I doubled
the residual depth — twice as many blocks, twice the representational budget — and bought 1.48 points of
CIFAR-100 accuracy. That is a terrible exchange rate. A 110-layer net that is genuinely using its extra 27
blocks should be pulling *clear* of a 56-layer one, not crawling 1.48 past it; the flat gap says the marginal
blocks are contributing almost nothing. And the gate could not have fixed this: a single per-block multiplier
rescales how *loud* each block is, but it cannot change how cleanly signal and gradient pass *through* the
stack, and at 110 layers "through the stack" is the binding constraint. So the diagnosis is sharp and it is
structural — the limit is not the residual *magnitude* (which the gate already let SGD tune, to no avail on
the deep nets) but the residual *wiring*: what sits on the identity path and in what order the block applies
its normalization and nonlinearity. That is the one lever I deliberately held fixed on the last rung.

Before I reach for wiring I should rule out the cheaper explanation, because if the deep stall were a
*capacity* problem the fix would be more parameters, not a reorder. But the gated ResNet-110 already carries
roughly twice the residual blocks and parameters of the gated ResNet-56, and it still only edges it by 1.48
points — so on this substrate more depth-worth-of-parameters is demonstrably *not* converting into accuracy.
Widening the blocks would lift both nets together and leave the depth-utilization gap where it is; the
evidence in front of me is specifically that the extra 27 blocks of the 110-net are wasted, which is a
through-stack symptom, not a too-small-model symptom. So capacity is not the lever. The lever is whatever is
stopping those extra blocks from contributing, and that is a property of the path they sit on.

So let me look hard at the piece I froze. The default block — and the gated block, which kept the default
ordering — is *post-activation*: `out = ReLU(BN(conv1(x)))`, `out = BN(conv2(out))`, then the addition, then
a **final ReLU on the sum**, `H = ReLU(shortcut(x) + alpha·F(x))`. Two things on that identity path bother me
when I imagine stacking it 110 deep. First, that last ReLU sits *after* the addition, so the value carried
from one block's output to the next block's input is `ReLU(...)` — every block's output is forced
non-negative before the next block and the next shortcut ever see it. Second, follow the shortcut across many
blocks within a stage, where it is a bare identity: the map from the input of block ℓ to its output is
`x ↦ ReLU(x + F(x))`. Chain that across the whole stage and the clean additive identity I wanted —
`x_out = x_in + Σ F` — is interrupted at *every single block* by a rectifier. The identity path is not
actually clean; it is a sequence of additions each clamped through a ReLU.

Let me quantify how badly that clamping hurts, because "a ReLU at every block" is only alarming if I can show
it destroys the property the identity path exists to provide. Differentiate the post-activation recurrence
`x_{ℓ+1} = ReLU(x_ℓ + F_ℓ(x_ℓ))`. The block Jacobian is `diag(m_ℓ)·(I + ∂F_ℓ/∂x)`, where `m_ℓ ∈ {0,1}` is
the ReLU's derivative mask at that block's output. Backpropagating across a stage of `k` blocks multiplies
these together, and the "clean" term — the product of the identity parts that is supposed to carry the
gradient to the bottom undiminished — is `diag(m_k)·…·diag(m_1)`, a product of `k` binary masks. For a
roughly zero-mean pre-activation each `m` passes a given coordinate with probability about `1/2`, so a
specific coordinate survives all `k` clamps with probability about `2^{−k}`. In a ResNet-110 stage of 18
blocks that is `2^{−18} ≈ 3.8×10⁻⁶`; across all 54 blocks it is `2^{−54} ≈ 5.6×10⁻¹⁷`. The identity path's
whole promise was "a gradient at the top reaches the bottom along a path of all 1's," and the post-activation
mask erases that promise coordinate by coordinate — the pure `I` route survives essentially nowhere at depth
110. Gradient still reaches the bottom through the `∂F` terms, but that is exactly the diminishable route the
residual reformulation was meant to make optional. At depth 20 the same product is only `2^{−6}` per stage
and the clamps barely matter; at depth 110 they are the difference between a real highway and a rectified
trickle. That depth dependence matches the shape of where the gated numbers left headroom precisely.

The two CIFAR-100 nets sharpen it further, and this is what makes me confident the clamps are the culprit and
not something I am pattern-matching onto. ResNet-56 has 9 within-stage blocks per stage, so its per-stage
mask survival is about `2^{−9} ≈ 2.0×10⁻³`; ResNet-110 has 18, so its per-stage survival is about
`2^{−18} ≈ 3.8×10⁻⁶`. The ratio is `2^9 = 512`: the identity path in the deepest net is roughly five hundred
times more thoroughly clamped, per stage, than in the merely-deep net. If the clamps are what is capping the
deep nets, then a fix that removes them should help ResNet-110 far more than ResNet-56 — the 512× asymmetry
predicts an ordering, not just a lift — and that is a much more specific claim than "deeper is better," one
the three-setting table can confirm or refute.

Let me make the mechanism completely concrete with a two-block hand trace, because the mask argument is only
convincing if I can point to a case where it transmits *nothing*. Take two stacked within-stage blocks and
follow one coordinate. Post-activation: `x₂ = ReLU(x₁ + f₁(x₁))`, `x₃ = ReLU(x₂ + f₂(x₂))`, so
`∂x₃/∂x₁ = m₂·(1 + f₂')·m₁·(1 + f₁')` with `m₁, m₂ ∈ {0,1}` the two ReLU masks. Now put the block in the
regime the deep-refining stall is about — quiet blocks contributing little locally, `f₁' ≈ f₂' ≈ 0` — and the
derivative collapses to `m₂·m₁`, a product of two indicators, which is `0` at any coordinate where either
pre-ReLU sum happened to be negative. That coordinate transmits *zero* gradient to `x₁`. Pre-activation on the
same coordinate: `x₂ = x₁ + f₁(x₁)`, `x₃ = x₂ + f₂(x₂)`, so `∂x₃/∂x₁ = (1 + f₂')(1 + f₁') = 1` when the
branches are quiet — full transmission, guaranteed, regardless of any sign. So at exactly the blocks the deep
net most needs to pass gradient through untouched — the quiet late ones — the post-activation path can carry
none while the pre-activation path carries all of it. That is not a stylistic difference; it is the identity
highway working versus not working.

What I want, then, is a block where the identity path is *genuinely* unobstructed: the output of block ℓ
should be addable to block ℓ+1's branch without a nonlinearity sitting on the through-line, so that across a
stage the map is the pure sum `x_out = x_in + Σ_ℓ F_ℓ(x_ℓ)`. If I can arrange that, the stage Jacobian
becomes `I + Σ ∂F_ℓ` with no mask, and a gradient at the top reaches the bottom along the `I` term no matter
how small the `∂F_ℓ` factors get — which is the property a 110-layer net needs, a 56-layer net needs less,
and a 20-layer net hardly needs at all.

The obstruction is the activation order. The post-activation block computes the nonlinearity *after* the
addition because the convs come first and BN/ReLU follow them; the only way to get the ReLU off the
through-line is to move it *inside* the branch, ahead of the convolutions. So reorder to **pre-activation**:
each conv is preceded by its own BN and ReLU rather than followed by them. The branch becomes
`F(x) = conv2( ReLU(BN( conv1( ReLU(BN(x)) ) )) )` — BN-ReLU-Conv, twice — and the block output is just
`shortcut(x) + F(x)`, with **no activation after the addition**. Now the branch consumes a *pre-activated*
copy of `x`, does its work, and the result is added to the shortcut with nothing rectifying the sum; the next
block applies *its own* entrance BN-ReLU to whatever it receives. So the signal that travels block-to-block
along the shortcut is the raw additive stream, never clamped; each block's nonlinearity acts only on the
*copy* it pulls into its branch, not on the highway.

Before I accept the full reorder I should ask whether a smaller edit gets the same win, because if it does
the extra change is unjustified. The tempting half-measure is to keep the post-activation order but simply
*delete the final ReLU*: `out = BN(conv2(ReLU(BN(conv1(x)))))`, then `return shortcut(x) + out`, no
post-add rectifier. That alone gives the clean backward path — the through-line Jacobian is `I + ∂F`, the
masks are gone — so it captures the gradient-flow benefit I just argued for. But trace what `conv1` consumes:
in this half-measure `conv1` reads the *raw* additive stream `x` directly, with no normalization at the
branch entrance. And the raw stream is exactly the thing that grows without a bound in a mask-free residual
net (I work its variance out below), so `conv1` would be fed an increasingly badly-scaled input as depth
grows. Full pre-activation fixes both at once: putting BN at the *entrance* of each branch means the input to
*every* convolution in the network is normalized, a strictly better-conditioned input than a conv consuming a
raw growing sum. The half-measure buys the backward win and leaves the forward-conditioning win on the table;
since both effects scale with depth and the whole point is to maximize the deep-net payoff, I take the full
reorder. That is the real reason it is a reorder and not merely a deletion.

I want to double-check this is a genuine improvement and not a cosmetic shuffle, because moving BN and ReLU
around a conv feels like it shouldn't matter. It matters for two concrete, separable reasons. Backward: with
no post-addition ReLU, the derivative of `x_out` with respect to the stage input is exactly `I + Σ ∂F_ℓ`, an
identity plus corrections — the gradient at the top is *guaranteed* to reach the bottom undiminished along
the `I` term, no matter how small the `∂F_ℓ` get, replacing the `2^{−54}` mask-survival above with a clean
1. Forward: putting BN at each branch entrance normalizes every conv's input. Both effects scale with depth,
which is exactly why I expect pre-activation to help ResNet-110 most, ResNet-56 next, and ResNet-20 barely —
the same depth-sorted shape the gated numbers exposed, now with a mechanism attached.

The forward benefit is worth pinning to the *fixed* optimizer, because that is what makes it bite. The recipe
hands every layer one global learning rate, `lr = 0.1`, and a conv's effective update size scales with the
magnitude of the input it convolves — the weight gradient is a product of the incoming gradient and the layer
input. In the post-activation block `conv1` consumes the raw through-line, whose variance compounds with
depth (the ~19×-per-stage growth I quantify below), so different convs across the stack see systematically
different input scales and therefore experience the single global `lr` very differently — one number is a
compromise that is too large for the deep, large-input layers and too small for the shallow ones.
Pre-activation pins `conv1`'s input to unit variance via `bn1` at *every* depth, so every convolution sees
the same input scale and the same effective per-layer step, and the one global `lr = 0.1` is simultaneously
appropriate for all of them. That is a real, depth-scaling reason the reorder helps beyond the backward mask
argument — and it costs nothing, since it only relocates BN layers that were already present.

Now the substrate-specific care, because the task's block differs from the textbook pre-activation idea and I
have to derive the *actual* fill, not the generic one. Two deviations, both deliberate. First, the shortcut
on the dimension-changing blocks. In a fully "pure" identity-mapping block the shortcut would also be a
clean, parameter-free path, but the stride-2, channel-doubling transitions — the first block of stages 2 and
3 — cannot be a bare identity, the shapes do not match. This scaffold resolves that with a *projection*: a
1×1 stride-2 conv followed by BN, applied to the **raw input `x`**, not to the pre-activated input. So the
through-line is the clean unobstructed identity only on the *within-stage* blocks; at the two transition
blocks per net the shortcut is a small Conv-BN projection of `x`. That is the right choice here — putting BN
on that projection keeps the two transition points well-scaled, and there are only two of them per net (out
of 9, 27, or 54 blocks), so the overwhelming majority of blocks still enjoy the clean additive highway. The
branch reads the pre-activated `x` while the projection reads the raw `x`, which is the natural wiring when
the shortcut needs its own normalization anyway.

Second, and this is the deviation I would not have reached for on my own but which the substrate's depth
demands: a fixed-small **residual scale**, and I have to derive its value rather than borrow one. Work out the
variance of the through-line at init. In pre-activation nothing normalizes the *within-stage* shortcut, so
the stream accumulates: `x_out = x_in + Σ_ℓ alpha·F_ℓ`, and each freshly-initialized `F_ℓ` (ending in
`conv2` on a BN-ReLU-normalized input) carries roughly unit per-channel variance. Treating the 18 branches
of a stage as roughly independent, the variance the residuals add to the stream is about `18·alpha²`. At
`alpha = 1` that is ~18, so a stream entering the stage with variance ~1 leaves it with variance ~19 — a
roughly 19× blow-up *within a single stage*, before the next transition's BN can renormalize, which is
exactly the kind of scale explosion that makes the first epochs of a very deep pre-activation net jittery. At
`alpha = 0.1` the added variance is `18·0.01 ≈ 0.18`, so the stream grows from ~1 to ~1.18 — the clean
identity path stays dominant and the residuals are a gentle perturbation the convs can grow into. So I scale
each branch by a learnable `alpha` initialized to **0.1**: the net *begins* close to the identity plus a soft
residual, and SGD is free to grow `alpha` toward where each block wants it.

Is 0.1 the right point on that interval, or just a round number? Push the same `18·alpha²` arithmetic to the
neighbors and it stops being arbitrary. At `alpha = 0.2` the residuals add `18·0.04 ≈ 0.72` to a stream
entering with variance ~1, so it leaves the stage at ~1.72 — a 72% swell per stage, which across three stages
and before each transition's BN can renormalize is enough scale drift to make the opening epochs of the
depth-110 stack jittery again, the very thing the soft-start is meant to avoid. At `alpha = 0.05` the added
variance is `18·0.0025 ≈ 0.045`, so the stream barely moves, ~1.045 — but now each branch contributes under
five percent of the running variance at init, and against the weight decay simultaneously pulling `alpha` down,
so timid a start means the residual must climb a long way before it does real work, wasting the early budget
the "0.1 not 0" argument exists to preserve. 0.1 is the value where the added per-stage variance (~0.18) is a
genuine fraction of the stream — real capacity from step one — yet small enough that the identity term stays
clearly dominant and the init stays stable. The neighbors bracket it cleanly: 0.2 trades stability for a fast
start, 0.05 trades a fast start for stability, and 0.1 is the interior point that buys enough of both.

The initialization value carries the one lesson the gated rung already taught me. I do *not* init this scale
at zero — that would waste the early budget on an effectively-shallow net against a fixed 200-epoch schedule,
the same mistake I rejected last time — and I do not init at 1, which the variance arithmetic just showed
blows the stream up 19× per stage. 0.1 is the middle: small enough to soft-start the depth-110 stack, large
enough that every block still contributes a real fraction of a block's worth of capacity from step one. Note
this is a *different role* than the gated rung's scalar: there it started at 1 to *match* the baseline and
re-weight around it; here it starts at 0.1 to *stabilize* a much deeper, mask-free stack at init while still
growing in. The two scalars look alike and mean opposite things — the gate at 1 says "behave like the
baseline, adjust," the pre-activation scale at 0.1 says "start near identity, grow."

It is worth estimating how fast "grow" happens, because the whole warm-up cost lives in that transient and it
is what I will pay at shallow depth. The gradient on `alpha` is `<g, F(x)>` — the alignment of the branch
output with the incoming gradient — and for a block the loss genuinely wants, that drive is persistently
positive, pushing `alpha` up. The frozen recipe runs 50k images at batch 128, about `391` steps per epoch,
under a cosine schedule whose mean learning rate is near `0.05`, and momentum 0.9 amplifies the effective step
by roughly `1/(1 − 0.9) = 10×`. So a block carrying a steady, order-unit drive on its `alpha` accumulates an
effective step of order `0.05·10 ≈ 0.5` over a short run of iterations — meaning the climb from `0.1` toward
order `1` is a matter of the first handful of epochs, not the whole schedule. That is the asymmetry that sorts
the depth sweep: the `0.1` soft-start is a *front-loaded* cost, spent in the opening epochs and repaid across
the rest of training. On a very deep net that cost buys a clean highway whose benefit compounds over 54 blocks,
so it comes out ahead; on ResNet-20, where the reorder repairs almost nothing — three blocks per stage barely
feel the clamps — those same opening epochs of near-identity are a small net loss with little clamp damage to
offset them, which is exactly why I expect the shallow number to land a *hair under* the gated 92.96 rather
than above it.

Two limit checks confirm the scale is doing what I think. As `alpha → 0` the within-stage blocks collapse to
`x_out = x_in`, a pure identity, so the net degenerates cleanly to stem → (two transition projections) → pool
— a well-defined, shallow-but-valid mapping, which is why a soft start near this limit is safe rather than
broken. As `alpha` grows large the identity term becomes negligible relative to the residuals and the block
behaves like a plain, mask-free deep conv stack — exactly the un-soft-started regime whose init variance I
just showed blows up 19× per stage. 0.1 sits deliberately near the safe end of that interval: close enough to
the identity limit to start stable, far enough from it to carry a real fraction of a block's capacity from
step one. And because `alpha` is learnable, SGD can walk it toward the plain-stack end wherever a block wants
more residual, so I am fixing only the *starting point* of the interval, not confining the block to it.

One budget check to be sure this is a reorder and not a capacity change smuggled in as one. Post-activation
has `bn1`, `bn2` each on `planes` channels; pre-activation has `bn1` on `in_planes` and `bn2` on `planes`.
For within-stage blocks `in_planes = planes`, so the BN parameter count is identical; at the two transition
blocks `bn1` is on the *smaller* incoming width, so pre-activation actually carries a hair *fewer* BN
parameters there. The two convs, their kernel sizes, and the dimension-matching logic are unchanged, and the
only genuine addition is the one scalar `alpha` per block. So the parameter budget is, to within one scalar
per block, identical to the gated rung's — what changed is the *order*, and the order is precisely the thing
the gated numbers said was limiting the deep nets. This is a structural intervention that costs nothing in
capacity, which is exactly the kind I want when the diagnosis is "wiring, not size."

So the edit relative to the gated block is: move both BNs and ReLUs to *precede* their convs (BN-ReLU-Conv
ordering), delete the post-addition ReLU so the through-line is a pure sum, route the dimension-change
shortcut through a Conv-BN projection of the raw input, and scale the branch by a 0.1-initialized learnable
`alpha`. The full module is in the answer.

Now the falsifiable expectations against the gated rung's measured numbers, because the whole point of
reordering is a depth-dependent payoff and the three settings are a depth sweep. On ResNet-20/CIFAR-10 the
clean highway buys little — 20 layers barely feel the repeated post-activation clamps (`2^{−6}` per stage,
not `2^{−18}`) — so I expect to land *around* the gated 92.96, possibly a hair under, because at shallow
depth the 0.1 residual init costs a few epochs of warm-up the gated block did not pay. That is fine; the
shallow case was never where the room was. The real test is the deep CIFAR-100 nets, where the gate stalled
at 71.98 (56) and 73.46 (110). If the diagnosis is right — that the limit there was the obstructed identity
path, not the residual magnitude — then pre-activation should lift the *very deep* net most: I expect
ResNet-110 to clear the gated 73.46 by the largest margin of the three settings, and ideally to pull
decisively above the ResNet-56 number, because a clean additive highway is exactly what the 54-block net was
missing. ResNet-56 should improve too but by less, since 27 blocks suffered the clamps less than 54. The
falsifiable failure mode is explicit: if the gate's problem had really been residual magnitude rather than
wiring, pre-activation would *not* differentiate the deep nets from the shallow one, and all three settings
would move together — if I see that flat, un-depth-sorted response in the `test_acc` table, the reordering
story is wrong. And one risk I flag now for the next rung: pre-activation fixes how signal flows *through*
the depth, but it does nothing to make the deep net *exploit* its full depth or to regularize the
100-class fit — so if the deep CIFAR-100 numbers improve on gradient flow but still leave the very deep net
under-generalizing, the next lever is not flow but how the deep net's 100-class fit is *regularized across its
depth*, which a per-block scalar reordering can never provide.
