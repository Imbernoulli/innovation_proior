The single sign step landed where the linearization argument said it would, and the shape of the
result is the whole story. FGSM scored 0.825 / 0.839 / 0.825 on ResNet20-C10 / ResNet20-C100 /
MobileNetV2-C100, 0.878 on MobileNetV2-C10 — already a large majority flipped on the bottlenecked and
depthwise architectures — but only 0.617 and 0.621 on VGG11BN-C10 and VGG11BN-C100. That is the split
I expected, and it is diagnostic, not noise. One signed jump of size `eps` in every coordinate
maximizes the loss's *tangent plane* at the clean point, and over a full-budget jump the tangent plane
stops describing the true loss surface. Wherever the loss bends fast away from its linearization, the
corner FGSM lands on is not where the loss is actually high — and VGG11-BN, whose wider, shallower
low-resolution feature maps absorb small per-pixel perturbations more robustly, is exactly where that
bending costs the most. So roughly 38% of VGG samples survived not because they are safe but because
one linearized step could not reach them. The fix is forced by the diagnosis: stop trusting one
linearization across the whole `eps`-box, and re-read the slope as I move.

Let me read the survivor counts off the feedback exactly, because the *ratio* is what tells me where
the headroom lives. FGSM's ASR is the flipped fraction, so `1 − ASR` is the survivors: ResNet20 leaves
`0.175` (C10) and `0.161` (C100), MobileNetV2 leaves `0.122` (C10) and `0.175` (C100), while VGG11-BN
leaves `0.383` and `0.379`. Average the non-VGG survivors, `(0.175 + 0.161 + 0.122 + 0.175)/4 = 0.158`,
and compare to the VGG average `(0.383 + 0.379)/2 = 0.381`: VGG carries `0.381/0.158 = 2.4×` the
survivor mass of the other architectures. The mean ASR is `0.768`, and essentially the entire deficit
from `1.0` is those two VGG columns plus a thin residual on the easy ones. So the diagnosis is not a
vague "VGG is harder" — it is that a single linearized step under-reaches on a concentrated,
identifiable `~2.4×`-denser pool of samples, and any method that spends more gradient reads should cash
out overwhelmingly on VGG and barely move the already-saturated columns. That asymmetry is the ruler I
will hold the next method against.

I cannot trust a single tangent plane over a `2eps`-wide box, so I take a *small* step in the sign
direction, re-evaluate the gradient where I land, step again, and repeat. With step size `alpha`,
`x_{t+1} = x_t + alpha·sign(∇_x J(θ, x_t, y))`, projecting back into the budget each step. Each step is
still the `L_inf`-steepest move — sign of the gradient, for exactly the Hölder reason FGSM used — but
now read from the *current* point, so I follow the curved surface instead of betting everything on the
plane at `x`. This iterative sign-ascent is what I'd expect to recover the VGG headroom: the samples
that survived FGSM are the ones where the surface curves, and that is precisely what extra
re-linearized steps correct. But before I just run iterative FGSM, I want to ask whether iterating
*greedily* on the raw gradient is the best use of those steps, because there is a known pathology in
greedy gradient ascent that I can pre-empt cheaply, and the harness lets me.

Here is the pathology. The classifier's loss surface around a point is curved and bumpy; the raw
input-gradient jitters from step to step as the iterate moves across that bumpiness. A greedy
sign-ascent that chases whatever the local slope says *right now* zig-zags — consecutive perturbation
increments do not pull consistently one way — and a zig-zagging ascent dives into the nearest sharp
local maximum of the loss. In the classic transfer setting that is fatal, because those sharp maxima
are one model's private quirks and do not generalize to a different model. In *this* task I attack and
am graded by the same architecture, so transfer is not the headline concern — but the optimization
pathology is the same one, and it still hurts the white-box number: a direction that zig-zags wastes
steps oscillating instead of climbing, and it can pin the iterate in a shallow sharp maximum well
short of the high-loss region a smoother trajectory would reach. So I want the iterative scheme — it
is what recovers FGSM's underfit — but with the per-step direction stabilized so it settles along the
component of the gradient that keeps pointing the same way and coasts through the small sharp humps
rather than getting caught in them.

Once I name the attack correctly, the cure is obvious. Iterative sign-FGSM is plain sign-gradient
ascent on `J`, and zig-zagging into poor local optima on a noisy surface is the oldest pathology in
optimization, with an equally old remedy: momentum. Polyak's trick — do not step on the raw gradient,
step on an accumulated velocity `g_{t+1} = μ·g_t + (current gradient)` that averages the recent
gradients. The averaging cancels the components that flip sign step to step (the zig-zag) and
reinforces the components that persist (the consistent ascent direction), and the built-up velocity
carries the iterate over small humps instead of letting each one trap it. The mechanism is a low-pass
filter on the gradient sequence: a coordinate whose sign oscillates `+,−,+,−` across steps sums toward
zero in the velocity and contributes little to the step, while a coordinate that points the same way
every step accumulates coherently and dominates — the accumulator keeps the DC component of the
gradient and discards the alternating part. Everything I said I wanted —
stabilize the direction, keep the consistent part, escape the sharp maxima — is what momentum does. So
I put a velocity into the iteration: maintain `g`, accumulate the gradient into it each step, and take
the pixel step along `g` rather than the bare current gradient. I should be clear-eyed that in the classic
telling momentum's headline payoff is *transfer* — the averaged direction keeps the component shared
across models and sheds one model's private quirks — and that payoff is not on the table here, since I
attack and am graded by one architecture. But the *optimization* half of momentum's story is
model-agnostic: whether or not a second model exists, a running average of directions still cancels the
step-to-step sign flips and still coasts the iterate over small humps on a single bumpy surface. So I keep
momentum for the optimization reason and explicitly do *not* import the transfer justification as the
operative one; that distinction will matter when I set the step size and the start, both of which I tune
for a strong single-model climb rather than for cross-model generalization.

Two details decide the method, and I want to derive both, not guess them. Take the step first, because
it is forced. The budget is `L_inf`, and I already know the `L_inf`-optimal move for a given direction
is the sign: each nonzero coordinate goes `±alpha`, no coordinate moves more than `alpha`. So the
update is `x_{t+1} = x_t + alpha·sign(g_{t+1})`. Same reason FGSM used the sign, and it buys explicit
per-step budget control. Now the accumulation, where the naive `g_{t+1} = μ·g_t + ∇_x J(x_t, y)` has a
problem that would quietly wreck the momentum. The magnitude of the input gradient is not stable
across iterations — large far from a boundary, small near one, sometimes spiking. Dump the raw
gradient into `g` and a single big-magnitude iteration dominates the running sum while a small one
contributes almost nothing; the "average" becomes whichever step happened to have the largest
gradient, which is the very magnitude-noise I am trying to smooth out, sneaking back in. Momentum must
be an average of *directions* over time, every step getting a fair vote. So I normalize the current
gradient before accumulating: `g_{t+1} = μ·g_t + ∇_x J(x_t, y) / ||∇_x J(x_t, y)||_1`. Dividing by the
gradient's total absolute mass makes each iteration's contribution a unit-mass vector, so `μ` becomes
a clean trade-off between accumulated history and the present direction rather than between whatever
raw magnitudes the surface handed me. In code this is naturally written as dividing by the mean
absolute value over the pixel dimensions — proportional to the per-sample `L_1` norm — and since I
take `sign(g_{t+1})` downstream, only the *pattern of relative magnitudes across coordinates* in
`g_{t+1}` matters; the per-iteration normalization keeps old and new gradients comparable so neither a
stale big-gradient step nor a fresh one can unilaterally flip the accumulated direction.

Let me make the magnitude problem concrete with a two-coordinate, two-step trace, because "a big step
dominates the sum" is the kind of claim I should be able to compute. Suppose the first gradient is
`g_1 = (100, −100)` — total mass `||g_1||_1 = 200`, pointing down-right — and the next, at the point I
land on, is `g_2 = (1, 3)`, total mass `4`, pointing up-right; the second reading disagrees with the
first about coordinate two. With raw accumulation `g = g_1 + g_2 = (101, −97)`, so `sign(g) = (+, −)`:
the stale, hundred-times-larger `g_1` completely drowns `g_2`, and the step still drives coordinate two
*down* even though the freshly-read gradient says up. That is the failure — the "momentum" is just the
biggest single gradient. Now normalize each before summing: `g_1/200 = (0.5, −0.5)` and
`g_2/4 = (0.25, 0.75)`, sum `(0.75, 0.25)`, `sign = (+, +)` — the unit-mass vote lets the newer
direction actually overturn coordinate two, which is what an average of directions should do. So the
`L_1` normalization is not cosmetic; it is the difference between the momentum tracking the consensus
direction and it tracking whichever step had the largest gradient.

Now the two constants this task's harness sets. The decay is `μ = 1`: the recursion
`g_{t+1} = g_t + grad/||grad||_1` simply *sums* all the normalized gradients seen so far with equal
weight — undiscounted accumulation of the consensus direction, with no decay throwing away history and
no blow-up of any single step because each addend is unit-mass. So `g_t` means the running sum of all
past unit-normalized gradient directions. It is worth asking why `μ = 1` and not the more common
`μ = 0.9` a momentum optimizer would reach for. A discount `μ < 1` down-weights the older gradients
geometrically — after `k` further steps a contribution has weight `μ^k`, so at `μ = 0.9` a gradient
from 20 steps ago retains only `0.9^20 ≈ 0.12` of its vote. That decay is what you want when the early
iterates are in a genuinely different, now-irrelevant region and their gradients are stale. But here
the whole trajectory lives inside one tiny `2·eps`-wide box; every point I visit is within `0.016` of
every other in each coordinate, so the "old" gradients are not from a foreign region — they are votes
from a neighborhood essentially as relevant as the current one, and the honest thing is to let them all
count equally. Undiscounted summation also cannot blow up, because each addend is unit-`L_1`-mass and I
take the sign at the end, so the running sum's *scale* is irrelevant and only its coordinate pattern
survives. So `μ = 1` is the right default for a short in-box climb: maximal smoothing, no arbitrary
horizon, and no instability to pay for it. And there is a clean consistency check that I have built the
right generalization rather than a third unrelated trick: set `μ = 0` and `g_{t+1} = grad/||grad||_1`,
whose sign is `sign(grad/||grad||_1)`. Since `||grad||_1` is a positive scalar, dividing by it cannot
flip any coordinate's sign, so `sign(grad/||grad||_1) = sign(grad)` exactly and the update becomes
`x_{t+1} = x_t + alpha·sign(grad)` — plain iterative sign-FGSM, the normalization completely annihilated
by the downstream sign. Take a single such step of size `eps` from the clean point and I recover FGSM.
So the two ancestors sit at the endpoints `μ = 0` (iterative FGSM) and `μ = 0, T = 1` (FGSM), and
`μ = 1` interpolates a smoothed direction between them — the family contains both, which is the test
that I generalized rather than invented. It also tells me what running plain iterative FGSM (the `μ = 0`
member, sometimes called BIM) would buy on its own: it already recovers the tangent-plane underfit by
re-reading the gradient, so it should lift VGG well above FGSM — but with nothing damping the step-to-
step sign flips it climbs the bumpy surface greedily, oscillating near the boundary and settling in
whatever sharp maximum it stumbles into first. The `μ = 1` velocity is the cheap upgrade over that: same
40 gradient reads, but the accumulated direction coasts through the small humps instead of being
deflected by each one. Since the two cost the same, keeping the momentum is free insurance.

The step count and step size matter more than they look, and this is where the task's fill diverges
from the bare momentum recipe in a way I should be explicit about. The natural momentum-attack
default ties `alpha = eps/T` so that `T` aligned sign steps exactly fill the budget. Let me price that
default out before rejecting it. With `T = 40` the tied step is `alpha = eps/40`, and `40` aligned
steps then reach `40·eps/40 = eps` of travel in a coordinate — but the box diameter is `2·eps` (from
`−eps` to `+eps`), and if the random start already sits near one wall the iterate cannot even cross to
the opposite wall, let alone settle there. Worse, near the high-loss boundary the surface bends and the
sign flips some coordinates, so part of the `eps` of nominal travel is spent undoing earlier moves; the
effective reach is *less* than `eps`. So the tied `alpha = eps/T` budgets the iterate to barely graze
the box and leaves it stranded before it can refine — it is tuned for a regime where a handful of steps
is all you get, not for `40`. This task instead runs `T = 40` steps with `alpha = eps/10`, and starts
from a *random* point in the `eps`-box rather than from the clean image. Both choices push the method
toward the strong-white-box regime rather than the transfer regime. With `alpha = eps/10` and 40 steps,
the total reach is `40·eps/10 = 4·eps`, twice the `2·eps` box diameter, so from *any* interior start
the iterate can reach the far wall in ~20 aligned steps and still has ~20 left to refine on the
high-loss boundary — more gradient updates correcting stale-gradient error rather than trusting the
first plane. The projection guarantees this larger step is safe: `alpha = eps/10` never lets a single
step exceed the budget on its own, and the per-step clip to `[−eps, eps]` catches the accumulated
overshoot, so a bigger step buys faster exploration at no risk of leaving the box.

Why `eps/10` in particular, between the tied `eps/40` I just rejected and a coarse `eps/4`? The step
size sets the resolution at which the iterate can sit on the boundary: once pinned, the smallest move
it can make in any coordinate is `alpha`, so `alpha` is the grain of the final refinement. At
`eps/10 = 0.000784` the iterate can settle to within one-tenth of the budget of the true boundary
optimum, fine enough that the residual is well below the harness's `1e-6` slack and the curvature it is
chasing on VGG. A coarser `eps/4` would reach the boundary in ~8 steps but could only refine in
quarter-budget jumps, chattering around the optimum; `eps/40` refines beautifully but, as computed,
never reliably crosses the box. `eps/10` with 40 steps is the balance the fill picks: `4·eps` of total
reach so exploration is never the bottleneck, and a `0.1·eps` grain so refinement is fine — momentum
handles the direction, this handles the resolution. And the random start
matters for exactly the reason it mattered for the iterative attack generally: the gradient evaluated
at the clean point is corrupted by sharp curvature artifacts localized at `x`, so launching from a
uniform perturbation in the `eps`-box, clipped to `[0,1]`, jumps off that brittle point before the
first gradient read and lets the momentum accumulate from a better-behaved direction. Quantitatively
the launch is not timid: each coordinate starts uniform on `[−eps, eps]`, so the expected `L_2` offset
from the clean image is `sqrt(3072)·eps/sqrt(3) = 55.4 · 0.00784 / 1.73 ≈ 0.25`, a substantial jump
already halfway across the box in Euclidean terms, which is precisely why the first gradient is read
somewhere other than the pathological clean point. It costs nothing — one `uniform_` fill and a clamp
before the loop — and it means the momentum's first vote comes from a typical interior point rather
than the one place the surface is most jagged. In a pure
transfer-motivated momentum attack the random start is not part of the recipe; here it is in the fill,
and it makes this method a genuine member of the projected-iteration family — iterative sign-ascent
with a momentum-smoothed direction and a random launch — rather than a transfer-tuned variant. So I
should not import the transfer story as the *reason* this works on the leaderboard; on a single-model
white-box budget, the operative reasons are that iterating recovers FGSM's underfit and that momentum
plus a random start gives a steadier, less-trapped ascent than greedy iterative FGSM.

Before I commit I should confirm the cost is still nothing next to the method I rejected at rung one.
MI-FGSM runs `T = 40` forward/backward pairs per batch instead of FGSM's one, so across `60` batches
that is `2400` network evaluations for the whole benchmark. The per-image L-BFGS existence method I
discarded cost on the order of `6·10^5`; MI-FGSM at `2400` is still `~250×` cheaper than that, and only
`40×` FGSM's cost, for what the survivor arithmetic says should be a large jump on the VGG columns. The
momentum buffer, the `L_1` normalization, and the random start add three elementwise tensor ops per
step — negligible against the forward/backward. So the budget question is settled: I can afford to
re-read the gradient 40 times, and the only real design questions were the two I derived, the sign step
and the direction-averaged accumulation.

There is one more thing the diagnosis tells me to keep modest. The samples FGSM already flipped — the
~82–88% on ResNet and MobileNetV2 — are the easy ones where even the tangent plane reached the
boundary; iterating cannot do worse on those, it pins them harder. The marginal value of momentum is
concentrated on the survivors, the curved-surface samples, and those are concentrated on VGG. So the
clean falsifiable expectation is asymmetric. I expect MI-FGSM to roughly match FGSM where FGSM already
saturated (ResNet20-C10/C100, MobileNetV2 should land in the high-0.96–0.99 band once iterated) and to
make its real gain on the two VGG scenarios, lifting them substantially above FGSM's 0.617 and 0.621 —
though, because VGG's robustness to small `L_inf` is architectural, I do not expect it to close the
gap to 1.0; I expect it to land in the low-0.7s, recovering most but not all of the underfit. Put a
number on "most but not all": FGSM left `0.383` VGG-C10 survivors; if iterating recovers, say, a
quarter of them the ASR climbs from `0.617` toward `~0.71`, which is the low-0.7s I am predicting — a
`~0.1` absolute jump on VGG against a near-zero move on the already-saturated columns. That is the
asymmetric signature the survivor ratio demanded: the `2.4×`-denser VGG pool is where 39 extra gradient
reads have room to work, and the easy columns, already at `0.83–0.88`, can only creep the last few
points to the ceiling. If instead the two VGG columns barely move, the survivors are not
curvature-underfit but genuinely robust, and I will have to look past first-order sign-ascent
altogether. The
contract fill — 40 iterations of normalize-accumulate-sign-step from a random start, projecting the
perturbation to `[-eps,eps]` and the image to `[0,1]` each step — is in the answer. If instead MI-FGSM
fails to beat FGSM on VGG, my whole diagnosis (that the VGG survivors are curvature-underfit, not
genuinely safe) is wrong, and the headroom is architectural robustness no first-order iteration can
touch — a question the next rung would then have to settle.
