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
carries the iterate over small humps instead of letting each one trap it. Everything I said I wanted —
stabilize the direction, keep the consistent part, escape the sharp maxima — is what momentum does. So
I put a velocity into the iteration: maintain `g`, accumulate the gradient into it each step, and take
the pixel step along `g` rather than the bare current gradient.

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

Now the two constants this task's harness sets. The decay is `μ = 1`: the recursion
`g_{t+1} = g_t + grad/||grad||_1` simply *sums* all the normalized gradients seen so far with equal
weight — undiscounted accumulation of the consensus direction, with no decay throwing away history and
no blow-up of any single step because each addend is unit-mass. So `g_t` means the running sum of all
past unit-normalized gradient directions. And there is a clean consistency check that I have built the
right generalization rather than a third unrelated trick: set `μ = 0` and `g_{t+1} = grad/||grad||_1`,
whose sign is just `sign(grad)`, so I recover iterative sign-FGSM exactly (the normalization is
annihilated by the sign); take a single step from the clean point and I recover FGSM. The family
contains both ancestors and interpolates between them — that is the test of the right generalization.

The step count and step size matter more than they look, and this is where the task's fill diverges
from the bare momentum recipe in a way I should be explicit about. The natural momentum-attack
default ties `alpha = eps/T` so that `T` aligned sign steps exactly fill the budget. This task instead
runs `T = 40` steps with `alpha = eps/10`, and starts from a *random* point in the `eps`-box rather
than from the clean image. Both choices push the method toward the strong-white-box regime rather than
the transfer regime. With `alpha = eps/10` and 40 steps, the total reach is `40·eps/10 = 4·eps`, well
past the `2·eps` box diameter, so after the iterate is pinned to the high-loss boundary it spends its
remaining steps refining its position there instead of merely crawling toward it — more gradient
updates correcting stale-gradient error rather than trusting the first plane. And the random start
matters for exactly the reason it mattered for the iterative attack generally: the gradient evaluated
at the clean point is corrupted by sharp curvature artifacts localized at `x`, so launching from a
uniform perturbation in the `eps`-box, clipped to `[0,1]`, jumps off that brittle point before the
first gradient read and lets the momentum accumulate from a better-behaved direction. In a pure
transfer-motivated momentum attack the random start is not part of the recipe; here it is in the fill,
and it makes this method a genuine member of the projected-iteration family — iterative sign-ascent
with a momentum-smoothed direction and a random launch — rather than a transfer-tuned variant. So I
should not import the transfer story as the *reason* this works on the leaderboard; on a single-model
white-box budget, the operative reasons are that iterating recovers FGSM's underfit and that momentum
plus a random start gives a steadier, less-trapped ascent than greedy iterative FGSM.

There is one more thing the diagnosis tells me to keep modest. The samples FGSM already flipped — the
~82–88% on ResNet and MobileNetV2 — are the easy ones where even the tangent plane reached the
boundary; iterating cannot do worse on those, it pins them harder. The marginal value of momentum is
concentrated on the survivors, the curved-surface samples, and those are concentrated on VGG. So the
clean falsifiable expectation is asymmetric. I expect MI-FGSM to roughly match FGSM where FGSM already
saturated (ResNet20-C10/C100, MobileNetV2 should land in the high-0.96–0.99 band once iterated) and to
make its real gain on the two VGG scenarios, lifting them substantially above FGSM's 0.617 and 0.621 —
though, because VGG's robustness to small `L_inf` is architectural, I do not expect it to close the
gap to 1.0; I expect it to land in the low-0.7s, recovering most but not all of the underfit. The
contract fill — 40 iterations of normalize-accumulate-sign-step from a random start, projecting the
perturbation to `[-eps,eps]` and the image to `[0,1]` each step — is in the answer. If instead MI-FGSM
fails to beat FGSM on VGG, my whole diagnosis (that the VGG survivors are curvature-underfit, not
genuinely safe) is wrong, and the headroom is architectural robustness no first-order iteration can
touch — which the next rung, an adaptive-step ensemble, would then have to settle.
