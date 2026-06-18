Iterating paid off where the diagnosis said it would. MI-FGSM lifted the easy scenarios to the ceiling
— 0.981 / 0.984 on ResNet20-C10/C100, 0.999 on MobileNetV2-C10, 0.967 on MobileNetV2-C100 — confirming
that the samples FGSM left on those architectures were curvature-underfit, not safe: re-reading the
slope across forty steps reached them. And on VGG it did move the needle, 0.617 → 0.714 and
0.621 → 0.720, recovering a chunk of the underfit. But the VGG numbers are still in the low 0.7s, and
the gain there is smaller than the easy-scenario gains were dramatic. So one of two things is true:
either VGG's residual ~28% is architectural robustness that no first-order attack can touch, or the
*particular* iterative scheme I ran is leaving white-box loss on the table. Before I concede the
former, I want to interrogate the latter, because there is a specific reason to suspect the momentum I
added is, in this single-model setting, slightly *blunting* the maximizer rather than sharpening it.

Reread why I put momentum in. The justification was that greedy sign-ascent zig-zags into sharp local
maxima, and in the *transfer* setting those maxima are one model's private quirks that do not
generalize — so averaging the direction across steps keeps the component shared with other models and
discards the idiosyncratic part. That argument is about *generalizing the perturbation to a different
model*. But here I attack and am graded by the same architecture: there is no held-out model whose
boundary I need to align with. The thing I actually want is the strongest possible solution to the
*inner* maximization `max_{||δ||_inf ≤ eps} J(θ, x+δ, y)` on this one model — climb its own loss
surface as high as I can inside the box. And for that objective, an accumulated velocity is not
obviously helpful and can hurt. The velocity is a running sum of *past* normalized gradients, from
points the iterate has already left; near the high-loss boundary, where the surface bends, the
direction that maximizes the *current* loss is the current gradient's sign, and a stale momentum term
can point the step partly along a direction that was good several iterations ago but is now wrong,
overshooting past the local maximum or oscillating along the boundary instead of settling on it.
Momentum trades faithful local ascent for direction stability — a good trade when stability buys
transfer, a possibly-bad trade when the only goal is to maximize *this* loss.

So let me strip back to the cleanest strong inner maximizer and ask what it should be, deriving it
from the objective rather than from the momentum patch. I have a fixed classifier, white-box access so
I can read `∇_x L(θ, x, y)` for any loss, a clean image the model gets right, and a budget `eps`. I
want `δ` solving `max_δ L(θ, x+δ, y)` subject to `||δ||_inf ≤ eps` and `x+δ ∈ [0,1]^d`. That is a
constrained maximization of the loss over a little box around `x`. There is no closed form — `L` is a
deep network, wildly non-concave in `x` — so I will not solve it exactly; I will climb it as well as a
gradient method can. The reason to climb it *well*, rather than just get some misclassification, is
exactly the VGG number: a sloppy maximizer reports the model as more robust than it is, because the
samples it fails to flip are not safe, they are out of its reach. The VGG survivors are precisely the
samples where the maximizer's quality is being tested.

The cheapest move is to linearize: `L(x+δ) ≈ L(x) + g^T δ` with `g = ∇_x L`, maximized over the box at
`δ = eps·sign(g)` — the corner, gain `eps·||g||_1`. That is FGSM, and I already know it underfits the
curved surface. The honest correction is what FGSM and MI-FGSM both reached for: take a *small* step in
the sign direction, re-read the gradient where I land, and repeat — `x_{t+1} = x_t + alpha·sign(∇_x
L(θ, x_t, y))` — so I follow the curved surface instead of betting on one tangent plane. But the
instant I iterate, two things break the constraint: after a few steps the accumulated move can wander
outside the `eps`-box, and the iterate can leave `[0,1]`. So I must project back into the feasible set
after every step. This is what projected gradient methods are for, and I want to *know* the projection
is right, not just clamp and hope.

Work out the projection. The feasible set is the intersection of `||x' − x||_inf ≤ eps` (budget) and
`x' ∈ [0,1]^d` (valid image). For the `L_inf` ball, the Euclidean projection
`Π(z) = argmin_{||x'−x||_inf ≤ eps} ||x' − z||_2` decouples coordinate by coordinate, because both the
squared distance and the constraint are per-coordinate; minimizing `(x'_i − z_i)^2` subject to
`x_i − eps ≤ x'_i ≤ x_i + eps` gives `z_i` if it is already inside the interval, else the nearer
endpoint — which is exactly clamping `x'_i` to `[x_i − eps, x_i + eps]`. So element-wise clipping of
the perturbation to `[−eps, eps]` *is* the Euclidean projection onto the `L_inf` box, not a heuristic.
The pixel-box `[0,1]` is the same kind of object — another set of per-coordinate intervals — so its
projection is clamping to `[0,1]`. The intersection is the coordinate interval
`[max(x_i − eps, 0), min(x_i + eps, 1)]`, and the implementation reaches the same point by clipping the
perturbation to `[−eps, eps]`, adding it to `x`, then clipping the image to `[0,1]`. That is projected
gradient ascent under `L_inf`, and it is precisely the same dual-clamp I have been using in every rung
— now justified as the exact projection rather than carried along by analogy.

Two design levers remain: the step size with the step count, and where I start. For the step, this task
runs `steps = 40` with `alpha = eps/4`, no momentum — and now I can reason about why that is the right
inner-maximizer setting rather than MI-FGSM's `eps/10` with a velocity. The requirement is that from
any start inside the `eps`-ball the iterate can reach the opposite side and still have steps left to
move *along* the boundary once pinned there; the box diameter in any coordinate is `2·eps`, so
`steps·alpha` must comfortably exceed `2·eps`. With `alpha = eps/4` and 40 steps the total reach is
`40·eps/4 = 10·eps`, five times the diameter, so the iterate easily crosses the box from any interior
start and spends most of its steps refining its position on the high-loss boundary. The larger per-step
`eps/4` (versus MI-FGSM's `eps/10`) is affordable precisely *because* there is no momentum to overshoot
with: each step is a faithful local sign-ascent of the current loss, re-corrected the next step, so a
bigger step explores the box faster without a stale velocity carrying it past the maximum. More steps,
up to diminishing returns, keep correcting stale-gradient error rather than trusting the first plane.
This is the lever momentum was contesting, and dropping the velocity is what lets the step size do its
job cleanly.

For the start, I keep the random launch — and here the reasoning is sharper than "dodge the corrupted
gradient." The gradient evaluated at the exact data point `x` is suspect: the loss surface right next
to a data point has sharp curvature artifacts localized at `x`, so the steepest direction *at `x`* is a
poor guide to where the loss climbs a little further out. Worse, if I always launch from `x`, I commit
the whole trajectory to that one possibly-corrupted initial direction and climb into whatever single
local maximum sits in the basin containing `x` — and since the inner problem is non-concave, there are
many maxima scattered across the box, and I never see the others. Sampling each coordinate uniform in
`[−eps, eps]`, setting `x_0 = x + u`, and clipping into `[0,1]` jumps me off the non-smooth point
before the first gradient read, so the first gradient I trust is evaluated somewhere better-behaved,
and the trajectory can land in a different, possibly higher, basin. The random start costs no extra
gradient evaluation, and it is exactly what a strong first-order adversary should do.

The claim that this is the *strongest* first-order attack rests on whether random-start projected
ascent is finding the worst case or just *a* high-loss point. The inner maximization is non-concave; in
principle a hidden isolated maximum could be missed, badly underestimating vulnerability. The
diagnostic is to run the projected iteration from many random starts and look at the loss of the final
iterate each time: the useful pattern is that the loss climbs consistently from each start and
plateaus quickly, and across restarts the plateau values *concentrate* — many genuinely distinct
maxima (far apart, nearly orthogonal in the box) reaching almost the same loss. If that is the
landscape, then formal non-concavity is not the practical blocker it first appears: many local maxima
are about equally good, and a first-order adversary that already uses input-gradients repeatedly from
several starts has little evidence of a significantly better point to chase. Random restarts do not
prove no hidden higher maximum exists, but the concentration is the honest basis for calling this the
strongest *first-order* attack. The loss itself is a pluggable choice — cross-entropy is the simplest
default and the one the harness fill uses, with a logit-margin loss the drop-in when I want to push on
the decision boundary directly — but the iteration is the same. And Danskin's theorem closes the loop:
the negative loss-gradient at the inner maximizer is, in the tie-free exact-max case, a descent
direction for the saddle-point objective a robust-training defense would solve, so the same maximizer I
am building is exactly what such a defense would call in its inner loop — which is why "solve the inner
max well" is the principled target, not a heuristic.

So I drop the momentum, keep the random start, run 40 plain sign-ascent steps at `alpha = eps/4`, and
project the perturbation to `[−eps, eps]` and the image to `[0,1]` each step. The contract fill is in
the answer. The falsifiable expectation, against MI-FGSM's numbers: on the easy scenarios I expect a
dead heat — both attacks have already saturated ResNet20-C10/C100 and MobileNetV2, so PGD should land
in the same 0.97–0.99 band (it may even edge MobileNetV2-C10 down a hair if the bigger step is slightly
noisier there, since those are already essentially solved). The real test is VGG: if dropping the
stale velocity and taking faithful larger local steps genuinely sharpens the inner maximizer, VGG
should tick *above* MI-FGSM's 0.714 and 0.720 — modestly, into the low 0.72s — confirming that part of
the residual was a blunted maximizer, not pure architectural robustness. If PGD instead matches or
underperforms MI-FGSM on VGG, then the VGG headroom really is the architecture absorbing small `L_inf`
perturbations, and only changing the loss or moving to a worst-case ensemble of attacks — not a better
single-loss gradient climber — can settle it. Either way I will have separated "weak maximizer" from
"robust architecture," which is what the VGG number has been asking me to do since FGSM.
