Iterating paid off where the diagnosis said it would. MI-FGSM lifted the easy scenarios to the ceiling
— 0.981 / 0.984 on ResNet20-C10/C100, 0.999 on MobileNetV2-C10, 0.967 on MobileNetV2-C100 — confirming
that the samples FGSM left on those architectures were curvature-underfit, not safe: re-reading the
slope across forty steps reached them. And on VGG it did move the needle, 0.617 → 0.714 and
0.621 → 0.720, recovering a chunk of the underfit. Let me read the gains as differences, because the
pattern is sharper than "everything went up." The easy columns gained `0.981 − 0.825 = 0.156` and
`0.984 − 0.839 = 0.145` on ResNet, `0.999 − 0.878 = 0.121` and `0.967 − 0.825 = 0.142` on MobileNetV2;
VGG gained `0.714 − 0.617 = 0.097` and `0.720 − 0.621 = 0.099`. So iterating actually delivered a
*larger* absolute jump on the easy architectures than on VGG — the opposite of where the survivors
were densest — and the reason is a ceiling effect: the easy columns had `~0.16` of headroom and
iteration closed almost all of it, while VGG had `~0.38` of headroom and iteration closed barely a
quarter of it. After MI-FGSM the VGG survivors are `1 − 0.714 = 0.286` and `1 − 0.720 = 0.280`, still
`~28%`, and the mean has risen `0.768 → 0.894`. So the residual is now overwhelmingly two VGG columns,
and the question is exactly why iteration stalled *there* when it saturated everywhere else. One of two
things is true: either VGG's residual `~28%` is architectural robustness that no first-order attack can
touch, or the *particular* iterative scheme I ran is leaving white-box loss on the table. Before I concede the
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

There is a sharper way to see why the velocity can hurt *here* specifically. Momentum's variance
reduction is a genuine win when the gradient I read is a noisy estimate of a stable underlying
direction — average several noisy reads and the noise cancels. But in this white-box single-model
setting the input-gradient is not a noisy estimate of anything; it is the exact gradient of the exact
loss at the exact point, computed by autograd. There is no sampling noise to average away. What the
velocity actually averages, then, is the *genuine change* in the gradient as the surface bends — and
that change is signal, not noise. Smoothing it means deliberately lagging behind the true local
direction, which is precisely what I do not want when the only goal is to reach the highest loss inside
this box on this model. So the case for momentum, which was airtight when it bought transfer, is
actively backwards here: it is trading away faithful local ascent to suppress a noise term that does not
exist. That is the mechanism-level reason to strip it out and not merely a hunch.

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

Before working out the projection I should be honest that "drop momentum" is only one of a few moves I
could make against a stalled VGG, and check I am picking the right one by pricing the alternatives. I
could keep MI-FGSM's velocity and simply run more steps — but the momentum is a *systematic* bias, not
a variance I can average out: if a stale direction is pulling the step off the current-loss ascent,
more of the same steps compound the bias rather than cancel it, so 100 momentum steps would refine the
wrong direction more precisely. I could keep momentum but drop the random start — but the start is not
what I suspect; the velocity is, and the random launch is pure upside (it costs no gradient and only
diversifies the basin). I could switch the loss now — but I have no evidence yet that cross-entropy is
the bottleneck rather than the optimizer, and changing two things at once would tell me nothing about
which mattered; the disciplined move is to change exactly the component I have a mechanism-level
suspicion about, the accumulated velocity, and hold the loss and the restart fixed. So the controlled
experiment is: same CE loss, same random start, same 40-step budget, momentum removed and the step size
retuned to match. That isolates the one variable the VGG number is asking about.

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

Let me trace one coordinate to be sure the order of the two clamps matters and that I have it right.
Take a clean pixel `x_i = 0.98`, budget `eps = 0.00784`, and suppose the iterate has driven the raw
value to `z_i = 1.02` — over budget *and* out of the image. Clip the perturbation first:
`δ_i = z_i − x_i = 0.04`, clipped to `[−0.00784, 0.00784]` gives `0.00784`, so `x_i + δ_i = 0.98784`;
then clamp to `[0,1]` leaves `0.98784`, which is within budget and valid. Now try the wrong order —
clamp the image to `[0,1]` first, getting `1.0`, then enforce budget on `1.0 − 0.98 = 0.02 → 0.00784`,
again `0.98784`. Here they agree, but they do not always: at `x_i = 0.999`, budget would allow up to
`1.00684`, which the image clamp caps at `1.0`, and the feasible top is `min(x_i + eps, 1) = 1.0` — the
perturbation-then-image order gives `min(0.999 + 0.00784, ...) = 1.00684 → 1.0`, correct, while
image-first would also give `1.0`. The safe, always-correct order is the one the fill uses: clip `δ` to
the budget box, add, then clamp the sum into `[0,1]`, so the result lands in exactly the intersection
interval `[max(x_i − eps, 0), min(x_i + eps, 1)]`. Good — the projection is exact and the pixel-box
boundary is handled, not hoped through.

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
bigger step explores the box faster without a stale velocity carrying it past the maximum. Put the two
regimes side by side: MI-FGSM had to keep `alpha` small (`eps/10`) because the velocity already carries
momentum across steps, so a large per-step sign move on top of an accumulated push would sail past the
boundary optimum and the clip would waste the overshoot. With the velocity gone, the only thing moving
the iterate is the *current* gradient's sign, which the next step re-reads and corrects, so I can afford
`2.5×` the step (`eps/4` vs `eps/10`) and still land cleanly — the correction is now per-step instead of
fighting a running sum. The reach bookkeeping confirms both are safe: MI-FGSM reached `4·eps`, PGD
reaches `10·eps`, both exceed the `2·eps` diameter, but PGD gets there in `~8` steps and then spends
`~32` steps refining, a longer refinement phase than MI-FGSM's, which is exactly what a faithful local
maximizer wants. More steps, up to diminishing returns, keep correcting stale-gradient error rather than
trusting the first plane. This is the lever momentum was contesting, and dropping the velocity is what
lets the step size do its job cleanly.

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
gradient evaluation, and it is exactly what a strong first-order adversary should do. There is a subtle
compounding here with the dropped momentum: with a velocity, the random start's benefit would be partly
washed out, because the accumulated direction remembers the launch and drags the early trajectory back
toward it; without a velocity, each step reads the current gradient fresh, so the random start genuinely
places the *first trusted gradient* somewhere new and the trajectory is free to follow the local surface
from there. So the two design choices reinforce each other — dropping the velocity makes the random
start more effective, not just independently good — which is another way of seeing that the faithful
local maximizer is the coherent design and the momentum variant was fighting itself.

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
strongest *first-order* attack. That said, the fill runs a *single* random start, not a sweep of them,
and that is a deliberate budget choice worth defending. Each restart is another full 40-step trajectory,
so `R` restarts cost `40·R` forward/backward passes per batch; on an undefended model the concentration
argument says the plateau values barely differ across starts, so the second and third restarts almost
never find a sample the first one missed — the marginal ASR per extra restart is near zero here. One
start with 40 well-tuned steps at `eps/4` spends the budget where it pays: crossing the box once and
then refining for `~32` steps on the boundary, which flips the reachable samples, and the unreachable
ones are unreachable from most starts anyway. So a single restart is the right point on the
cost/coverage curve for these models; restarts would be the lever to pull against a defense that
actively scatters the loss landscape, not against these. The loss itself is a pluggable choice — cross-entropy is the simplest
default and the one the harness fill uses, with a logit-margin loss the drop-in when I want to push on
the decision boundary directly — but the iteration is the same. I keep cross-entropy at this rung on
purpose: I am running a *controlled* change against MI-FGSM, isolating the optimizer, and swapping the
loss simultaneously would confound which change moved VGG. The whole point of this rung is to answer one
clean question — is the stale velocity what stalled VGG? — so I change exactly the one component I have
a mechanism-level suspicion about and hold everything else, loss included, fixed. If faithful CE-PGD
lifts VGG above MI-FGSM, the velocity was blunting the maximizer; if it does not, the maximizer was
already tight and the residual lives somewhere I have not yet touched. Either verdict is only readable
because I held the loss constant, which is why cross-entropy stays exactly where it was even though the
interface would let me swap it.

The cost stays trivial, which is why I can be this thorough. PGD is 40 forward/backward pairs per batch,
identical to MI-FGSM, so `2400` network evaluations across the `60`-batch benchmark; dropping the
momentum buffer and its `L_1` normalization actually removes two elementwise ops per step, so PGD is if
anything marginally *cheaper* than the rung it replaces, for a strictly more faithful maximizer. There
is no budget reason not to run it. And Danskin's theorem closes the loop and tells
me *why* the inner max is the right object, not just a convenient one. Robust training solves a
saddle-point problem, `min_θ E[max_{||δ||_inf ≤ eps} L(θ, x+δ, y)]` — minimize the worst-case-in-box
loss. Danskin's theorem says that when the inner max is attained at a unique `δ*`, the gradient of the
outer objective in `θ` is simply the ordinary gradient of `L(θ, x+δ*, y)` evaluated *at that
maximizer*, as if `δ*` were held fixed — the dependence of `δ*` on `θ` contributes nothing to first
order (the envelope theorem). So the defender's descent direction is computed by first finding the
inner maximizer and then differentiating the loss there; the attacker's inner max and the defender's
training step are the *same* computation. That is the deep reason "solve the inner max well" is the
principled target and not a heuristic: an evaluation attack that under-solves the inner max reports a
number the defender's own training loop would never believe, because the defender is required to solve
that same max to train at all. My job on this rung is to be the honest inner maximizer.

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
perturbations — not a weakness of this particular gradient climber — and a fundamentally different kind
of attack, rather than one more retuned single-loss sign-ascent run, would be needed to settle it.
Either way I will have separated "weak maximizer" from "robust architecture," which is what the VGG
number has been asking me to do since FGSM.
