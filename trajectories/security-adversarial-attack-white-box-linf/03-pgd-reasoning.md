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

Reread why I put momentum in. The justification was transfer: greedy sign-ascent zig-zags into sharp
maxima that are one model's private quirks, and averaging the direction keeps the component shared with
other models. But here I attack and am graded by the same architecture — there is no held-out model
whose boundary I need to align with. What I actually want is the strongest solution to the *inner*
maximization `max_{||δ||_inf ≤ eps} J(θ, x+δ, y)` on this one model, and for that a velocity can hurt.
The velocity is a running sum of *past* gradients; near the high-loss boundary where the surface bends,
the direction that maximizes the current loss is the current gradient's sign, and a stale momentum term
can point the step partly along a direction that was good several iterations ago, overshooting or
oscillating instead of settling. Sharper still: momentum's variance reduction is a win when the gradient
is a noisy estimate of a stable direction, but the autograd input-gradient here is exact — there is no
sampling noise to average. What the velocity smooths is the *genuine change* in the gradient as the
surface bends, which is signal, not noise. So the case for momentum, airtight when it bought transfer,
is backwards here: it trades faithful local ascent to suppress a noise term that does not exist.

So I strip back to the cleanest inner maximizer. I want `δ` solving `max_δ L(θ, x+δ, y)` subject to
`||δ||_inf ≤ eps` and `x+δ ∈ [0,1]^d` — a constrained maximization over a box around `x`. `L` is a deep
network, wildly non-concave, so there is no closed form; I climb it as well as a gradient method can,
and climbing it *well* is exactly the point, because a sloppy maximizer reports the model as more robust
than it is — the samples it fails to flip are out of its reach, not safe. Linearizing gives FGSM's
corner `δ = eps·sign(g)`, which underfits the curve; the correction is the same small-step re-read-and-
repeat FGSM and MI-FGSM already reached for, `x_{t+1} = x_t + alpha·sign(∇_x L(θ, x_t, y))`, projected
back into the feasible set each step since iterating can wander outside the `eps`-box and out of
`[0,1]`. This is a controlled experiment against MI-FGSM: I change exactly the component I suspect — the
velocity — and hold the CE loss, the random start, and the 40-step budget fixed, retuning only the step
size, so the VGG number reads on the one variable. (Keeping momentum and running more steps would only
refine a systematic bias more precisely; the loss I leave alone because I have no evidence yet it is the
bottleneck, and changing two things would tell me nothing.)

The projection is the same dual-clamp every step so far has used, now justified as exact rather than carried
along by analogy. The feasible set intersects `||x' − x||_inf ≤ eps` and `x' ∈ [0,1]^d`, both
per-coordinate boxes, so the Euclidean projection decouples: minimizing `(x'_i − z_i)^2` on
`[x_i − eps, x_i + eps]` gives `z_i` clamped into that interval, i.e. element-wise clipping of the
perturbation to `[−eps, eps]` *is* the projection onto the `L_inf` ball, and clamping to `[0,1]` handles
the image box. The intersection is `[max(x_i − eps, 0), min(x_i + eps, 1)]`, reached by clipping the
perturbation to `[−eps, eps]`, adding, then clamping the image to `[0,1]`.

Two levers remain: step size and start. I run `steps = 40`, `alpha = eps/4`. The requirement is that
from any interior start the iterate can cross the box (diameter `2·eps`) and still have steps to refine
on the boundary; `40·eps/4 = 10·eps` of reach is five diameters, so it crosses in ~8 steps and refines
for ~32. The larger per-step `eps/4` (versus MI-FGSM's `eps/10`) is affordable precisely because there
is no velocity to overshoot with — each step is a faithful local sign-ascent re-corrected the next step,
where MI-FGSM had to keep `alpha` small so a big step on top of an accumulated push would not sail past
the optimum. Dropping the velocity is what lets the step size do its job cleanly, and buys a longer
refinement phase than MI-FGSM had.

I keep the random launch. The gradient at the exact data point `x` is suspect — the loss surface right
next to it has sharp curvature artifacts, so the steepest direction there is a poor guide further out —
and always launching from `x` commits the whole trajectory to one possibly-corrupted basin of a
non-concave problem with many maxima. Sampling `x_0 = x + u`, `u` uniform in `[−eps, eps]`, clipped into
`[0,1]`, places the first trusted gradient somewhere better-behaved and lets the trajectory land in a
different, possibly higher basin, at no extra gradient cost. This compounds with the dropped velocity:
a momentum term would remember the launch and drag the early trajectory back toward it, whereas reading
the current gradient fresh lets the random start genuinely place the first step somewhere new.

Whether this is the *strongest* first-order attack rests on whether random-start projected ascent finds
the worst case or just *a* high-loss point. The inner max is non-concave, so a hidden isolated maximum
could in principle be missed. The empirical answer is that running from many starts, the final losses
*concentrate* — genuinely distinct maxima reaching almost the same loss — so formal non-concavity is not
the practical blocker it looks like, and a gradient adversary has little evidence of a significantly
better point to chase. That concentration is the honest basis for calling this the strongest first-order
attack, and it is why the fill runs a *single* start: on an undefended model the plateau barely differs
across starts, so extra restarts (each a full 40-step trajectory) almost never flip a sample the first
missed. Restarts would be the lever against a defense that scatters the loss landscape, not against
these. The loss stays cross-entropy (a margin loss is the drop-in) because this is a controlled
change: swapping the loss and the optimizer at once would confound which moved VGG, so I hold everything
but the velocity fixed. If faithful CE-PGD lifts VGG, the velocity was blunting the maximizer; if not,
the residual lives somewhere I have not touched — a verdict readable only because the loss was held.

The cost is trivial: 40 forward/backward pairs per batch as before, `2400` evaluations across the
benchmark, and dropping the momentum buffer makes it marginally cheaper than the method it replaces, for
a strictly more faithful maximizer. Danskin's theorem tells me *why* the inner max is the right object,
not just a convenient one. Robust training solves the saddle-point problem
`min_θ E[max_{||δ||_inf ≤ eps} L(θ, x+δ, y)]`; when the inner max is attained at a unique `δ*`, the
gradient of the outer objective in `θ` is just the ordinary gradient of `L(θ, x+δ*, y)` at that
maximizer, the dependence of `δ*` on `θ` contributing nothing to first order (the envelope theorem). So
the attacker's inner max and the defender's training step are the *same* computation, and an evaluation
attack that under-solves it reports a robustness number the defender's own training loop would never
believe. My job here is to be the honest inner maximizer.

So I drop the momentum, keep the random start, run 40 plain sign-ascent steps at `alpha = eps/4`, and
project the perturbation to `[−eps, eps]` and the image to `[0,1]` each step. The contract fill is in
the answer. The falsifiable expectation, against MI-FGSM's numbers: on the already-saturated scenarios
(ResNet20-C10/C100, MobileNetV2) a dead heat in the same high band, since both attacks have solved
them. The real test is VGG: if the faithful larger-step maximizer genuinely sharpens, VGG should tick
*above* MI-FGSM's 0.714 and 0.720 — but only modestly, since a better single-loss climber recovers
the blunted-maximizer part of the residual, not whatever part is architectural. If it instead matches
or underperforms MI-FGSM on VGG, the headroom really is the architecture absorbing small `L_inf`
perturbations, and a fundamentally different attack rather than one more retuned sign-ascent run would
be needed. Either way I separate "weak maximizer" from "robust architecture," which is what the VGG
number has been asking since FGSM.
