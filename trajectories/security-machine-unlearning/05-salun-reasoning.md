SCRUB is the strongest rung so far, and reading its numbers tells me both that the ladder's main idea —
bounded, reference-anchored forgetting — is right, and where even SCRUB still pays a structural tax. On
the two visible benchmarks it matched or edged Bad Teacher: resnet20-cifar10 `unlearn_score` 0.8115
(`retain_acc` 0.8787, `forget_acc` 0.005, MIA AUC 0.4393), vgg16bn-cifar100 0.6915 (`retain_acc` 0.5394,
MIA AUC 0.4648). And decisively, where Bad Teacher *collapsed* on the hidden mobilenetv2-fmnist (0.1009),
SCRUB recovered it — `retain_acc` 0.8834, `unlearn_score` 0.8198 — confirming my diagnosis that removing
the noisy random teacher and using a single clean reference was the right stabilizer. Let me confirm the
scores decompose the way I expect: resnet20 is `(0.8787 + (1 − 0.005) + (1 − 0.4393))/3 = 2.4344/3 =
0.8115`, and the recovered fmnist is `(0.8834 + 1 + (1 − 0.424))/3 = 2.4594/3 = 0.8198`, both exactly the
reported numbers — so SCRUB genuinely lifted every benchmark above its passive floor and, on fmnist, above
even the passive 0.8185. So SCRUB is good.

But look at *how* it gets there, and at what it leaves on the table. It runs a min-max, pushing the student
away from the teacher on `D_f` and pulling it back on `D_r`, through the *same shared weights*, every step.
That push/pull interference is exactly why it needs the careful `msteps` schedule and the small learning
rate to avoid oscillating — and why, on the most fragile architecture, it sits at the edge of the same
instability that sank Bad Teacher, recovered but not by a comfortable margin. And the forget MIA AUC still
sits at 0.4393 / 0.4648 — below 0.5, so still short of the never-saw-it point from the wrong side, meaning
there is forgetting headroom SCRUB is not claiming: it has not driven the membership signal all the way to
0.5 on the visible benchmarks. So SCRUB spends its entire design budget *managing* a forget pressure that
flows through the whole model, and still leaves both a stability margin and a privacy margin unclaimed. The
structural question SCRUB does not ask is the one I want to push on: *which weights should the unlearning
update even be allowed to touch?* Every rung to here — finetune, NegGrad, Bad Teacher, SCRUB — edits all
the parameters and then spends its design budget containing the leakage through the shared trunk. What if I
cut the leakage off at the source by *localizing the edit*?

Before I pick a localization mechanism, let me walk the candidates, because "localize the edit" is a
family and I want the cheapest member that actually aligns with the forget/retain split. The crudest option
is to freeze by *architecture position* — say, only update the final classifier layer, or only the last
block — on the intuition that class identity lives late in the network. But that is a hand-guess about
where class-0 knowledge sits, and it is both too coarse and misplaced: too coarse because a whole layer
mixes forget-relevant and retain-relevant weights, and misplaced because on a depthwise-separable trunk
like mobilenetv2 the class-specific computation is distributed differently than on a dense convnet, so any
fixed layer choice that helps one benchmark could hurt another. A second option is a *second-order
importance* score — Fisher information, the diagonal of the Hessian of the loss — which is the principled
way to ask "how much does each weight matter." But estimating a Fisher diagonal needs squared gradients
accumulated over data and is exactly the expensive computation the whole cheap-unlearning premise is trying
to avoid, and it measures importance for *all* the data, not specifically for the forget behavior I want to
target. The third option, and the one that fits, is *first-order forget-gradient magnitude*: the gradient
of the forgetting loss with respect to each weight, evaluated at the original weights. That is one backward
pass on the forget set — as cheap as localization can possibly be — and, unlike Fisher-over-all-data, it is
scored *specifically by the forget task*, so it ranks weights by exactly the axis I care about.

Here is the move, made precise. In model explanation, *input saliency* is the gradient of the output with
respect to the input — large where a pixel drives the decision. The exact analogue in weight space is
`∂ℓ_f/∂θ`, the gradient of the *forgetting loss* with respect to each weight, evaluated at the original
weights: a weight with large `|∂ℓ_f/∂θ|` is one whose movement most changes the forget-set behaviour — a
*forget-salient* weight — while a weight with near-zero forget gradient is mostly there for `D_r`.
Threshold the gradient magnitude to get a binary mask `m_S = 𝟙(|∇_θ ℓ_f(θ_o)| ≥ γ)`, and choose `γ`
*relatively* — the median of the gradient-magnitude vector — so the mask keeps the top 50% most
forget-salient weights and freezes the bottom half. The median is parameter-free and scale-free (no
absolute gradient scale to calibrate across three different architectures and datasets), and 50% sparsity
is aggressive enough to localize without starving the salient set. Then the unlearned model is
`θ_u = m_S ⊙ θ + (1 − m_S) ⊙ θ_o`: the salient weights are free, the rest pinned at their original values.
If the forget set's footprint really is concentrated — the class-0 output rows, the late features specific
to class 0 — then freezing the other half of the weights protects `D_r` *by construction*, before I choose
any forgetting loss, which is precisely the leakage SCRUB had to schedule its way around.

Let me check the premise that makes this worth doing — that the forget set's footprint is actually
concentrated rather than smeared uniformly across the weights. If `|∂ℓ_f/∂θ|` were roughly constant over
all parameters, a median threshold would keep an arbitrary half and localization would buy nothing — the
mask would be noise and I would be back to editing an effectively random half of the model. But that is not
what a class-forgetting gradient looks like, and I can reason about its structure. The forget loss is
cross-entropy on class-0 images against the class-0 label; its gradient with respect to a weight is largest
exactly where the network commits to *class 0 specifically* — the output-layer row that scores class 0 (its
gradient is directly proportional to the class-0 prediction error on every forget image, so it is large and
coherent), and the late-layer features that fire on the class-0 concept — and small in the shared early
layers that merely compute generic edges and textures, because those are needed equally for every class and
the class-0 cross-entropy has little leverage on them (their gradient contributions from "make class 0
right" are small and partially cancel across the diverse forget images). So the forget gradient *is*
concentrated, and concentrated precisely on the weights I want to move and *not* on the shared-trunk
weights I want to protect. The median threshold then does something principled: it keeps the
class-0-committing weights (well above median) and freezes the generic-feature weights (below median). This
is the structural reason localization should help — the saliency ranking aligns with the forget/retain
split far better than "all weights" does, so editing only the salient half is close to editing only the
forget-relevant subnetwork. It is also why the median, not a tiny top-k, is the right cut: I want *all* the
forget-committing weights free, and they plausibly make up a substantial fraction (the full class-0 output
row plus its late-feature support), so a 50% keep is generous enough to contain them while still freezing
the entire generic trunk.

Localization alone does not bound the forgetting, though — freezing half the weights does not turn an
unbounded objective into a bounded one, and I already watched NegGrad's unbounded ascent wreck a model. So
on the salient weights I run a *bounded* forget signal: random labeling. Assign each forget example a fresh
uniformly-random label and minimize ordinary cross-entropy toward it. This is descent toward a finite
target, not ascent away from one — it has a genuine minimum, so it cannot run the weights off the way
NegGrad did — and it pushes the forget set *off* its memorized class-0 answer toward an arbitrary class,
landing the forget behaviour near where a never-trained model would sit: decorrelated, not confidently
inverted. That bounded, near-generalization target is the same property the whole ladder has been chasing
since NegGrad's overshoot, now delivered by the simplest possible loss — a plain cross-entropy, no teacher,
no min-max, no schedule. And I keep a true-label retain CE on `D_r`, run through the *same* mask, so
wherever the random-label term would drag a shared salient weight off, the retain term pulls it back to a
`D_r`-correct value. The combined per-step objective is `CE(forget, random_labels) + CE(retain,
true_labels)`, equal weight, and its gradient is masked before the optimizer step so only salient weights
move. It is worth noting *why* random labeling is a cleaner bounded target than SCRUB's push-away-from-
teacher: SCRUB's forget signal is defined relative to a frozen teacher and still flows through all weights,
so it needed the schedule to stay controlled; random-label CE is self-contained (its target is just a
label), and combined with the mask its reach is already cut to the salient subnetwork, so it needs no
scheduling at all — the two structural fixes, bounded target and localized reach, are orthogonal and I get
both at once.

The mask mechanics have to be exactly right, and the harness shapes them. The mask must be computed at the
*original* weights `θ_o`, before any unlearning step has moved them — weight salience for the forget set is
a property of the trained model, not a half-unlearned one, so if I recomputed it later it would drift with
the very edits it is supposed to guide. So I capture it lazily on the very first call: forward the forget
minibatch, backward the forget cross-entropy, read the per-parameter gradient magnitudes, concatenate them
into one flat vector, take its median as `γ`, build the 0/1 mask per parameter, then `zero_grad` so that
mask-building gradient does not contaminate the first real update. The harness hands me one forget
minibatch per step, never the full `D_f` at once, so this first-minibatch gradient is a stochastic estimate
of the full-forget direction — but because I threshold *magnitudes at their median*, the mask only needs
the weight *ranking* by salience to be roughly right, not the exact gradient values, and the dominant
class-0 weights (the output row especially) sit far above the median for any forget minibatch, so the
ranking is faithful even from a 128-image estimate. After capture, every step applies the mask the same
way: after `loss.backward()`, multiply each parameter's `.grad` by its mask entry — `p.grad.mul_(mask[n])` —
zeroing the gradient on every frozen weight, then `optimizer.step()`. Critically I mask the *gradient*, not
the forward pass or the loss: the forward must use the full model so the salient weights see the real
representation the frozen weights produce; only the *update* is localized. A subtlety with the fixed Adam:
masking zeros the gradient on frozen weights every step, so their Adam first/second-moment buffers stay at
zero and they genuinely do not drift — the optimizer cannot resurrect a frozen weight from stale momentum,
because there is no momentum to carry. The random labels are drawn fresh each step
(`torch.randint(0, num_classes, ...)` on device), so the forget target keeps shuffling and the model cannot
just re-memorize a single permuted labeling; I read `num_classes` once from the model's output width so the
same fill works across the cifar10 (10), cifar100 (100), and fmnist (10) benchmarks.

One detail I am deliberately keeping faithful to the standard recipe rather than "improving": random labels
are drawn uniformly over *all* classes including possibly the true class 0, not forced to differ. It is
tempting to exclude the true label to guarantee every draw is a "wrong" target, but that adds a special
case and, with the label re-drawn fresh every step over 20 epochs, the occasional true-label draw is
inconsequential — the expected fraction of true-label draws is `1/num_classes` (a tenth on cifar10/fmnist,
a hundredth on cifar100), and averaged over the run they contribute a negligible, non-systematic reinforcing
nudge that the constant re-shuffling washes out. Simpler and faithful wins over a guard that buys nothing.

Let me trace the first two calls to confirm the ordering does what I intend, because the mask is built
lazily and order-sensitivity is where this kind of code quietly breaks. On the very first call, `self.mask`
is `None`, so I read `num_classes` from a single forward, then build the mask: `model.zero_grad()`, forward
`forget_x`, `F.cross_entropy` to the *true* class-0 labels, `backward` — now every parameter carries
`∂ℓ_f/∂θ` at the original weights — collect `|grad|` per parameter, flatten, `quantile(0.5)` for `γ`, set
`mask[n] = (|grad| ≥ γ)`, and `model.zero_grad()` again so this diagnostic gradient is discarded. Only then
does the *real* update run: random labels drawn, `CE(forget, rand_y) + CE(retain, true_y)`, backward, mask
the grads, step. So the mask is genuinely computed at `θ_o` (the update has not moved the weights yet at the
moment of the backward that builds it), and the mask-building gradient never reaches the optimizer. On the
second and every later call, `self.mask` is not `None`, so the build is skipped and the same frozen mask is
reused — the ranking is fixed once, at the original weights, exactly as the premise requires. If I had
built the mask *after* a step, or reused the update gradient to threshold, the mask would reflect
half-unlearned weights and the localization guarantee would be gone; the trace confirms neither happens.

The equal weighting of the two cross-entropies is a choice I should justify rather than default into.
Unlike SCRUB, where I deliberately made the retain objective 99-to-1 CE-over-KL because the KL was only a
light anchor, here both terms are *true-label cross-entropies* of the same kind — one on `D_f` toward random
labels, one on `D_r` toward correct labels — so there is no reason to privilege one scale over the other,
and equal weight keeps the forget decorrelation and the retain preservation on the same footing. More
importantly, the mask has already done the separation work that a weight ratio would otherwise have to do:
the retain term and the forget term only contend on the *shared salient* weights (a class-0 output-row
weight is purely forget-side and sees no retain gradient; a generic-feature weight is frozen and sees no
update at all), and on those few genuinely-shared salient weights equal weighting lets the correct-label
retain gradient and the random-label forget gradient balance directly, with the retain gradient — coherent
across a large retain minibatch — reliably outvoting the deliberately-incoherent random-label gradient. So
the mask, not a tuned loss weight, is what protects retain accuracy, which is exactly why this method needs
no schedule and no weight sweep where SCRUB needed both.

This harness has no extra imports to add — `torch` and `F` are already provided at the module top — so the
fill is just the `UnlearningMethod` class. So the delta from SCRUB is structural, not just another loss.
SCRUB edits all weights and *schedules* a min-max to keep the forget pressure from corroding `D_r` through
the shared trunk; SalUn *localizes* the edit to the forget-salient half of the weights so the leakage is
cut off at the source, and on those weights runs a bounded random-label forget loss plus a masked retain
loss — no min-max, no `msteps` schedule, no teacher to capture, no temperature. The forgetting cannot
explode (bounded random target), and it cannot spread (masked to forget-salient weights); the two failure
axes of the whole ladder are closed by two independent, cheap mechanisms.

What I would validate against SCRUB's measured numbers. On the visible benchmarks I expect SalUn to *match
or beat* SCRUB's `unlearn_score` — resnet20-cifar10 at or above 0.8115, vgg16bn-cifar100 at or above
0.6915 — with `retain_acc` held at or above SCRUB's 0.8787 / 0.5394 because the frozen half of the weights
protects the retained classes more directly than SCRUB's scheduled retain pull, and `forget_acc` near zero
from the random-label decorrelation. The sharpest test is the membership signal, the headroom SCRUB left: because the random-label target lands the forget set near generalization behaviour while the mask keeps the
retained classes' weights in place, I expect the forget MIA AUC to come in at or below SCRUB's 0.4393 /
0.4648, closer to the 0.5 never-saw-it point. And on the fragile mobilenetv2-fmnist benchmark where SCRUB
recovered to 0.8198, I expect SalUn to *hold or improve* it without needing any schedule, because
localization removes the very shared-trunk interference that made that benchmark fragile in the first place —
if it instead regresses there, the lesson would be that the depthwise-separable architecture's
forget-salient weights are too entangled with `D_r` for a 50% mask to separate cleanly, and the next move
would be a per-layer or adaptive sparsity that allocates the keep-budget differently across depthwise and
pointwise weights. But the structural argument says localizing the edit is exactly the lever that should
carry SalUn past SCRUB on all three. The full scaffold module is in the answer.
