SCRUB is the strongest rung so far, and its numbers tell me both that the ladder's main idea — bounded,
reference-anchored forgetting — is right, and where even SCRUB still pays a structural tax. On the two
visible benchmarks it matched or edged Bad Teacher: resnet20-cifar10 `unlearn_score` 0.8115 (`retain_acc`
0.8787, `forget_acc` 0.005, MIA 0.4393), vgg16bn-cifar100 0.6915 (`retain_acc` 0.5394, MIA 0.4648). And
decisively, where Bad Teacher *collapsed* on the hidden mobilenetv2-fmnist (0.1009), SCRUB recovered it —
`retain_acc` 0.8834, `unlearn_score` 0.8198 — confirming that removing the noisy random teacher and using a
single clean reference was the right stabilizer. Every benchmark landed above its passive floor, fmnist above
even the passive 0.8185. So SCRUB is good.

But look at *how* it gets there, and what it leaves on the table. It runs a min-max, pushing the student away
from the teacher on `D_f` and pulling it back on `D_r` through the *same shared weights* every step. That
push/pull interference is exactly why it needs the careful `msteps` schedule and the small learning rate to
avoid oscillating — and why, on the most fragile architecture, it sits at the edge of the same instability
that sank Bad Teacher, recovered but not by a comfortable margin. And the forget MIA AUC still sits at
0.4393 / 0.4648 — below 0.5, short of the never-saw-it point from the wrong side, meaning forgetting headroom
SCRUB has not claimed. So SCRUB spends its entire design budget *managing* a forget pressure that flows
through the whole model, and still leaves both a stability margin and a privacy margin unclaimed. Every rung
to here — finetune, NegGrad, Bad Teacher, SCRUB — edits all the parameters and then spends its budget
containing the leakage through the shared trunk. The structural question none of them asks: *which weights
should the unlearning update even be allowed to touch?* What if I cut the leakage at the source by
*localizing the edit*?

"Localize the edit" is a family, and I want the cheapest member that actually aligns with the forget/retain
split. Freezing by *architecture position* — update only the final classifier, or the last block — is a
hand-guess about where class-0 knowledge sits: too coarse, because a whole layer mixes forget-relevant and
retain-relevant weights, and misplaced, because on a depthwise-separable trunk the class-specific computation
is distributed differently than on a dense convnet, so any fixed layer choice that helps one benchmark hurts
another. A *second-order importance* score (Fisher diagonal) is the principled way to ask "how much does each
weight matter," but it needs squared gradients accumulated over data — the expensive computation the whole
cheap-unlearning premise avoids — and it measures importance for *all* the data, not for the forget behavior
I want to target. The fit is *first-order forget-gradient magnitude*: the gradient of the forgetting loss with
respect to each weight at the original weights. That is one backward pass on the forget set — as cheap as
localization gets — and, unlike Fisher-over-all-data, scored *specifically* by the forget task.

Made precise: the weight-space analogue of input saliency is `∂ℓ_f/∂θ`, evaluated at the original weights.
A weight with large `|∂ℓ_f/∂θ|` is one whose movement most changes forget-set behavior — *forget-salient* —
while a near-zero forget gradient means the weight is mostly there for `D_r`. Threshold the magnitude,
`m_S = 𝟙(|∇_θ ℓ_f(θ_o)| ≥ γ)`, with `γ` the *median* of the gradient-magnitude vector, keeping the top 50%
most forget-salient weights and freezing the rest. The median is parameter-free and scale-free — no absolute
gradient scale to calibrate across three architectures and datasets — and 50% sparsity is aggressive enough
to localize without starving the salient set. The unlearned model is `θ_u = m_S ⊙ θ + (1 − m_S) ⊙ θ_o`: the
salient weights free, the rest pinned. If the forget footprint is concentrated, freezing the other half
protects `D_r` *by construction*, before I choose any forgetting loss — exactly the leakage SCRUB had to
schedule around.

The whole move rests on that footprint being concentrated rather than smeared. If `|∂ℓ_f/∂θ|` were roughly
constant over all parameters, a median threshold would keep an arbitrary half and localization would buy
nothing. But a class-forgetting gradient is not flat. The forget loss is cross-entropy on class-0 images
against the class-0 label; its gradient is largest exactly where the network commits to *class 0
specifically* — the output-layer row scoring class 0 (its gradient is proportional to the class-0 prediction
error on every forget image, so large and coherent), and the late-layer features that fire on the class-0
concept — and small in the shared early layers computing generic edges and textures, where the class-0
cross-entropy has little leverage and its contributions partially cancel across diverse forget images. So the
forget gradient *is* concentrated, and precisely on the weights I want to move and not on the shared-trunk
weights I want to protect. The median then keeps the class-0-committing weights and freezes the generic-
feature ones — and the median, not a tiny top-k, is right because I want *all* the forget-committing weights
free (the full class-0 output row plus its late-feature support, plausibly a substantial fraction), generous
enough to contain them while still freezing the entire generic trunk.

Localization does not bound the forgetting, though — freezing half the weights does not turn an unbounded
objective into a bounded one, and I watched NegGrad's unbounded ascent wreck a model. So on the salient
weights I run a *bounded* forget signal: random labeling. Assign each forget example a fresh uniformly-random
label and minimize cross-entropy toward it — descent toward a finite target, not ascent away from one, with a
genuine minimum, so it cannot run the weights off — pushing the forget set off its memorized class-0 answer
toward an arbitrary class, landing near where a never-trained model sits: decorrelated, not confidently
inverted. That bounded near-generalization target is the property the ladder has chased since NegGrad's
overshoot, now delivered by the simplest possible loss — a plain cross-entropy, no teacher, no min-max, no
schedule. I keep a true-label retain CE on `D_r` through the *same* mask, so wherever the random-label term
drags a shared salient weight off, the retain term pulls it back. The per-step objective is
`CE(forget, random_labels) + CE(retain, true_labels)`, equal weight, gradient masked before the step so only
salient weights move. The two structural fixes are orthogonal — random-label CE is bounded, and the mask cuts
its reach to the salient subnetwork — so unlike SCRUB's teacher-referenced signal flowing through all weights,
this needs no scheduling at all.

The mask mechanics have to be exactly right, and the harness shapes them. The mask must be computed at the
*original* weights `θ_o`, before any step moves them — weight salience for the forget set is a property of
the trained model, not a half-unlearned one, so recomputing it later would drift with the very edits it is
meant to guide. So I capture it lazily on the first call: forward the forget minibatch, backward the forget
cross-entropy, read per-parameter gradient magnitudes, flatten, take the median as `γ`, build the 0/1 mask,
then `zero_grad` so the mask-building gradient does not contaminate the first update. The harness hands one
forget minibatch per step, so this is a stochastic estimate of the full-forget direction — but because I
threshold *magnitudes at their median*, the mask only needs the weight *ranking* to be roughly right, and the
dominant class-0 weights (the output row especially) sit far above the median for any minibatch, so the
ranking is faithful even from 128 images. Every step then, after `loss.backward()`, multiplies each
parameter's `.grad` by its mask entry — `p.grad.mul_(mask[n])` — and steps. I mask the *gradient*, not the
forward pass: the forward uses the full model so the salient weights see the real representation the frozen
weights produce; only the update is localized. A subtlety with the fixed Adam: masking zeros the gradient on
frozen weights every step, so their first/second-moment buffers stay at zero and they genuinely do not
drift — the optimizer cannot resurrect a frozen weight from stale momentum, because there is none. Random
labels are drawn fresh each step (`torch.randint(0, num_classes, ...)` on device) so the model cannot
re-memorize a single permuted labeling, and `num_classes` is read once from the model's output width so the
same fill works across cifar10 (10), cifar100 (100), and fmnist (10).

I keep random labels uniform over *all* classes, including possibly the true class 0, rather than forcing
them to differ — faithful to the standard recipe over a guard that buys nothing. Excluding the true label
adds a special case, and with labels re-drawn fresh every step over 20 epochs the occasional true-label draw
(expected `1/num_classes`) is a negligible, non-systematic nudge the constant reshuffling washes out. The
ordering is where lazy mask-building could quietly break: on the first call the mask is built at `θ_o` (the
update has not moved the weights yet at the moment of the backward that builds it) and its gradient is
discarded by `zero_grad` before the real update runs; on every later call the frozen mask is reused. If I had
built the mask after a step or reused the update gradient to threshold, it would reflect half-unlearned
weights and the localization guarantee would be gone.

Equal weighting of the two cross-entropies is deliberate, not a default. Unlike SCRUB, where I made retain
99-to-1 CE-over-KL because the KL was a light anchor, here both terms are true-label cross-entropies of the
same kind, so there is no reason to privilege one scale. More importantly, the mask has already done the
separation a weight ratio would otherwise do: the two terms only contend on the *shared salient* weights (a
class-0 output-row weight is purely forget-side and sees no retain gradient; a generic-feature weight is
frozen and sees no update), and on those genuinely-shared salient weights equal weighting lets the coherent
retain gradient reliably outvote the deliberately-incoherent random-label gradient. So the mask, not a tuned
loss weight, protects retain accuracy — which is why this method needs neither schedule nor weight sweep where
SCRUB needed both.

So the delta from SCRUB is structural, not just another loss: SCRUB edits all weights and *schedules* a
min-max to keep forget pressure from corroding `D_r` through the shared trunk; SalUn *localizes* the edit to
the forget-salient half of the weights so the leakage is cut at the source, and on those weights runs a
bounded random-label forget loss plus a masked retain loss — no min-max, no `msteps`, no teacher, no
temperature. The forgetting cannot explode (bounded random target) and cannot spread (masked to
forget-salient weights); the ladder's two failure axes are closed by two independent, cheap mechanisms.
Validating against SCRUB's numbers: on the visible benchmarks I expect SalUn to match or beat SCRUB —
resnet20 at or above 0.8115, vgg16bn at or above 0.6915 — with `retain_acc` at or above 0.8787 / 0.5394
because the frozen half protects the retained classes more directly than SCRUB's scheduled pull, and
`forget_acc` near zero from the decorrelation. The sharpest test is the membership signal SCRUB left
unclaimed: because the random-label target lands the forget set near generalization behavior while the mask
holds the retained weights in place, I expect the forget MIA at or below 0.4393 / 0.4648, closer to 0.5. And
on the fragile mobilenetv2-fmnist where SCRUB recovered to 0.8198, I expect SalUn to hold or improve it
without any schedule, because localization removes the shared-trunk interference that made that benchmark
fragile — if it instead regresses there, the lesson would be that the depthwise-separable architecture's
forget-salient weights are too entangled with `D_r` for a flat 50% mask to separate, and the next move would
be a per-layer or adaptive sparsity allocating the keep-budget differently across depthwise and pointwise
weights. The full scaffold module is in the answer.
