SCRUB is the strongest rung so far, and reading its numbers tells me both that the ladder's main idea —
bounded, reference-anchored forgetting — is right, and where even SCRUB still pays a structural tax. On
the two visible benchmarks it matched or edged Bad Teacher: resnet20-cifar10 `unlearn_score` 0.8115
(`retain_acc` 0.8787, `forget_acc` 0.005, MIA AUC 0.4393), vgg16bn-cifar100 0.6915 (`retain_acc` 0.5394,
MIA AUC 0.4648). And decisively, where Bad Teacher *collapsed* on the hidden mobilenetv2-fmnist (0.1009),
SCRUB recovered it — `retain_acc` 0.8834, `unlearn_score` 0.8198 — confirming my diagnosis that removing
the noisy random teacher and using a single clean reference was the right stabilizer. So SCRUB is good. But
look at *how* it gets there: it runs a min-max, pushing the student away from the teacher on `D_f` and
pulling it back on `D_r`, through the *same shared weights*, every step. That push/pull interference is
exactly why it needs the careful `msteps` schedule and the small learning rate to avoid oscillating — and
why, on the most fragile architecture, it sits at the edge of the same instability that sank Bad Teacher.
The forget pressure still flows through the whole model and has to be *managed* away from the retained
classes. The retain MIA AUC also still sits in the high 0.43-0.46 range — not all the way to 0.5 — so there
is forgetting headroom left too. The structural question SCRUB does not ask is the one I want to push on:
*which weights should the unlearning update even be allowed to touch?* Every rung to here — finetune,
NegGrad, Bad Teacher, SCRUB — edits all the parameters and then spends its design budget containing the
leakage through the shared trunk. What if I cut the leakage off at the source by *localizing the edit*?

Here is the move. In model explanation, *input saliency* is the gradient of the output with respect to the
input — large where a pixel drives the decision. The exact analogue in weight space is the gradient of the
*forgetting loss* with respect to each weight, `∂ℓ_f/∂θ`, evaluated at the original weights: a weight with
large `|∂ℓ_f/∂θ|` is one whose movement most changes the forget-set behaviour — a *forget-salient* weight —
while a weight with near-zero forget gradient is mostly there for `D_r`. That is one backward pass on the
forget set, no Hessian, no extra model — as cheap as the localization can possibly be, which matters
because the whole point is to avoid expensive computation. Threshold the gradient magnitude to get a binary
mask `m_S = 𝟙(|∇_θ ℓ_f(θ_o)| ≥ γ)`, and choose `γ` *relatively* — the median of the gradient-magnitude
vector — so the mask keeps the top 50% most forget-salient weights and freezes the bottom half. The median
is parameter-free and scale-free; 50% sparsity is aggressive enough to localize without starving the
salient set. Then the unlearned model is `θ_u = m_S ⊙ θ + (1 − m_S) ⊙ θ_o`: the salient weights are free,
the rest pinned at their original values. If the forget set's footprint really is concentrated — the class-0
output rows, the late features specific to class 0 — then freezing the other half of the weights protects
`D_r` *by construction*, before I choose any forgetting loss, which is precisely the leakage SCRUB had to
schedule its way around.

Let me check the premise that makes this worth doing — that the forget set's footprint is actually
concentrated rather than smeared uniformly across the weights. If `|∂ℓ_f/∂θ|` were roughly constant over
all parameters, a median threshold would keep an arbitrary half and localization would buy nothing. But
that is not what a class-forgetting gradient looks like. The forget loss is cross-entropy on class-0
images against the class-0 label; its gradient is largest exactly where the network commits to *class 0
specifically* — the output-layer row that scores class 0, and the late-layer features that fire on the
class-0 concept — and small in the shared early layers that merely compute generic edges and textures,
because those are needed equally for every class and the class-0 cross-entropy has little leverage on
them. So the forget gradient *is* concentrated, and concentrated precisely on the weights I want to move
and *not* on the shared-trunk weights I want to protect. The median threshold then does something
principled: it keeps the class-0-committing weights (well above median) and freezes the generic-feature
weights (below median). This is the structural reason localization should help — the saliency ranking
aligns with the forget/retain split far better than "all weights" does, so editing only the salient half
is close to editing only the forget-relevant subnetwork.

Localization alone does not bound the forgetting, though — freezing half the weights does not turn an
unbounded objective into a bounded one, and I already watched NegGrad's unbounded ascent wreck a model. So
on the salient weights I run a *bounded* forget signal: random labeling. Assign each forget example a fresh
uniformly-random label and minimize ordinary cross-entropy toward it. This is descent toward a finite
target, not ascent away from one — it has a genuine minimum, so it cannot run the weights off the way
NegGrad did — and it pushes the forget set *off* its memorized class-0 answer toward an arbitrary class,
landing the forget behaviour near where a never-trained model would sit: decorrelated, not confidently
inverted. That bounded, near-generalization target is the same property the whole ladder has been chasing
since NegGrad's overshoot, now delivered by the simplest possible loss. And I keep a true-label retain CE on
`D_r`, run through the *same* mask, so wherever the random-label term would drag a shared salient weight off,
the retain term pulls it back to a `D_r`-correct value. The combined per-step objective is
`CE(forget, random_labels) + CE(retain, true_labels)`, and its gradient is masked before the optimizer step
so only salient weights move.

The mask mechanics have to be exactly right, and the harness shapes them. The mask must be computed at the
*original* weights `θ_o`, before any unlearning step has moved them — weight salience for the forget set is
a property of the trained model, not a half-unlearned one. So I capture it lazily on the very first call:
forward the forget minibatch, backward the forget cross-entropy, read the per-parameter gradient
magnitudes, take their global median as `γ`, build the 0/1 mask, then `zero_grad` so that mask-building
gradient does not contaminate the first real update. The harness hands me one forget minibatch per step,
never the full `D_f` at once, so this first-minibatch gradient is a stochastic estimate of the full-forget
direction — but because I threshold *magnitudes at their median*, the mask only needs the weight *ranking*
by salience to be roughly right, and the dominant class-0 weights sit far above the median for any forget
minibatch, so the estimate is faithful. After capture, every step applies the mask the same way: after
`loss.backward()`, multiply each parameter's `.grad` by its mask entry — `p.grad.mul_(mask[n])` — zeroing
the gradient on every frozen weight, then `optimizer.step()`. Critically I mask the *gradient*, not the
forward pass or the loss: the forward must use the full model so the salient weights see the real
representation; only the update is localized. The random labels are drawn fresh each step
(`torch.randint(0, num_classes, ...)` on device), so the forget target keeps shuffling and the model cannot
just re-memorize a single permuted labeling. I read `num_classes` once from the model's output width so the
same fill works across the cifar10 (10), cifar100 (100), and fmnist (10) benchmarks.

This harness has no extra imports to add — `torch` and `F` are already provided at the module top — so the
fill is just the `UnlearningMethod` class. The fixed `Adam(lr=0.001)` is, as with SCRUB, fortunate: masking
zeros the gradient on frozen weights, and with a small Adam step the frozen weights' optimizer buffers stay
at zero so they genuinely do not drift, while the salient weights take ordinary small steps. There is one
detail I am deliberately keeping faithful to the reference rather than "improving": random labels are drawn
uniformly over *all* classes including possibly the true one, not forced to differ — the reference's
CIFAR-10 recipe does exactly this, and the constant re-draw makes the occasional true-label draw
inconsequential.

So the delta from SCRUB is structural, not just another loss. SCRUB edits all weights and *schedules* a
min-max to keep the forget pressure from corroding `D_r` through the shared trunk; SalUn *localizes* the
edit to the forget-salient half of the weights so the leakage is cut off at the source, and on those
weights runs a bounded random-label forget loss plus a masked retain loss — no min-max, no `msteps`
schedule, no teacher to capture. The forgetting cannot explode (bounded random target), and it cannot
spread (masked to forget-salient weights).

What I would validate against SCRUB's measured numbers. On the visible benchmarks I expect SalUn to *match
or beat* SCRUB's `unlearn_score` — resnet20-cifar10 at or above 0.8115, vgg16bn-cifar100 at or above
0.6915 — with `retain_acc` held at or above SCRUB's 0.8787 / 0.5394 because the frozen half of the weights
protects the retained classes more directly than SCRUB's scheduled retain pull, and `forget_acc` near zero
from the random-label decorrelation. The sharpest test is the membership signal: because the random-label
target lands the forget set near generalization behaviour while the mask keeps the retained classes' weights
in place, I expect the forget MIA AUC to come in at or below SCRUB's 0.4393 / 0.4648, closer to the 0.5
never-saw-it point — that is the forgetting headroom SCRUB left. And on the fragile mobilenetv2-fmnist
benchmark where SCRUB recovered to 0.8198, I expect SalUn to *hold or improve* it without needing any
schedule, because localization removes the very shared-trunk interference that made that benchmark fragile
in the first place — if it instead regresses there, the lesson would be that the depthwise-separable
architecture's forget-salient weights are too entangled with `D_r` for a 50% mask to separate, and the next
move would be a per-layer or adaptive sparsity. But the structural argument says localizing the edit is
exactly the lever that should carry SalUn past SCRUB on all three. The full scaffold module is in the
answer.
