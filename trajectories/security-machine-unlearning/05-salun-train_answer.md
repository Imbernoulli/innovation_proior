SCRUB is the strongest rung so far, and its numbers say both that the ladder's main idea — bounded, reference-anchored forgetting — is right, and where even SCRUB still pays a structural tax. On the visible benchmarks it matched or edged Bad Teacher: resnet20-cifar10 `unlearn_score` 0.8115 (`retain_acc` 0.8787, `forget_acc` 0.005, MIA AUC 0.4393), vgg16bn-cifar100 0.6915 (`retain_acc` 0.5394, MIA AUC 0.4648). And decisively, where Bad Teacher collapsed on the hidden mobilenetv2-fmnist (0.1009), SCRUB recovered it — `retain_acc` 0.8834, `unlearn_score` 0.8198 — confirming that the single clean reference was the right stabilizer. But look at *how* it gets there: a min-max that pushes the student away from the teacher on $D_f$ and pulls it back on $D_r$ through the *same shared weights* every step. That interference is exactly why it needs the careful `msteps` schedule and the small learning rate to avoid oscillating, and why on the most fragile architecture it sits at the edge of the instability that sank Bad Teacher. The forget pressure still flows through the whole model and has to be *managed* away from the retained classes, and the MIA AUC in the high-0.43-to-0.46 range leaves forgetting headroom. The structural question SCRUB never asks is the one I want to push on: *which weights should the unlearning update even be allowed to touch?*

I propose **SalUn**: localize the edit to the forget-salient weights, and on those weights run a bounded random-labeling forget loss. The move starts from an analogy. In model explanation, *input saliency* is the gradient of the output with respect to the input, large where a pixel drives the decision. The exact analogue in weight space is the gradient of the *forgetting loss* with respect to each weight, $\partial \ell_f/\partial\theta$, evaluated at the original weights $\theta_o$: a weight with large $|\partial \ell_f/\partial\theta|$ is one whose movement most changes the forget-set behavior — a *forget-salient* weight — while a weight with near-zero forget gradient is mostly there for $D_r$. That is one backward pass on the forget set, no Hessian and no extra model, as cheap as localization can be. Threshold the gradient magnitude to get a binary mask

$$m_S = \mathbb{1}\big(|\nabla_\theta \ell_f(\theta_o)| \ge \gamma\big),$$

and choose $\gamma$ *relatively* — the median of the gradient-magnitude vector — so the mask keeps the top 50% most forget-salient weights and freezes the bottom half. The median is parameter-free and scale-free, and 50% sparsity is aggressive enough to localize without starving the salient set. The unlearned model is then $\theta_u = m_S \odot \theta + (1 - m_S)\odot\theta_o$: salient weights free, the rest pinned at their original values.

The premise that makes this worth doing is that the forget set's footprint is *concentrated*, not smeared uniformly. If $|\partial\ell_f/\partial\theta|$ were roughly constant over all parameters, a median threshold would keep an arbitrary half and buy nothing. But a class-forgetting gradient does not look like that. The forget loss is cross-entropy on class-0 images against the class-0 label, and its gradient is largest exactly where the network commits to *class 0 specifically* — the output-layer row that scores class 0, and the late features that fire on the class-0 concept — and small in the shared early layers that compute generic edges and textures, because those are needed equally by every class and the class-0 cross-entropy has little leverage on them. So the forget gradient is concentrated precisely on the weights I want to move and *not* on the shared-trunk weights I want to protect. The median threshold then does something principled: it keeps the class-0-committing weights (well above median) and freezes the generic-feature weights (below median), so editing only the salient half is close to editing only the forget-relevant subnetwork — and freezing the rest protects $D_r$ *by construction*, before I even choose a forgetting loss. That is precisely the leakage SCRUB had to schedule its way around.

Localization alone does not *bound* the forgetting — freezing half the weights does not turn an unbounded objective into a bounded one, and I already watched NegGrad+'s unbounded ascent wreck a model — so on the salient weights I run a bounded forget signal: **random labeling**. Assign each forget example a fresh uniformly-random label and minimize ordinary cross-entropy toward it. This is descent toward a finite target, not ascent away from one, so it has a genuine minimum and cannot run the weights off; and it pushes the forget set *off* its memorized class-0 answer toward an arbitrary class, landing near where a never-trained model would sit — decorrelated, not confidently inverted. That bounded, near-generalization target is the same property the ladder has chased since NegGrad+'s overshoot, now delivered by the simplest possible loss. I keep a true-label retain CE on $D_r$ run through the *same* mask, so wherever the random-label term would drag a shared salient weight off, the retain term pulls it back to a $D_r$-correct value. The combined per-step objective is $\mathrm{CE}(\text{forget}, \text{random\_labels}) + \mathrm{CE}(\text{retain}, \text{true\_labels})$, with its gradient masked before the optimizer step so only salient weights move.

The mask mechanics have to be exact, and the harness shapes them. The mask must be computed at the *original* weights $\theta_o$, before any unlearning step has moved them — weight salience for the forget set is a property of the trained model, not a half-unlearned one — so I capture it lazily on the very first call: forward the forget minibatch, backward the forget cross-entropy, read the per-parameter gradient magnitudes, take their global median as $\gamma$, build the 0/1 mask, then `zero_grad` so that mask-building gradient does not contaminate the first real update. The harness hands me one forget minibatch at a time, never all of $D_f$, so this is a stochastic estimate of the full-forget direction — but because I threshold *magnitudes at their median*, the mask only needs the weight *ranking* by salience to be roughly right, and the dominant class-0 weights sit far above the median for any minibatch, so the estimate is faithful. After capture, every step masks the *gradient*, not the forward pass or the loss: after `loss.backward()` I multiply each parameter's `.grad` by its mask entry (`p.grad.mul_(mask[n])`), zeroing the gradient on every frozen weight, then step. The forward must use the full model so the salient weights see the real representation; only the update is localized. The random labels are drawn fresh each step over *all* classes including possibly the true one — rather than forcing them to differ, since the constant re-draw makes the occasional true-label draw inconsequential and stops the model from re-memorizing a single permuted labeling — and `num_classes` is read once from the model's output width so the same fill works across cifar10 (10), cifar100 (100), and fmnist (10).

So the delta from SCRUB is structural, not just another loss. SCRUB edits all weights and *schedules* a min-max to keep the forget pressure from corroding $D_r$ through the shared trunk; SalUn *localizes* the edit to the forget-salient half of the weights so the leakage is cut off at the source, and on those weights runs a bounded random-label forget loss plus a masked retain loss — no min-max, no `msteps` schedule, no teacher to capture. The forgetting cannot explode (bounded random target) and it cannot spread (masked to forget-salient weights). I expect SalUn to match or beat SCRUB on all three benchmarks, with `retain_acc` held by the frozen half, `forget_acc` near zero from the random-label decorrelation, and the forget MIA AUC closer to the 0.5 never-saw-it point — including on the fragile mobilenetv2-fmnist, because localization removes the very shared-trunk interference that made that benchmark fragile in the first place. The fixed `Adam(lr=0.001)` is, as with SCRUB, fortunate: masking zeros the gradient on frozen weights, and with a small step their optimizer buffers stay at zero so they genuinely do not drift, while the salient weights take ordinary small steps.

```python
class UnlearningMethod:
    """SalUn: saliency-masked random-labeling unlearning."""

    def __init__(self):
        self.sparsity = 0.5     # keep the top (1 - sparsity) most forget-salient weights
        self.mask = None        # {param_name: 0/1 tensor}, built lazily at the original weights
        self.num_classes = None

    def _build_saliency_mask(self, model, forget_x, forget_y):
        # Weight saliency = magnitude of the forget-loss gradient at the ORIGINAL weights.
        model.zero_grad()
        loss = F.cross_entropy(model(forget_x), forget_y)
        loss.backward()
        grads = {n: p.grad.detach().abs() for n, p in model.named_parameters() if p.grad is not None}
        flat = torch.cat([g.flatten() for g in grads.values()])
        gamma = flat.quantile(self.sparsity)           # median when sparsity = 0.5
        self.mask = {n: (g >= gamma).float() for n, g in grads.items()}
        model.zero_grad()

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        retain_x, retain_y = retain_batch
        forget_x, forget_y = forget_batch
        if self.num_classes is None:
            self.num_classes = int(model(forget_x[:1]).shape[1])
        if self.mask is None:
            self._build_saliency_mask(model, forget_x, forget_y)   # at the original weights, first step

        # Bounded forget target: fresh uniform random labels on D_f.
        rand_y = torch.randint(0, self.num_classes, forget_y.shape, device=forget_y.device)
        forget_loss = F.cross_entropy(model(forget_x), rand_y)
        retain_loss = F.cross_entropy(model(retain_x), retain_y)
        loss = forget_loss + retain_loss

        optimizer.zero_grad()
        loss.backward()
        # Localize the edit: zero the gradient of every non-salient (frozen) weight.
        for n, p in model.named_parameters():
            if p.grad is not None and n in self.mask:
                p.grad.mul_(self.mask[n])
        optimizer.step()

        return {
            "loss": float(loss.item()),
            "forget_loss": float(forget_loss.item()),
            "retain_loss": float(retain_loss.item()),
        }
```
