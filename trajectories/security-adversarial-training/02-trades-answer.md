**Problem (from step 1).** PGD-AT lifts robustness over the floor but its single CE term has no knob:
on CIFAR it pays clean accuracy down (C10 0.85→PGD-50 0.4585; C100 0.5828→0.2214, a 36-point gap) with
no way to dial the robustness/accuracy balance. I want the trade-off written into the objective.

**Key idea.** Decompose the robust error *exactly*: `R_adv = R_nat + R_bdy` — natural error plus the
mass of correctly-classified points crowding the decision boundary. `R_nat` is the accuracy term
(calibrated cross-entropy); `R_bdy` is bounded by "some perturbation disagrees with the clean
prediction," which relaxes to a KL between clean and perturbed outputs. Train on the sum with a weight:
`min_f E[CE(f(x), y) + beta * max_{x' in B(x,eps)} KL(p(x) || p(x'))]`.

**Why it works.** The boundary regularizer measures perturbed-vs-*clean-prediction*, not
perturbed-vs-label — it pushes the boundary away from the data while leaving the clean term to hold
accuracy, so `beta` is the dial PGD-AT lacked. A toy oscillating-label example proves the
accuracy-optimal and robustness-optimal classifiers are genuinely different models, so the knob is
mandatory; the upper bound is tight, so this is the right regularizer.

**Scaffold edit.** Same eval-mode-attack / train-mode-update structure, but the inner PGD ascends the
*KL* to the clean prediction (not the label CE), so it must start from a tiny random nudge
`x + 0.001*randn` — at `x' = x` the KL is its global minimum and the gradient is zero. The clean logits
are computed once and detached as the frozen KL target during the attack. KL uses `reduction='batchmean'`
(no separate `1/batch_size`); the clean branch is the `softmax` target, the adversarial branch the
`log_softmax` input.

**Hyperparameters.** `beta = 6.0` (robustness-heavy); `eps`, `alpha`, `attack_steps` passed in per
scenario; tiny-nudge start (forced by the KL geometry).

**What to watch.** MNIST PGD-50 should rise above 0.8932 with clean held near 0.98 (free slack); on
CIFAR, robustness should lift but `beta = 6.0` may trade away some clean accuracy (C10 clean below
0.8501, C100 below 0.5828). If per-architecture CIFAR-10 robustness does *not* beat PGD-AT's
0.4585 / 0.4194, the knob pushes every example equally — including the ones already nailed — and
reallocating effort toward the hard examples is the next move.

```python
class AdversarialTrainer:
    """TRADES (Zhang et al., 2019)."""

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.attack_steps = attack_steps
        self.num_classes = num_classes
        self.beta = 6.0  # TRADES regularization weight

    def train_step(self, images, labels, optimizer):
        self.model.train()

        # Clean forward pass (the anchor for the KL target).
        logits_clean = self.model(images)
        loss_clean = F.cross_entropy(logits_clean, labels)

        # Inner max: generate adversarial examples by maximizing KL(clean || adv).
        self.model.eval()
        adv_images = images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)

        for _ in range(self.attack_steps):
            adv_images.requires_grad_(True)
            logits_adv = self.model(adv_images)
            loss_kl = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_clean.detach(), dim=1),   # frozen clean anchor
                reduction='batchmean',
            )
            grad = torch.autograd.grad(loss_kl, adv_images)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()

        # Outer min: clean CE + beta * KL(clean || adv).
        self.model.train()
        logits_adv = self.model(adv_images)
        loss_kl = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_clean.detach(), dim=1),
            reduction='batchmean',
        )
        loss = loss_clean + self.beta * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            'loss': loss.item(),
            'loss_clean': loss_clean.item(),
            'loss_kl': loss_kl.item(),
        }
```
