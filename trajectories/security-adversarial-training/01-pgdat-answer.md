**Problem.** The scaffold default is standard training: plain cross-entropy on clean images, which
aces the clean test set and collapses under any `L_inf` perturbation. I need a defense with a precise
guarantee, not another gradient-masking patch that one untried attack walks past.

**Key idea.** Write robustness as the saddle point
`min_theta E[max_{||delta||_inf <= eps} L(theta, x+delta, y)]` — attack is the inner max, defense the
outer min. Solve the inner non-concave max with random-start `L_inf` PGD (the strongest first-order
adversary; its loss values concentrate across restarts, so it is a reliable stand-in for the whole
class of gradient attacks). Solve the outer min by Danskin: freeze the PGD-found point and take one
ordinary cross-entropy SGD step on it — the `d delta*/d theta` term drops out at the active maximizer.

**Why it works.** Driving `rho(theta)` small means the loss is small for *every* `delta` in the ball,
so by construction no adversarial example survives within the budget — the guarantee bolt-on defenses
could never make. Multi-step PGD (not one-step FGSM) matters because the loss is not linear out to
`eps`, so a model trained against the linearized adversary falls to an iterative one.

**Step-1 edit.** The whole training loop, schedule, and architectures are the fixed substrate; the
only thing the baseline edits is `AdversarialTrainer`. Eval mode during the attack (freeze BN/dropout),
random uniform start, `attack_steps` of signed-gradient ascent on the true-label cross-entropy with
per-step projection back into the `eps`-box and `[0,1]`, then train mode and one SGD step on the
adversarial batch. It is the floor of adversarial methods.

**Hyperparameters.** `eps`, `alpha`, `attack_steps` are passed in per scenario (`eps = 0.3` on MNIST,
`8/255` on CIFAR; `alpha < eps` with `attack_steps * alpha > eps`). No extra regularizer weight — the
single CE term is exactly the weakness the next rung attacks.

**What to watch.** A large jump in PGD-50 robustness over the floor on every scenario, but clean
accuracy paid down and robust accuracy left well below clean — a gap a single-term objective cannot
close. That forces a two-term, knobbed objective at step 2.

```python
class AdversarialTrainer:
    """PGD Adversarial Training (Madry et al., 2018)."""

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.attack_steps = attack_steps
        self.num_classes = num_classes

    def train_step(self, images, labels, optimizer):
        # Inner max: random-start L_inf PGD (eval mode so BN/dropout are frozen).
        self.model.eval()
        adv_images = images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)

        for _ in range(self.attack_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(adv_images)
            loss = F.cross_entropy(outputs, labels)               # untargeted: maximize true-label CE
            grad = torch.autograd.grad(loss, adv_images)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)  # project into eps-box
            adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()            # project into [0,1]

        # Outer min: one ordinary CE step on the adversarial batch (valid by Danskin).
        self.model.train()
        outputs = self.model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {'loss': loss.item()}
```
