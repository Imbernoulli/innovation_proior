# MART, distilled

MART (Misclassification Aware adveRsarial Training) is an adversarial-training objective that
explicitly differentiates **misclassified** natural examples from **correctly classified** ones
during the outer minimization. It generates adversarial inputs with the same strong CE-PGD attack
as standard adversarial training, but replaces the single cross-entropy loss with (a) a *boosted*
cross-entropy on the adversarial input and (b) a clean-vs-adversarial KL regularizer that is
re-weighted per example by the model's error probability `1 - p_y(x)`, so the regularizer leans
hardest on the examples the model already gets wrong.

## Problem it solves

Standard PGD adversarial training treats every example identically in both the inner attack and
the outer loss, leaving a large gap between robust and clean accuracy. An adversarial example is
only defined relative to a *correctly classified* natural example, yet during training many
natural examples are already misclassified. A diagnostic split of the training set (by the current
model's clean prediction) shows that misclassified examples dominate the final robustness, that
the *inner attack strength* on them is nearly irrelevant, and that the *outer loss* on them is
decisive. MART puts the new structure entirely in the outer loss and singles out the misclassified
examples.

## Key idea

Partition by the current model's prediction on the clean image: `S+ = {h(x)=y}`,
`S- = {h(x)!=y}`. Against the 0-1 loss, keep the standard adversarial risk for everyone and add a
**stability** term for the misclassified ones (asking the perturbed prediction to match the clean
prediction rather than the label). On `S+` the stability term collapses into the adversarial risk,
so the two halves merge into one gated risk:

```
R_misc(h) = (1/n) sum_i { 1(h(xhat'_i) != y_i)
                          + 1(h(x_i) != h(xhat'_i)) · 1(h(x_i) != y_i) },
   xhat'_i = argmax_{x' in B_eps(x_i)} 1(h(x') != y_i).
```

The second term is the **misclassification-aware regularizer**, gated by `1(h(x_i) != y_i)`.

## Surrogate losses (the trainable objective)

Replace each indicator with a meaningful differentiable surrogate:

- `1(h(xhat')!=y)` → **boosted cross-entropy (BCE)**, CE plus a margin term that drives down the
  best competing class (robust classification needs a wider-margin / stronger classifier):

  ```
  BCE(p(xhat'), y) = -log p_y(xhat') - log(1 - max_{k!=y} p_k(xhat')).
  ```

- `1(h(x)!=h(xhat'))` → **KL divergence** between clean and adversarial outputs,
  `KL(p(x) || p(xhat')) = sum_k p_k(x) log(p_k(x)/p_k(xhat'))`.

- `1(h(x)!=y)` (the gate) → **soft decision** `1 - p_y(x)`: large for misclassified / low-confidence
  examples, near zero for confident-correct ones. Differentiable and jointly learnable, unlike a
  hard threshold; it gives a *graded* per-example weight.

The per-example MART loss and objective:

```
ell(x, y, theta) = BCE(p(xhat'), y) + lambda · KL(p(x) || p(xhat')) · (1 - p_y(x)),
L_MART(theta)    = (1/n) sum_i ell(x_i, y_i, theta),
```

with `xhat'` from CE-PGD and a single global trade-off scalar
`lambda` (default `lambda = 6`), fixed across examples — the per-example weighting is already
carried by `1 - p_y(x)`.

Inner maximization (standard PGD, same as standard adversarial training):

```
x'_0 = clamp(x + random_noise, 0, 1);
x'_{s+1} = Proj_{B_eps(x)} ( x'_s + eta · sign( grad_{x'} CE(p(x'_s), y) ) );   xhat' = x'_TI.
```

## Defaults and why

- `lambda = 6`: a global natural-vs-robust balance (same range as the KL-decomposition baseline);
  fixed per example because `1 - p_y(x)` already reallocates weight toward hard examples.
- Inner attack = strong CE-PGD (small random nudge, step `eps/4`, 10 steps for CIFAR): the diagnostic
  shows attack strength on misclassified examples barely affects robustness, so reuse the cheap,
  strong, standard attack and put all new structure in the minimization.
- The implementation uses small numerical guards (`1.0001`, `1.0000001`, `+1e-12`) around the
  margin/KL logarithms and the soft gate for stable arithmetic at probability extremes.

## Relation to prior methods

- **Standard PGD-AT**: `CE(p(xhat'), y)` only — one loss for all examples, no differentiation.
- **TRADES**: `CE(p(x), y) + (1/lambda) max KL(p(x)||p(x'))` — fits the *clean* label, KL inner-max,
  and a *uniform* per-example regularizer weight. MART fits the *adversarial* input with boosted CE
  and weights the KL *per example* by `1 - p_y(x)`.
- **ALP/CLP**: `L2` logit pairing with uniform weight. **MMA**: a *hard* correct/misclassified
  split with per-example `eps`; MART uses a *soft*, jointly-learnable gate `1 - p_y(x)` and a fixed
  `eps`.

## Working code

Fills the `train_step` slot of the adversarial-training harness:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialTrainer:
    """MART: Misclassification Aware adveRsarial Training."""

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.attack_steps = attack_steps
        self.num_classes = num_classes
        self.beta = 6.0                     # lambda: KL-regularizer weight
        self.kl = nn.KLDivLoss(reduction='none')

    def train_step(self, images, labels, optimizer):
        # inner maximization: strong L_inf CE-PGD with a small random nudge
        self.model.eval()
        adv = images.detach() + 0.001 * torch.randn_like(images)
        adv = torch.clamp(adv, 0.0, 1.0)
        for _ in range(self.attack_steps):
            adv.requires_grad_(True)
            loss_ce = F.cross_entropy(self.model(adv), labels)
            grad = torch.autograd.grad(loss_ce, adv)[0]
            adv = adv.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv - images, min=-self.eps, max=self.eps)
            adv = torch.clamp(images + delta, 0.0, 1.0).detach()

        # outer minimization: MART loss
        self.model.train()
        optimizer.zero_grad()
        batch_size = images.size(0)

        logits = self.model(images)
        logits_adv = self.model(adv)
        adv_probs = F.softmax(logits_adv, dim=1)

        # boosted CE = CE(adv, y) - log(1 - max_{k!=y} p_k(adv))
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == labels, tmp1[:, -2], tmp1[:, -1])
        loss_adv = F.cross_entropy(logits_adv, labels) \
            + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

        # misclassification-aware KL regularizer, weighted by (1 - p_y(x))
        nat_probs = F.softmax(logits, dim=1)
        true_probs = nat_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        kl_per_sample = torch.sum(
            self.kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1)
        loss_robust = (1.0 / batch_size) * torch.sum(
            kl_per_sample * (1.0000001 - true_probs))

        loss = loss_adv + self.beta * loss_robust
        loss.backward()
        optimizer.step()
        return {'loss': loss.item()}
```
