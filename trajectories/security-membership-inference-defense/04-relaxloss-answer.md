**Problem.** All three mean-control rungs failed on the attack's actual lever: smoothing only
relocated the member-confidence distribution, the confidence penalty dented but never broke it on
easy/moderate cases and *collapsed* CIFAR-100 (acc 0.01), and none made the member and non-member loss
distributions *overlap*. The optimal membership attack reads one number — the per-sample loss — so the
defense must make member loss `P` and non-member loss `Q` indistinguishable, without destabilizing
training.

**Key idea.** RelaxLoss (Chen, Yu, Fritz, ICLR 2022): relax the target member loss to a finite level
`α > 0` and hold it there with gradient ascent, which simultaneously (i) shrinks the mean gap to `Q`
and (ii) *fattens* `Var(P)` — a Taylor expansion shows ascent raises high-loss samples more than
low-loss ones (`Cov(ℓ, Δℓ) > 0`), spreading the member spike until it overlaps the wider non-member
hump. Both moves shrink the Gaussian/Hellinger bound on the optimal-attack AUC. Because relaxing
confidence can flip the argmax of hard samples, flatten the non-true-class posterior to the uniform
`(1−p_gt)/(K−1)` (maximizing the margin at fixed, *detached* `p_gt`), applied only to misclassified
samples.

**Why it breaks the wall.** The thermostat `|L − α|` holds member loss at finite `α` instead of driving
it to 0 (the leak) or to a degenerate point (the CIFAR-100 collapse) — the relief valve the constant
entropy penalty lacked. The variance lever is the one no mean-control rung pulled, and it is what
produces overlap, i.e. attack AUC moving toward 0.5.

**Two-phase rule (per batch).** Even epochs: `loss = |mean_CE − α|`. Odd epochs: if `mean_CE > α`,
`loss = mean_CE` (still undertrained → descend); else posterior flattening,
`loss = mean_i[(1 − correct_i)·CE_soft_i − CE_full_i]` with detached soft targets.

**Hyperparameters.** `α = 0.5` if `K = 100` else `1.0` (read from `logits.size(1)`, matching the
released configs); `upper = 1.0` (no-op clamp, kept for faithfulness); uses the `epoch` argument for
parity alternation. Fixed loop otherwise (SGD lr 0.1, mom 0.9, wd 1e-4, MultiStepLR ×0.1 at 150/225,
300 epochs, batch 128, no augmentation).

```python
import torch
import torch.nn.functional as F


class MembershipDefense:
    """RelaxLoss training rule (Chen et al., ICLR 2022).

    Even epochs: loss = |mean_CE - alpha|   (drive loss toward target level)
    Odd  epochs: if mean_CE > alpha -> CE descent
                 else -> posterior flattening with sign-flipped CE
    """

    def __init__(self):
        self.upper = 1.0

    def compute_loss(self, logits, labels, epoch):
        num_classes = logits.size(1)
        alpha = 0.5 if num_classes == 100 else 1.0

        loss_ce_full = F.cross_entropy(logits, labels, reduction='none')
        loss_ce = loss_ce_full.mean()

        if epoch % 2 == 0:
            return (loss_ce - alpha).abs()

        if loss_ce.item() > alpha:
            return loss_ce

        probs = torch.softmax(logits, dim=1)
        confidence_target = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        confidence_target = torch.clamp(confidence_target, min=0.0, max=self.upper)
        confidence_else = (1.0 - confidence_target) / (num_classes - 1)

        onehot = F.one_hot(labels, num_classes=num_classes).float()
        soft_targets = (
            onehot * confidence_target.unsqueeze(1)
            + (1.0 - onehot) * confidence_else.unsqueeze(1)
        )
        soft_targets = soft_targets.detach()

        log_probs = F.log_softmax(logits, dim=1)
        ce_soft = -(soft_targets * log_probs).sum(dim=1)

        pred = logits.argmax(dim=1)
        correct = pred.eq(labels).float()

        loss = (1.0 - correct) * ce_soft - loss_ce_full
        return loss.mean()
```
