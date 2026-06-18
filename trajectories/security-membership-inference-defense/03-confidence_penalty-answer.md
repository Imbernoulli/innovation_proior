**Problem.** Label smoothing relocated the member-confidence distribution but did not change its
separability from non-members, so the attack was untouched (resnet20 AUC even rose to 0.7678) and a
little accuracy was paid for nothing. The fix must apply pressure *where the over-confidence is*,
adaptively, not uniformly across every sample.

**Key idea.** Confidence penalty (Pereyra et al., 2017): subtract the predictive entropy from the
loss, `L = L_CE − β·H(p)` with `H(p) = −Σ_i p_i log p_i`. Penalizing low entropy penalizes confidence.
Its logit gradient `∂H/∂z_i = p_i(−log p_i − H)` weights each class by the model's own `p_i`, so the
update pushes hardest on the dominant, over-confident class and barely touches near-dead classes —
formally the *reverse* KL to uniform (`D_KL(p‖u) = −H(p) + log K`), versus label smoothing's *forward*
KL whose weight is the constant `1/K`. Adaptive where smoothing is uniform.

**Why (and its limit).** Concentrating on the spiked outputs is exactly what the membership signal
needs, so it should dent the attack where smoothing left it flat. But it is still a mean-region
intervention: it pushes spiked outputs toward uniform without a lever on the *variance* of the member
distribution, so it shifts/compresses rather than overlapping members with non-members. The scaffold
edit is the **plain, fixed-`β`, unscheduled** form — no annealing, no entropy hinge (the harness
exposes `epoch` but this baseline ignores it). That trades label smoothing's guaranteed safety for
adaptivity: a constant entropy push on the highest-capacity case (VGG-16-BN, 100 classes) is the
configuration most likely to fight training into a degenerate collapse, with no relief valve.

**Hyperparameters.** `β = 0.1` (`entropy_weight`), fixed across all three benchmarks, constant for all
300 epochs; `epoch` unused. `log_softmax` once for numerical stability. Fixed loop otherwise (SGD
lr 0.1, mom 0.9, wd 1e-4, MultiStepLR ×0.1 at 150/225, 300 epochs, batch 128, no augmentation).

```python
import torch
import torch.nn.functional as F


class MembershipDefense:
    """Cross-entropy minus a predictive-entropy bonus (Pereyra et al., 2017)."""

    def __init__(self):
        self.entropy_weight = 0.1

    def compute_loss(self, logits, labels, epoch):
        ce = F.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()
        return ce - self.entropy_weight * entropy
```
