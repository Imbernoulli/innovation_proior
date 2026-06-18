**Problem.** ERM leaks because cross-entropy's one-hot target sits at an infinite logit gap, so the
model drives member confidence to the ceiling and a threshold separates members from non-members
(worst on vgg16bn-cifar100: 0.8677 AUC). The first fix is to stop asking for perfect confidence.

**Key idea.** Label smoothing (Szegedy et al., 2016): bleed a fraction `ε` of the target mass onto a
uniform prior, `q'(k) = (1−ε)δ_{k,y} + ε/K`. Every class now carries positive target mass, so an
infinite logit gap becomes infinitely costly and member confidence is bounded below 1. Equivalently
`H(q',p) = (1−ε)H(q,p) + εH(u,p)` — hard cross-entropy plus a uniform-deviation penalty. In the
scaffold this is one call: `F.cross_entropy(logits, labels, label_smoothing=ε)`.

**Why (and its limit).** Reading the logit as negative squared distance to a class template, smoothing
forces each activation close to its own template and *equidistant* from all others at bounded
magnitude — tight, equally-separated clusters with a confidence ceiling below 1. But it applies the
*same* pressure to every sample, so it *translates* the member-confidence distribution down without
changing its *spread*: members and non-members slide together rather than overlapping. A threshold
attacker reads separability, not absolute level, so the attack AUC is expected to move little while a
small accuracy cost is paid. Its virtue here is safety — a strictly-positive target can never
destabilize training — making it the collapse-proof probe of "how much does bounding confidence, with
no variance lever, buy?"

**Hyperparameters.** `ε = 0.1` (standard vision default; datasets are class-balanced so uniform `u` is
correct), fixed across all three benchmarks; `epoch` is ignored. Fixed loop otherwise (SGD lr 0.1,
mom 0.9, wd 1e-4, MultiStepLR ×0.1 at 150/225, 300 epochs, batch 128, no augmentation).

```python
import torch
import torch.nn.functional as F


class MembershipDefense:
    """Cross-entropy with fixed label smoothing (Szegedy et al., 2016)."""

    def __init__(self):
        self.label_smoothing = 0.1

    def compute_loss(self, logits, labels, epoch):
        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
```
