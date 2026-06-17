# RelaxLoss

## Problem

A classifier trained to near-zero loss on its members leaks membership: the Bayes-optimal
membership inference attack depends on the model only through the per-sample loss
`ell(theta, z)`, so it reduces to thresholding the loss, and membership advantage equals the
generalization gap (the difference between typical non-member and member loss). RelaxLoss is
a training-time defense that suppresses this leak while preserving — or improving — test
accuracy, at negligible overhead, with a single hyperparameter and no assumption about the
attack.

## Key idea

The attacker can only separate the member-loss distribution `P` from the non-member
distribution `Q`. Vanilla cross-entropy crushes `P` into a tight spike near zero, far from
`Q`. RelaxLoss makes the two overlap with two coupled moves:

1. **Relax the target loss.** Aim the *mean* member loss at a level `alpha > 0` that is also
   achievable for non-members, instead of at 0. Descend when the batch loss is above `alpha`,
   **gradient-ascend** when it is below — a thermostat around `alpha`. This shrinks the mean
   gap `|mu_P - mu_Q|` and, because the per-sample loss increase under ascent grows with the
   squared gradient norm (which itself grows with the loss), it pushes high-loss samples up
   harder and thereby **increases the variance of `P`**: `Cov(ell, Delta ell) > 0` gives
   `Var(ell + Delta ell) > Var(ell)`. Together these effects lower the upper bound on the optimal
   attack AUC through `AUC <= -1/2 D_TV^2 + D_TV + 1/2` and the Gaussian Hellinger bound:
   the smaller mean gap shrinks `D_H`; increasing member variance raises the total variance
   denominator in the exponential term, and while `Q` is wider than `P`, it also moves the
   variance-ratio prefactor toward its maximum at equality. The isolated `c`-only exponential
   effect points the other way if `sigma_1` is frozen, so the monotonic claim is the combined
   one: smaller mean gap plus larger member variance lowers the Hellinger/AUC upper bound in
   the intended regime.

2. **Flatten the posterior to protect utility.** Relaxing the loss lets `p_gt` drop below 1,
   which can flip the argmax on hard samples. To keep accuracy at fixed `p_gt`, build a soft
   target that retains `p_gt` on the truth and spreads the remaining mass `1 - p_gt` evenly
   over the other `C - 1` classes — minimizing the largest competitor and maximizing the
   margin. The target is detached (stop-gradient) so it never re-sharpens `p_gt`.

The two operations are alternated by epoch parity. Even epochs minimize `|L_ce - alpha|`,
which descends when `L_ce > alpha` and ascends when `L_ce < alpha`. Odd epochs use ordinary
descent while `L_ce > alpha`, and switch to posterior flattening only once `L_ce <= alpha`.
`alpha` is the single privacy-utility knob.

## Algorithm (per batch)

Let `L_ce` be the mean batch cross-entropy, `ell_i` the per-sample cross-entropy, `C` the
number of classes, `alpha` the target loss, `correct_i = 1[argmax = label]`.

```
if epoch is even:                      # abs objective: descent above alpha, ascent below
    loss = |L_ce - alpha|
elif L_ce > alpha:                     # odd epoch, undertrained -> descent
    loss = L_ce
else:                                  # odd epoch, at/below target -> posterior flattening
    t_i^c = p_i^gt                       if c == gt
            (1 - p_i^gt) / (C - 1)       otherwise        (detached)
    ell_soft_i = -sum_c t_i^c log p_i^c
    loss = mean_i [ (1 - correct_i) * ell_soft_i - ell_i ]
```

## Code

```python
import torch
import torch.nn.functional as F


class MembershipDefense:
    """RelaxLoss: relax the target loss to alpha, hold it with gradient ascent
    (which also spreads the member-loss variance), and flatten the
    non-ground-truth posterior to keep the argmax correct at a fixed p_gt."""

    def __init__(self):
        # clamps p_gt before building the soft target; 1.0 is a no-op
        self.upper = 1.0

    def compute_loss(self, logits, labels, epoch):
        num_classes = logits.size(1)
        # single hyperparameter: a target loss level reachable by non-members
        alpha = 0.5 if num_classes == 100 else 1.0

        loss_ce_full = F.cross_entropy(logits, labels, reduction='none')
        loss_ce = loss_ce_full.mean()

        if epoch % 2 == 0:
            # thermostat: descend |L - alpha| -> ascent when below alpha
            return (loss_ce - alpha).abs()

        if loss_ce > alpha:
            # still undertrained -> ordinary cross-entropy descent
            return loss_ce

        # posterior flattening
        probs = torch.softmax(logits, dim=1)
        p_gt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        p_gt = torch.clamp(p_gt, min=0.0, max=self.upper)
        p_else = (1.0 - p_gt) / (num_classes - 1)

        onehot = F.one_hot(labels, num_classes=num_classes).float()
        soft_targets = onehot * p_gt.unsqueeze(1) + (1.0 - onehot) * p_else.unsqueeze(1)
        soft_targets = soft_targets.detach()                 # constant target

        log_probs = F.log_softmax(logits, dim=1)
        ce_soft = -(soft_targets * log_probs).sum(dim=1)     # soft cross-entropy

        correct = logits.argmax(dim=1).eq(labels).float()
        # flatten only misclassified samples; subtract per-sample CE (ascent)
        loss = (1.0 - correct) * ce_soft - loss_ce_full
        return loss.mean()
```

Operational details: the even branch always uses `(loss_ce - alpha).abs()`; the odd branch
uses ordinary cross-entropy when `loss_ce > alpha`; otherwise it builds a detached even-mass
soft target, applies soft cross-entropy to the incorrect predictions, and subtracts
per-sample cross-entropy. `alpha = 0.5` for 100-class tasks and `1.0` otherwise; `upper =
1.0` leaves `p_gt` unclamped.
