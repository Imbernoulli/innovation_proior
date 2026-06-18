**Problem (from step 3).** GCE drove CIFAR-100 `poison_fit` to `0.0264` but its `test_acc` fell to `0.6674` — below plain cross entropy. That is underfitting: `q = 0.7` slid the loss toward the MAE end, where the gradient flattens and hard classes never reach their clean ceiling. Every rung bought robustness by paying in fitting, because each blended a robust (symmetric) loss with a non-robust fitting term and tuned the balance. The gap left is an underfitting gap.

**Key idea.** Stop treating symmetry as a property to *find* and *make* any loss symmetric: `L_norm = L(f,y)/sum_k L(f,k)` has constant class-sum by construction, hence is noise-tolerant for any base loss. Normalized cross entropy is then robust *and* keeps the labeled-class up-push. But a normalized active loss alone still underfits, because it is *active* — defined only on `f_y`, no lever on wrong-class mass. Pair it with a *passive* loss (RCE, defined on `f_{k!=y}`): `alpha·L_active + beta·L_passive` stays symmetric (sum of symmetric losses, *both* robust — no compromise term) while the two complementary behaviors jointly concentrate the prediction. The instance is **NCE + RCE**.

**Why it works.** NCE `(Σ_k q_k log f_k)/(Σ_j log f_j)` is robust by normalization and restores cross entropy's hard-example pull — the dense gradients CIFAR-100 needs and GCE gave up. RCE `−A(1−f_y)` (with `A=log 0` set by clamping the one-hot) is already symmetric, the passive down-push; its logit gradient self-gates (suppress wrong-class mass `∝ f_y`, silent on examples it flags as mislabeled). Active up-push + passive down-push, both noise-tolerant: robustness and fitting are no longer on one axis. Fits the contract exactly — a pure function of the current batch's logits and labels, no indices, no model access, no cross-epoch state.

**Hyperparameters.** `alpha = beta = 1.0` (canonical untuned NCE+RCE; the harness fixes one global objective, so no per-dataset search). NCE from `log_softmax`, denominator the class-sum `−log_probs.sum(1)`. RCE clamps the prediction at `1e-7` and the one-hot at `1e-4` (so `A = log(1e-4)`). `num_classes` read from `logits.shape[1]` (the scaffold constructs the module with no arguments). No `epoch` use.

```python
class RobustLoss:
    """Active Passive Loss: normalized cross-entropy + reverse cross-entropy."""

    def __init__(self):
        self.alpha = 1.0
        self.beta = 1.0

    def compute_loss(self, logits, labels, epoch):
        num_classes = logits.shape[1]
        one_hot = F.one_hot(labels, num_classes).float()

        # Active term: normalized cross entropy (symmetric by construction)
        log_probs = F.log_softmax(logits, dim=1)
        nce_num = -(one_hot * log_probs).sum(dim=1)          # -log f_y
        nce_den = -log_probs.sum(dim=1)                      # sum_j -log f_j (class-sum)
        nce = (nce_num / nce_den).mean()

        # Passive term: reverse cross entropy (already symmetric; A = log(1e-4))
        probs = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
        rce_target = one_hot.clamp(min=1e-4, max=1.0)
        rce = -(probs * torch.log(rce_target)).sum(dim=1).mean()

        return self.alpha * nce + self.beta * rce
```
