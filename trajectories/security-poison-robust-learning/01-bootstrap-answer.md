**Problem.** Cross entropy regresses the prediction toward the one-hot label, and its `1/f_y` gradient pours the largest updates into exactly the poisoned examples (the model is confidently wrong about the flipped target), so high-capacity nets memorize the lies. The loss trusts every label unconditionally and trusts the wrong ones hardest.

**Key idea.** Keep the cross-entropy *form* but soften the *target*: regress toward a convex combination of the hard label and the model's own (detached) current prediction, `t = β·e_y + (1−β)·softmax(logits.detach())`. A perceptually-inconsistent label is then partially overridden by what the model already believes — the learner is given justification to disagree with a suspect label and effectively re-labels while training. As the model improves, targets clean up; cleaner targets give a better model; the learner bootstraps itself.

**Why it works (and its ceiling).** On clean points the detached prediction concentrates on `e_y`, so the target ≈ `e_y` and nothing changes. On a poisoned point where the model has learned the true structure, the target keeps `(1−β)·p̂_true` mass on the right class, so the descent target on the flipped logit is `t_ỹ = β + (1−β)p̂_ỹ < 1` and the memorization pressure is relieved by `(1−β)(1−p̂_ỹ)`. The relief is gated by the model's own belief — which is exactly the quantity the corrupt labels distort — so the protection is circular: weak early (model hasn't learned to disagree yet) and self-endorsing late (it ratifies whatever it memorized). This is the *conservative* end of the family and the smallest departure from ERM, deliberately so as the first rung.

**Hyperparameters.** Soft-bootstrap target weight `β = 0.8` (high trust in the label; soft self-vote rather than hard argmax). The inserted prediction is **detached** (`softmax(logits.detach())`) so it acts as a regression target — the gradient flows only through the model-fitting term; differentiating through `p̂` would let the model trivially match itself. No schedule, no auxiliary term, no `epoch` use.

```python
class RobustLoss:
    """Interpolate labels with model predictions."""

    def __init__(self):
        self.beta = 0.8

    def compute_loss(self, logits, labels, epoch):
        hard = F.one_hot(labels, num_classes=logits.shape[1]).float()
        soft = torch.softmax(logits.detach(), dim=1)
        target = self.beta * hard + (1.0 - self.beta) * soft
        log_probs = F.log_softmax(logits, dim=1)
        return -(target * log_probs).sum(dim=1).mean()
```
