**Problem (from step 3).** Cross-entropy against one-hot labels has no finite optimum — it rewards driving
the true-class logit gap to +∞, i.e. infinite confidence. On ImageNet (noisy, ambiguous labels) that means
memorizing label noise and producing a miscalibrated, overconfident model; the burned capacity is lost
generalization. Need cheap accuracy headroom from the *targets*, not the network.

**Key idea — Label Smoothing.** Soften the target: replace the one-hot label with a convex combination of
the one-hot and the uniform distribution, `y_smooth = (1 − α)·onehot + (α/K)·𝟙`. For ImageNet (K = 1000)
with α = 0.1, the true class gets ≈ 0.9001 and each other class 0.0001. Smooth the targets just before the
loss; restore the original hard targets afterward so training metrics use them.

**Why it works.** Cross-entropy against `y_smooth` *does* have a finite optimum: it's minimized when the
correct-class probability equals `1 − α + α/K < 1`, reachable with a finite logit gap. The infinite-
confidence pull is gone, so the model is confident-but-not-certain and stops being rewarded for memorizing
harder once it reaches the target confidence — a regularizer that curbs overfitting to label noise and
improves generalization. Throughput is unaffected (only the target tensor changes); the cost is on the
shared regularization budget (diminishing returns when stacked with weight decay and later regularizers).
The loss must accept a dense distribution; metrics must use the original hard labels.

**Change / code.** The functional core is one line — given logits (to read K and one-hot any index targets)
and targets, return the convex combination.

```python
def smooth_labels(logits: torch.Tensor, target: torch.Tensor, smoothing: float = 0.1):
    target = ensure_targets_one_hot(logits, target)
    n_classes = logits.shape[1]
    return (target * (1. - smoothing)) + (smoothing / n_classes)

# Usage in the loop: smooth for the loss, then restore the hard targets for metrics.
#   smoothed_targets = smooth_labels(y_hat, y, smoothing=0.1)
#   loss = loss_fn(y_hat, smoothed_targets)
#   # (class form restores y to the original targets after the loss is computed)
```
