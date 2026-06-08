**Problem.** Plain cross-entropy charges full price on examples the model has already mastered, so if easy examples dominate they can capture most of the gradient. On the frozen substrate (ResNet-56 / VGG-16-BN on CIFAR-100, MobileNetV2 on FashionMNIST; SGD + cosine over 200 epochs) the only lever is the loss.

**Key idea.** Focal loss multiplies cross-entropy by `(1-P_t)^γ`: near 1 for hard/wrong examples (`P_t` small), collapsing toward 0 for easy ones (`P_t → 1`), so the gradient flows to examples still getting it wrong. With `γ = 2`, an example at `P_t = 0.9` is down-weighted ~100×, one at `P_t = 0.5` only ~4×.

**Why (and the caveat).** Focal loss was built for dense detection's 1:1000 easy-background imbalance, where easy negatives *are* the loss. These three benchmarks are *balanced* classification, so there is no over-represented easy class to tamp down; the factor only shifts the gradient budget toward less-confident examples late in training. On this multiclass softmax surface there is no per-class `α`, no sigmoid/prior-bias machinery — only the modulating factor transfers.

**Step-1 edit.** Recover `P_t = exp(-L_CE)` from the unreduced softmax cross-entropy, multiply by `(1-P_t)²`, mean-reduce. `γ = 2.0` fixed, no `α`. It is the ladder's floor.

**What to watch.** Expect accuracy *near* cross-entropy on all three, since the factor does not change which class is correct. Risk: on balanced data, suppressing easy examples by ~100× may starve the late-training gradient — most visible on the high-accuracy MobileNetV2 / FashionMNIST pair.

```python
# EDITABLE region of pytorch-vision/custom_loss.py — step 1: focal loss (gamma=2.0)
def compute_loss(logits, targets, config):
    """Focal Loss (gamma=2.0): modulate CE by (1 - p_t)^gamma to focus on hard
    examples, reducing the relative loss on well-classified ones."""
    ce = F.cross_entropy(logits, targets, reduction='none')   # -log(p_t), [B]
    pt = torch.exp(-ce)                                        # p_t = exp(-CE), [B]
    return ((1 - pt) ** 2.0 * ce).mean()                      # (1 - p_t)^2 * CE
```
