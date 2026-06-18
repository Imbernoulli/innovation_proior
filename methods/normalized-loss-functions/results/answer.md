# Active Passive Loss: NCE + RCE

## Method

Use a robust active term plus a robust passive term:

```text
L(logits, y) = alpha * NCE(logits, y) + beta * RCE(logits, y)
```

where

```text
NCE = (-log p_y) / (-sum_j log p_j)
RCE = -sum_j p_j log q_j
```

`p = softmax(logits)`, `q` is the one-hot label vector, and the zero entries of `q` are clamped so
`log q_j = A < 0` off the labeled class.

## Guarantees

`NCE` is symmetric by normalization: its class-sum over candidate labels is `1`. `RCE` is already
symmetric: with off-label `log q_j = A`, `RCE = -A(1 - p_y)` and its class-sum is `-A(K - 1)`.
Therefore `alpha*NCE + beta*RCE` has constant class-sum `alpha + beta*(-A)(K - 1)` and inherits the
uniform-label-noise tolerance theorem for `eta < (K - 1)/K`. The class-conditional guarantee requires
extra assumptions: zero clean risk, bounded normalized losses, and a diagonally dominant
noise transition.

## Parameters

`alpha` and `beta` are explicit balance weights. The reference implementation passes them through
dataset configs rather than treating one pair as universal: examples include CIFAR-10 `alpha=1,
beta=1`, CIFAR-100 `alpha=10, beta=0.1`, and WebVision-mini `alpha=50, beta=0.1`.

## Code

```python
import torch
import torch.nn.functional as F


class RobustLoss:
    """Active Passive Loss: normalized cross entropy plus reverse cross entropy."""

    def __init__(self, num_classes, alpha=1.0, beta=1.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

    def _one_hot(self, labels, logits):
        return F.one_hot(labels, self.num_classes).to(dtype=logits.dtype, device=logits.device)

    def _nce(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=1)
        one_hot = self._one_hot(labels, logits)
        numerator = -(one_hot * log_probs).sum(dim=1)
        denominator = -log_probs.sum(dim=1)
        return (numerator / denominator).mean()

    def _rce(self, logits, labels):
        probs = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
        one_hot = self._one_hot(labels, logits).clamp(min=1e-4, max=1.0)
        return (-(probs * torch.log(one_hot)).sum(dim=1)).mean()

    def compute_loss(self, logits, labels, epoch):
        return self.alpha * self._nce(logits, labels) + self.beta * self._rce(logits, labels)
```
