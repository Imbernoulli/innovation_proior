# Label Smoothing

Replace the one-hot target `q_k = 1[k = y]` with a mixture of the hard label and the uniform distribution over `K` classes:

`q'_k = (1 - epsilon) 1[k = y] + epsilon/K`.

Thus `q'_y = 1 - epsilon + epsilon/K` and `q'_j = epsilon/K` for every incorrect class. The loss is

`H(q', p) = (1 - epsilon) H(q, p) + epsilon H(u, p)`,

where `u_k = 1/K` and `H(u, p) = KL(u || p) + H(u)`. At the finite softmax optimum, all incorrect logits are equal and

`z_y - z_j = log((1 - epsilon + epsilon/K) / (epsilon/K)) = log(1 + K(1 - epsilon)/epsilon)`.

That finite gap is the core repair: hard-label cross-entropy chases an infinite gap; smoothed targets make an infinite gap infinitely costly because every incorrect class has positive target mass.

The representation effect follows from `z_k = x^T w_k` and

`||x - w_k||^2 = x^T x - 2 x^T w_k + w_k^T w_k`.

Since `x^T x` cancels across classes in the softmax and template norms are approximately constant, logits behave like negative squared distances to class templates. The smoothed target encourages each activation to stay close to its correct template while being nearly equidistant from incorrect templates. This tightens class clusters and bounds logit scale, which improves calibration and can help downstream probability consumers such as beam search.

The cost is the same mechanism in reverse. Equalizing incorrect-class logits removes example-specific similarity structure in the teacher outputs. That harms knowledge distillation and can hurt transfer settings that depend on non-class-relevant information in the final representation. Use smoothing for regularization and calibration; avoid it when the trained model is meant to serve as a distillation teacher unless that tradeoff has been tested.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy for q'_k = (1-eps) 1[k=y] + eps/K.

    This matches the Inception formula and TensorFlow's
    softmax_cross_entropy(label_smoothing=eps) convention.
    """

    def __init__(self, smoothing=0.1, reduction="mean"):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1).")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'.")
        self.smoothing = float(smoothing)
        self.reduction = reduction

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        uniform = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll + self.smoothing * uniform

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()
```

For comparison, some sequence-model codebases use the alternate convention `q_y = 1 - epsilon`, `q_j = epsilon/(K - 1)` and may subtract the target entropy as a learning-neutral normalization. The class above intentionally implements the all-classes mixture used by the Inception formula and TensorFlow/PyTorch's built-in label-smoothing cross-entropy.
