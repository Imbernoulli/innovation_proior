I propose the canonical name "label smoothing" for this training recipe. In the standard multiclass softmax classifier, a model produces logits `z_k` and converts them to probabilities `p_k = exp(z_k) / sum_j exp(z_j)`. The usual cross-entropy loss then compares these probabilities against a one-hot target that places all of its mass on the true class `y`. Because that target assigns probability zero to every incorrect class, the loss is minimized only when the correct logit grows arbitrarily far above all the others. In other words, the one-hot objective chases an infinite true-vs-rest gap, which encourages the network to become overconfident and to keep increasing margins long after the example is classified correctly. Label smoothing repairs this by moving the target away from the one-hot corner.

Instead of using `q_k = 1[k = y]`, I replace the target with a mixture of the hard label and a uniform prior over all `K` classes. The smoothed target is `q'_k = (1 - epsilon) 1[k = y] + epsilon/K`, where `epsilon` is a small constant such as 0.1. This means the correct class receives probability `1 - epsilon + epsilon/K` and every incorrect class receives probability `epsilon/K`. Because every incorrect class now has positive target mass, an infinite logit gap becomes infinitely costly: if any incorrect probability is driven toward zero, its contribution to the cross-entropy diverges. The loss therefore prefers a finite gap. At the ideal softmax fit, where `p = q'`, all incorrect logits are equal and the correct-vs-incorrect gap is `z_y - z_j = log((1 - epsilon + epsilon/K) / (epsilon/K)) = log(1 + K(1 - epsilon)/epsilon)`. As `epsilon` shrinks, this gap grows, and in the limit `epsilon = 0` we recover the original hard-label objective. The finite gap is the central mechanism: it keeps the model's confidence bounded while preserving the discriminative signal of cross-entropy.

Cross-entropy is linear in the target, so the smoothed loss decomposes cleanly as `H(q', p) = (1 - epsilon) H(q, p) + epsilon H(u, p)`, where `q` is the one-hot target and `u` is the uniform distribution. For uniform `u`, the second term is `KL(u || p)` plus a constant. So label smoothing still trains on the hard label, but it adds a penalty that pulls the output distribution toward uniformity. This direction matters: it is `KL(u || p)`, not `KL(p || u)`, so it is especially intolerant of probabilities pushed all the way to zero. That is exactly the behavior we want to prevent the overconfidence runaway. A related but distinct idea is the confidence penalty, which penalizes low output entropy and corresponds roughly to `KL(p || u)`. The two regularizers act in opposite KL directions and should not be confused.

The effect of label smoothing is not limited to the output probabilities; it also changes the geometry of the learned representations. Writing the logit as `z_k = x^T w_k`, where `x` is a penultimate activation and `w_k` is the corresponding class template, the squared distance from `x` to template `k` is `||x - w_k||^2 = x^T x - 2 x^T w_k + w_k^T w_k`. Inside the softmax, the term `x^T x` is shared across classes, and the template norms are usually similar enough to treat as approximately constant. The varying term is therefore `-2 x^T w_k`, so larger logits correspond to smaller squared distances from the activation to the template. With hard labels, the loss asks each activation to be much closer to its own class template than to any other template, without saying much about the relative distances to the incorrect templates. With label smoothing, the target gives equal probability to every incorrect class, which encourages equal incorrect logits and therefore roughly equal squared distances from the activation to all incorrect templates. Each class cluster tightens around its own template, and the overall representation becomes more regular. This bounded logit scale and tighter clustering translate into better calibration, which can be measured with expected calibration error and reliability diagrams.

Better calibration matters beyond classification accuracy. In machine translation, beam search ranks sequence hypotheses using next-token probabilities, so a poorly calibrated or overconfident distribution can mislead decoding even when the per-token argmax is correct. Image models that feed probabilities into downstream retrieval or ranking systems also benefit from probabilities that match empirical frequencies. Label smoothing helps in these settings because it trains the model to report finite, better-calibrated confidence. However, the same mechanism has a cost when the trained model is meant to serve as a teacher for knowledge distillation. Distillation relies on the teacher's relative probabilities for incorrect classes: a teacher that says "this digit is mostly a 3 but somewhat like an 8" conveys similarity information that a hard label cannot. Because label smoothing pushes all incorrect-class logits toward equality, it erases example-specific similarity structure and can make an accurate teacher less useful for distillation. The same is true for transfer settings that depend on fine-grained structure in the final representation.

In practice, label smoothing is controlled by a single hyperparameter, `epsilon = 0.1` being the common default for image classification. Larger values pull the model more strongly toward uniform outputs and can reduce accuracy on easy tasks; smaller values retain more of the hard-label behavior. The recipe is applied during training through the loss function and requires no change to the model architecture or inference pipeline. It is worth monitoring both accuracy and calibration diagnostics, since the method can improve calibration even when the accuracy gain is small. When the goal is regularization and better-calibrated probabilities, label smoothing is a simple and effective modification of the standard cross-entropy target.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy for q'_k = (1 - eps) 1[k=y] + eps/K.

    This matches the Inception formula and the TensorFlow/PyTorch
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


if __name__ == "__main__":
    # Quick sanity check: for uniform logits, the loss equals log(K).
    K = 10
    logits = torch.zeros(4, K)
    target = torch.randint(0, K, (4,))
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss = criterion(logits, target)
    print("loss on uniform logits:", loss.item(), "expected log(K):", torch.log(torch.tensor(float(K))).item())

    # Check that the loss explodes when an incorrect probability is driven to zero.
    z = torch.zeros(1, K)
    z[0, 0] = 1e4
    z[0, 1] = -1e4
    hard = LabelSmoothingCrossEntropy(smoothing=0.0)
    smooth = LabelSmoothingCrossEntropy(smoothing=0.1)
    print("hard loss with one near-zero incorrect prob:", hard(z, torch.tensor([0])).item())
    print("smoothed loss with one near-zero incorrect prob:", smooth(z, torch.tensor([0])).item())
```
