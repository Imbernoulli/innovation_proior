# Label smoothing

## Problem

A softmax classifier trained with cross-entropy against one-hot targets has its optimum at an unreachable point: p(y) → 1 only as the correct logit z_y runs to +∞ above all others. Training therefore pushes the network to ever-greater confidence on its training labels, causing overfitting and reduced adaptability (the largest-minus-rest logit gap grows unbounded against a bounded gradient). Softening the target away from one-hot fixes this and reliably improves accuracy across vision, translation, and speech — but the question is *why and when* it helps: what it does to the learned representation, to calibration, and what it costs.

## Key idea

Replace the one-hot target with a mixture of the hard label and a uniform prior:

  q'(k) = (1 − ε) δ_{k,y} + ε u(k),   u(k) = 1/K   ⇒   q'(k) = (1 − ε)δ_{k,y} + ε/K.

Every class now has a positive target floor ε/K, so an infinite logit gap incurs infinite cross-entropy — the correct logit can no longer run away. The loss decomposes as

  H(q', p) = (1 − ε) H(q, p) + ε H(u, p),

i.e. hard cross-entropy plus an ε-weighted penalty on the prediction's deviation from uniform (H(u, p) = D_KL(u‖p) + const). Typical ε ≈ 0.1.

## Why it helps (and when it hurts)

- **Bounds over-confidence → regularizes and raises accuracy.** The positive floor forbids the runaway logit gap that hard targets chase to infinity.

- **Penultimate-layer geometry.** With z_k = x^T w_k, the identity ‖x − w_k‖² = x^T x − 2 x^T w_k + w_k^T w_k (with x^T x cancelling in the softmax and w_k^T w_k ≈ const) makes the logit a negative squared distance from the activation x to a class template w_k. Hard targets only require x much closer to w_y than to anything else → broad, large-magnitude clusters. Smoothing sets every wrong-class target equal (ε/K), forcing x to be *equidistant* from all incorrect templates and a fixed finite distance from the correct one → tight, equally-separated, bounded-magnitude clusters (three classes project to a regular triangle).

- **Implicit calibration.** Bounding logit magnitudes during training aligns confidence with accuracy, achieving an Expected Calibration Error comparable to post-hoc temperature scaling without any post-hoc tuning. Calibration is invisible to top-1 accuracy but decisive where soft probabilities feed a downstream search: beam search (approximate Viterbi MLE) consumes the next-token distribution, so smoothing improves BLEU — consistent with the observation that smoothing improves BLEU *despite* worse perplexity/NLL (the gain is calibration, only partly).

- **Hurts distillation and transfer.** The same cluster-tightening flattens the example-specific *relative* similarities among incorrect classes ("this 3 resembles an 8") — the "dark knowledge" distillation relies on. So a smoothing-trained teacher can be more accurate yet distill into a worse student; a better teacher is not necessarily a better distiller. Quantified as a drop in the mutual information between training-example index and logit-difference, decaying toward the one-bit (log 2) floor where only the class label survives. The same loss of non-class-relevant final-layer structure impairs transfer learning.

## Code

The criterion is the two-term decomposition with u uniform; no soft-label vector is materialized. The first term is the negative log-likelihood at the target; the second, since H(u, p) = mean_k(−log p_k), is ε times the mean of the negative log-probabilities.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """NLL loss with label smoothing: (1-eps) H(q,p) + eps H(u,p), u uniform."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)        # H(u,p), u(k)=1/K
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
```

Equivalently, modern frameworks expose this directly (e.g. `nn.CrossEntropyLoss(label_smoothing=0.1)`). Typical settings: ε = 0.1 for ImageNet (K = 1000) and Transformer translation; ε ≈ 0.05 suffices to calibrate CIFAR-100/ResNet-56. Use it for accuracy and calibration; avoid it on a teacher you intend to distill from, or on a backbone you intend to transfer.
