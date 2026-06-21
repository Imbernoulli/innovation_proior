## One-Hot Softmax Cross-Entropy

In multiclass classification, a network typically emits logits `z_k`, converts them to probabilities with `p_k = exp(z_k) / sum_j exp(z_j)`, and minimizes cross-entropy against a one-hot label. For an example with class `y`, that loss is `-log p_y`, and its logit gradient is `p_k - 1[k = y]`.

The research question is how to set the target distribution passed to the cross-entropy criterion during supervised training on image classification, translation, or speech recognition tasks.

## Existing Soft-Target Tools

Several pre-2019 recipes replace a hard target with a softer target distribution. The Inception-v2/Inception-v3 training recipe uses a mixture of the one-hot label and a prior over labels:

`q'_k = (1 - epsilon) 1[k = y] + epsilon u_k`.

With a uniform prior over `K` classes, this becomes `q'_y = 1 - epsilon + epsilon/K` and `q'_j = epsilon/K` for each incorrect class. Because cross-entropy is linear in the target, `H(q', p) = (1 - epsilon) H(q, p) + epsilon H(u, p)`. When `u` is uniform, the second term is `KL(u || p)` plus a constant.

Other nearby ideas soften confidence in related but not identical ways. A confidence penalty adds a low-entropy penalty to the model output, corresponding to the reverse KL direction from the uniform distribution. Label corruption methods randomly replace training labels; their expectation resembles a softened target.

## Calibration

A classifier's accuracy and its reported probability need not agree. Reliability diagrams and expected calibration error compare confidence bins against empirical accuracy; they expose failures that top-1 accuracy hides.

Temperature scaling is a simple post-training calibration baseline: divide logits by a scalar temperature before softmax. This leaves the argmax unchanged but can make probabilities less overconfident.

## Soft Outputs As Transfer Signal

Soft probabilities can be useful beyond the original classifier. In knowledge distillation, a student is trained to match a teacher's softened output distribution, often at elevated temperature. The value is not just the largest probability; the small probabilities assigned to incorrect classes carry similarity information about the example.

Sequence decoding also consumes probabilities directly. Beam search ranks partial hypotheses using next-token likelihoods, so the shape of the probability distribution can affect the final decoded sequence even when the per-token argmax is unchanged.

## Experimental Frame

The immediate ingredients are standard: logits, integer labels, a cross-entropy criterion, and supervised training on image classification, translation, or speech recognition benchmarks. Useful diagnostics include accuracy or error, negative log-likelihood or perplexity, expected calibration error, reliability diagrams, and probes of the final representation and logits.

```python
import torch.nn.functional as F


def hard_target_loss(logits, target):
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    return nll.mean()


# The unresolved design choice is which target distribution the
# criterion should compare against.
```
