## Research question

A multi-class classifier is trained by minimizing cross-entropy against a one-hot target — full probability on the correct class, zero on every other. With a softmax output this objective has a built-in pathology: the one-hot maximum is never reached at finite logits; it is only *approached* as the correct logit runs off toward +∞ relative to the rest. So gradient descent perpetually pushes the correct logit further above the others, and the network grows more and more confident on training examples. This is suspected to cause two related harms — overfitting (assigning full probability to the training label guarantees nothing about generalization) and reduced adaptability (the gap between the top logit and the rest grows unbounded while the per-logit gradient stays bounded). A simple, widely-adopted fix exists — softening the target away from one-hot — and it reliably improves accuracy across image classification, translation, and speech. The research question is not *whether* to soften the targets but *why and when* doing so helps: what does training against soft targets actually do to the network's internal representations and to the meaning of its output probabilities, when is that change beneficial, and when does it quietly destroy something a downstream task needs?

## Background

**Cross-entropy and the over-confidence mechanism.** With softmax p(k) = exp(z_k)/Σ_i exp(z_i) and a one-hot ground truth q(k) = δ_{k,y}, the loss is ℓ = −Σ_k q(k) log p(k) = −log p(y), with the clean gradient ∂ℓ/∂z_k = p(k) − q(k), bounded in [−1, 1]. Maximizing log p(y) is achieved only in the limit z_y ≫ z_k for all k ≠ y. Driving the correct logit arbitrarily far above the others is exactly what over-confidence is, and it is the documented source of the two harms above (Szegedy et al., 2016).

**Soft targets as a known regularizer.** Label smoothing (Szegedy et al., 2016, introduced for the Inception architecture on ImageNet) replaces the one-hot target with a mixture of the hard label and a fixed label distribution u(k): q'(k) = (1 − ε) δ_{k,y} + ε u(k), with u uniform, u(k) = 1/K, so q'(k) = (1 − ε)δ_{k,y} + ε/K. It can be read as a "marginalized label-dropout": with probability ε, the target label is resampled from u. Because every entry of q' now has a positive lower bound ε/K, an infinite logit gap incurs infinite cross-entropy, so the correct logit can no longer run away. The cross-entropy against q' splits as H(q', p) = (1 − ε) H(q, p) + ε H(u, p): ordinary hard-label cross-entropy plus a term penalizing the predicted distribution's deviation from the prior u (since H(u, p) = D_KL(u‖p) + H(u), and H(u) is constant). Typical setting on ImageNet: K = 1000, ε = 0.1.

**The confidence-penalty family.** Pereyra et al. (2017) penalize low-entropy (over-confident) outputs directly with a term −β H(p); they show label smoothing is equivalent to this confidence penalty with the direction of the KL divergence between the uniform distribution and the model's output reversed. They also generalize u from uniform to the empirical label prior (unigram label smoothing) for imbalanced outputs. DisturbLabel (Xie et al., 2016) randomly corrupts labels per minibatch ("label dropout"); label smoothing is its marginalized expectation. This family establishes that softening or entropy-regularizing the target is a general, task-agnostic regularizer.

**Calibration of modern networks.** Modern networks are accurate but poorly calibrated — their predicted confidence systematically exceeds their accuracy (Guo et al., 2017). Calibration is measured by the Expected Calibration Error (ECE): bin predictions by confidence and compare, per bin, average confidence against empirical accuracy on a reliability diagram. The standard post-hoc fix is temperature scaling — divide the logits by a learned scalar T before the softmax — which leaves the predicted class unchanged but flattens over-confident probabilities. Calibration is invisible to top-1 accuracy but critical wherever the soft probabilities feed a downstream algorithm: in sequence models, beam search approximates maximum-likelihood (Viterbi) decoding over the next-token probabilities, so a better-calibrated next-token distribution should yield better search.

**Knowledge distillation and "dark knowledge".** Knowledge distillation (Hinton et al., 2015) trains a small student to match a large teacher's softened outputs, loss (1 − β) H(y, p) + β H(p^t(T), p(T)) with temperature T exaggerating the relative probabilities among the *incorrect* classes. The value of distillation lies precisely in those relative incorrect-class probabilities — "this 3 looks somewhat like an 8" — information not present in the hard label. A geometric fact about the last layer makes the connection concrete: the logit z_k = x^T w_k (x the penultimate activations with a bias 1 appended, w_k the class-k template) relates to Euclidean distance by ‖x − w_k‖² = x^T x − 2 x^T w_k + w_k^T w_k; since x^T x is common to all classes (and cancels in the softmax) and w_k^T w_k is roughly constant across k, the logit is, up to those constants, the negative squared distance from the activation to the class template. So minimizing cross-entropy moves activations toward the correct template, and the geometry of where activations land is what encodes inter-class resemblance.

## Baselines

**Hard-target cross-entropy.** Train against the one-hot q(k) = δ_{k,y}. Core idea and math as above. Gap: the softmax maximum is unattainable, so the correct logit is pushed to run away, producing over-confidence — overfitting risk, reduced adaptability, and miscalibration (confidence > accuracy).

**Label smoothing (Szegedy et al., 2016).** Train against q'(k) = (1 − ε)δ_{k,y} + ε/K. Core idea: bound the logit gap by giving every class a positive target floor; equivalently, hard cross-entropy plus a uniform-deviation penalty. Gap to be understood: it reliably raises accuracy and is in every state-of-the-art recipe, yet *why* and *when* it helps — what it does to representations and to calibration, and whether it has costs — is not characterized.

**Confidence penalty (Pereyra et al., 2017).** Add −β H(p) to the loss to penalize over-confident outputs directly. Core idea: regularize by encouraging output entropy. Gap: a closely related but not identical objective (reversed KL direction); like label smoothing, its representational effect is uncharacterized.

**Temperature scaling (Guo et al., 2017).** Post-hoc calibration: divide logits by a scalar T tuned on a validation set. Core idea: rescale confidences without changing predictions. Gap: a post-processing step, not a training-time property; it fixes calibration but does nothing to the learned representation, and it is unknown how it relates to training with soft targets.

## Evaluation settings

The yardsticks span three supervised tasks. Image classification: CIFAR-10 (AlexNet), CIFAR-100 (ResNet-56), ImageNet (Inception-v4), reporting top-1 accuracy/error. Machine translation: English→German WMT with the Transformer, reporting BLEU, perplexity, and negative log-likelihood, with beam search as the decoder. Speech recognition: WSJ with a BiLSTM+attention model, reporting word error rate after beam search. The diagnostic instruments are: the Expected Calibration Error and 15-bin reliability diagrams (Guo et al., 2017); a penultimate-layer visualization — pick three classes, build an orthonormal basis of the plane through their three templates w_k, and project the penultimate activations onto it; and an estimate of the mutual information I(X; Y) between the training-example index X and the difference Y of two logits (randomness supplied by data augmentation, Y approximated as Gaussian). The smoothing strength ε (also written α) is swept over a range such as [0, 0.75], with small values (0.05–0.1) the usual operating point. Training protocols are standard SGD-with-momentum recipes (e.g. ResNet-56: batch 128, Nesterov momentum 0.9, lr 0.1 decayed ×10, weight decay, gradient clipping, crop+flip augmentation).

## Code framework

The available ingredients are a network producing logits, the softmax, a data loader of inputs and integer labels, and the standard training loop. The standard loss is cross-entropy against one-hot targets, implemented as log-softmax followed by a negative-log-likelihood gather at the target index. The open slot is the *criterion*: how the loss reads the target and what target distribution it implicitly compares the predictions against.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion(nn.Module):
    """Maps (logits, integer target) -> scalar loss.
    The baseline compares predictions to the one-hot target."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        logprobs = F.log_softmax(logits, dim=-1)
        nll = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        # TODO: decide what target distribution the loss compares against.
        return nll.mean()


# net producing logits, SGD(momentum), step-decay schedule, standard loop:
# for inputs, targets in loader:
#     loss = criterion(net(inputs), targets)
#     loss.backward(); optimizer.step(); optimizer.zero_grad()
```
