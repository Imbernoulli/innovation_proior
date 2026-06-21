The problem is training a classifier on a long-tailed dataset and then evaluating it on a balanced test set. In the training split, a few head classes contain most of the images while many tail classes have only a handful, so the empirical label distribution is heavily skewed. A model trained by ordinary cross-entropy learns the training posterior, which bakes in the training prior. At test time the prior is gone and the metric scores a uniform prior, so the model systematically under-predicts tail classes and balanced accuracy collapses.

The standard fixes all fall short. Inverse-frequency reweighting multiplies each class's loss by a scalar like 1/n_j, but this is model-blind: it rescales gradients uniformly without correcting where the decision boundary sits, and on separable data the converged max-margin solution is invariant to such weights. It also produces huge, unstable gradients when the imbalance is severe. Resampling the data to be balanced either overfits the tail or throws away the head's useful variation. Effective-number reweighting smooths the inverse-frequency weights, but it is still a static scalar on the loss and does not model the train-test posterior mismatch. Margin-based methods such as LDAM enforce larger margins for rare classes, yet they extend binary-hinge reasoning heuristically and require tuning. What all of these miss is that the Softmax posterior itself factorizes into a shared class-conditional likelihood and a class prior, and the prior is exactly the thing that differs between training and testing.

The right object to correct is the prior inside the Softmax. The proposed method is Balanced Softmax. It is a drop-in replacement for ordinary Softmax cross-entropy that makes the model's logits represent the balanced test posterior while training on the imbalanced split.

The derivation is short. The balanced posterior is φ_j = p(y=j|x) = p(x|y=j)p(y=j)/p(x), and with a uniform test prior p(y=j)=1/k this is proportional to the shared likelihood p(x|y=j). The training posterior is φ̂_j = p̂(y=j|x) = p(x|y=j)p̂(y=j)/p̂(x), with p̂(y=j)=n_j/n. If the logits η are the canonical Softmax parameterization of the balanced posterior, then φ_j = e^{η_j}/Σ_i e^{η_i}. Solving for the training posterior in terms of the same η gives φ̂_j = n_j e^{η_j} / Σ_i n_i e^{η_i}. In other words, the imbalanced training posterior is the balanced posterior with the class counts multiplied back in, which is additive in log-space. Therefore the training loss should be -log φ̂_y = -log Softmax(η + log n)_y. That is the entire method: add log n_j to each logit, run standard cross-entropy, and at test time use the bare logits because they now model the balanced posterior.

This is not a heuristic reweighting. The log-count offset is exact under the label-shift assumption that train and test share the likelihood, and it is also the Bayes-optimal correction for balanced error, which requires dividing the training posterior by its prior. The offset grows only logarithmically with the class count, so it stays bounded and avoids the gradient explosions of inverse-frequency reweighting. When classes are balanced it reduces to ordinary cross-entropy, as it should.

One caveat is that Balanced Softmax should not be stacked with a hand-set class-balanced sampler, because the loss already encodes the full rebalancing. Doing both would double-count and effectively produce a 1/n_j^2 weighting. The clean recipe is the loss with instance-balanced sampling.

```python
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def balanced_softmax_loss(labels, logits, sample_per_class, reduction='mean'):
    """Balanced Softmax cross-entropy.

    Trains logits to model the balanced-test posterior while training on the
    imbalanced split, by shifting each logit by +log(n_j) before standard
    cross-entropy. Derived from phi_hat_j = n_j e^{eta_j} / sum_i n_i e^{eta_i}.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)  # [batch, k]
    logits = logits + spc.log()                          # eta_j + log n_j
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


class BalancedSoftmax(_Loss):
    """Loss module holding per-class training counts n_1..n_k."""

    def __init__(self, sample_per_class):
        super().__init__()
        self.sample_per_class = torch.as_tensor(sample_per_class)

    def forward(self, logits, labels, reduction='mean'):
        return balanced_softmax_loss(
            labels, logits, self.sample_per_class, reduction
        )


# Test time: NO shift. The bare logits already model the balanced posterior.
@torch.no_grad()
def predict(model, x):
    return model(x).argmax(dim=-1)
```
