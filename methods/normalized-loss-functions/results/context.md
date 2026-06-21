## Research Question

Train a high-capacity softmax classifier when some training labels are wrong, changing only the
per-example loss. The labels are one-hot class targets, the model outputs logits `z` and probabilities
`p(k|x) = softmax(z)_k`, and the training loop simply evaluates a loss on `(logits, labels)` and
backpropagates its minibatch mean. What loss function design handles corrupted labels in this setting?

## Noise-Tolerance Lever

For multiclass label noise, a known sufficient condition is a constant class-sum loss:

```text
sum_{j=1}^K L(f(x), j) = C
```

with `C` independent of `x` and `f`. Under uniform noise rate `eta`, where the true class is kept with
probability `1 - eta` and each other class is chosen with probability `eta/(K - 1)`, this condition makes
the noisy risk an affine transform of the clean risk:

```text
R_eta(f) = (1 - eta*K/(K - 1)) R(f) + C*eta/(K - 1).
```

When `eta < (K - 1)/K`, the coefficient of `R(f)` is positive, so clean-risk minimizers remain noisy-risk
minimizers.

## Existing Losses

Cross entropy, `CE = -log p_y`, trains deep networks effectively because it gives large updates to hard
examples, but it is not symmetric. Mean absolute error is symmetric because its class-sum is constant,
and its gradients are uniform across classes. Reverse cross entropy is also symmetric after replacing `log 0`
with a finite negative constant.

Two combinations of these have been studied. Generalized cross entropy interpolates between CE and MAE
via a parameter `q`. Symmetric cross entropy adds a reverse-cross-entropy term to ordinary CE, combining
the learning behavior of each component.

## Evaluation Frame

The relevant experiments are ordinary image classifiers under synthetic and real label noise: MNIST,
CIFAR-10, CIFAR-100, and WebVision-style web labels. Synthetic noise includes uniform label flips and
class-conditional flips. The metric is clean validation or test accuracy.

## Code Frame

The scaffold exposes a single swappable loss object. It receives logits, integer labels, and the epoch,
and returns one scalar tensor:

```python
import torch
import torch.nn.functional as F


class RobustLoss:
    def compute_loss(self, logits, labels, epoch):
        return F.cross_entropy(logits, labels)
```

The implementation must therefore be a drop-in PyTorch loss. It may use the number of classes and scalar
loss weights, but it should not assume a different model, dataloader, target format, or training loop.
