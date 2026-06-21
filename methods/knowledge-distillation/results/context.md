## Research question

The setting is large-scale supervised classification with deep neural networks. The best accuracy on hard recognition problems (ImageNet-scale object recognition, large-vocabulary speech recognition, MNIST) is obtained either by a single very large, heavily regularized network, or by an *ensemble* of many separately trained networks whose predictions are averaged. Both are expensive to deploy: an ensemble multiplies the inference cost by the number of members, and a single huge net carries millions of parameters and a large memory/latency footprint. The training stage and the deployment stage have very different requirements — training can afford huge compute and offline batch processing, deployment is latency- and resource-constrained — yet the same model is typically used for both.

Can we transfer the *function* that a cumbersome model has learned into a single, much smaller and faster model, with most of the accuracy preserved? The crux is what "the knowledge" of a trained model even *is* — if we identify it with the specific learned parameter values, changing the architecture seems to destroy it, but a more useful abstraction is that the knowledge is the *learned mapping from inputs to a distribution over outputs*, which is independent of any particular parameterization.

## Background

- **The cumbersome model.** State-of-the-art systems are large networks trained with strong regularizers — dropout (Srivastava et al. 2014) can be seen as training an exponentially large ensemble of weight-sharing models — or explicit ensembles whose member predictive distributions are averaged (arithmetic or geometric mean). Their normal training objective is to maximize the average log probability of the correct class via a softmax output layer, $q_i = \mathrm{softmax}(z)_i$.
- **The cumbersome model's full output.** Trained with a softmax output layer, the cumbersome model returns a full distribution over classes for each input, not just its top prediction. On hard recognition tasks it is typically confidently correct, so almost all of the mass sits on the winning class; the remaining classes receive very small probabilities, often many orders of magnitude apart on a task like MNIST.
- **Caruana et al. (model compression).** Bucilă, Caruana & Niculescu-Mizil (2006) already demonstrated that the function learned by a large ensemble can be compressed into a single small net by training the small net on data labeled by the ensemble (optionally a large pool of unlabeled "transfer" data synthesized for the purpose). To get around the vanishing-influence problem of near-zero probabilities, they trained the small model to match the ensemble's **logits** (pre-softmax inputs) under a squared-error loss, rather than its probabilities.

## Baselines

- **Training the small model directly on hard labels.** Fit the small net to one-hot targets with ordinary cross-entropy.
- **Caruana-style logit matching.** Train the small net to minimize the squared difference between its logits and the cumbersome model's logits. Logits are unbounded, so small-probability information survives across the full class distribution.
- **Strong regularization of the small model alone** (dropout, weight constraints, input jitter / data augmentation). Improves the small model's own training without a cumbersome teacher.

## Evaluation settings

- **MNIST.** 60,000 training images; a large two-hidden-layer net (1200 ReLU units/layer) with dropout, weight constraints and ±2-pixel input jitter as the cumbersome model; smaller two-hidden-layer nets (e.g. 800 or 300 or 30 units/layer) as the deployment candidate. Metric: test errors (count). Includes a transfer-set ablation where some digit classes are *omitted* entirely from the transfer set to test generalization to unseen classes.
- **Automatic speech recognition.** A deep acoustic model predicting HMM states from spectrogram frames; baseline is an ensemble of many such DNNs; metrics are frame classification accuracy and Word Error Rate (WER) on a held-out test set.
- **Protocol knobs.** The transfer set may be the original training set or a separate (possibly unlabeled) set; the output representation used as the training signal; the relative weight on any true-label objective; whether to also supply true labels.

## Code framework

PyTorch-style scaffold: ordinary networks, a softmax, a cross-entropy loss, an optimizer, and a training loop.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Net(nn.Module):
    """A generic classification network producing logits z (pre-softmax)."""
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.body = nn.Sequential(*layers[:-1])  # drop last ReLU; output = logits
    def forward(self, x):
        return self.body(x.flatten(1))

# A large, heavily regularized classifier is assumed already trained.
large_model = Net([784, 1200, 1200, 10])   # trained elsewhere with dropout etc.
small_model = Net([784, 800, 800, 10])

def transfer_loss(small_logits, large_logits, hard_targets=None):
    # TODO: define the transfer objective.
    raise NotImplementedError

opt = torch.optim.SGD(small_model.parameters(), lr=0.1, momentum=0.9)
for x, y in transfer_loader:          # transfer set: same or separate data
    with torch.no_grad():
        large_logits = large_model(x) # large model logits, fixed
    small_logits = small_model(x)
    loss = transfer_loss(small_logits, large_logits, y)   # TODO body above
    opt.zero_grad(); loss.backward(); opt.step()
# At deployment the small model runs an ordinary softmax.
```
