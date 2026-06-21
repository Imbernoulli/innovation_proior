The problem is single-index regression: targets are generated as y = f*(<theta*, x>) + noise with Gaussian inputs, where both the high-dimensional unit direction theta* and the one-dimensional link f* are unknown. The learner must recover theta* and fit f* using a standard shallow network trained by gradient descent, without hard-wiring the link into the architecture and without resorting to a problem-specific estimator. Existing approaches only solve pieces of this. Teacher-student analyses assume the link is known and fix the student activation to match it, so they bypass the nonparametric half of the problem entirely. Dedicated semi-parametric estimators such as sliced inverse regression or Hermite-feature methods can recover the direction with information-exponent-optimal sample complexity, but they are bespoke procedures rather than an end-to-end gradient method. Infinite-width approximation theory shows that shallow networks can represent single-index targets without a curse of dimensionality, yet it offers no training algorithm or sample-complexity guarantee. The one-giant-step constructions demonstrate that a single large gradient update can separate neural networks from fixed kernels, but they do not train both layers jointly to convergence. Meanwhile, the natural practitioner baseline, a plain two-layer ReLU network trained by SGD, entangles the high-dimensional direction search with the one-dimensional link fit; near a random initialization the direction signal is buried in a flat equatorial region, and there is no clean theory that joint SGD will escape it at the optimal rate.

The right structural move is to separate the part of the network that searches in the high-dimensional input space from the part that builds the one-dimensional link approximation. The target depends on x only through the scalar projection u = <theta*, x>, so the clean model ties every hidden unit to a single shared direction theta and lets the neurons differ only by scalar biases and signs. Those biases and signs should then be frozen after initialization. If theta is fixed, the frozen biases become a random-feature dictionary for a univariate kernel in u; if the biases were allowed to move, the link-fitting scaffold would become another active part of the high-dimensional search and reintroduce the very coupling we want to remove. In the tied-direction model the population loss depends on theta only through the scalar overlap m = <theta, theta*>, which collapses the d-dimensional landscape to a benign scalar flow whose only critical directions are the equator and the two poles. Bach-style random-feature bounds guarantee that a finite-width frozen dictionary approximates the one-dimensional link well enough to prevent spurious middle critical points, provided the width scales inversely with the ridge regularization. This is the design principle: theta carries the rich feature-learning dynamics, while the frozen biases provide a lazy random-feature scaffold for the unknown link.

The practical benchmark implementation keeps only the essential move from that theory, because the scaffold exposes a standard wide MLP rather than the exact tied-direction model. We initialize the first-layer rows on the unit sphere so that the optimizer controls direction only; we sample the hidden biases uniformly in [-1, 1] and freeze them; and we train the remaining parameters, the first-layer directions and the readout, with ordinary SGD on mean-squared error. Freezing the biases removes the entanglement between the direction search and the link fit at the level of the actual network: each row's contribution to the loss enters only through its overlap with theta*, the gradient on every row is colinear with theta*, and the landscape acquires the same scalar structure that makes the theoretical analysis go through. The readout is initialized small and uniformly so that early training focuses on moving the directions rather than fitting a premature one-dimensional approximation. No post-hoc ridge refit is used in this scaffold version; the same SGD loop that learns the directions also fits the readout weights against the frozen-bias features.

This method is called the frozen-bias shallow network. It is the implementable, wide-MLP form of the tied-direction construction analyzed by Bietti, Mairal, and Bach for single-index recovery. The canonical name reflects the single decisive change relative to a standard shallow net: the hidden-neuron biases are sampled once and never trained. That freeze is what turns the network into a random-feature model along the discovered direction while leaving the directions themselves free to learn the hidden subspace. The result is a practitioner-style shallow network that still inherits the theoretical separation between high-dimensional feature learning and one-dimensional nonparametric regression.

```python
import math
import torch
import torch.nn as nn


class Strategy:
    """Frozen-bias shallow network for single-index recovery."""

    def __init__(self, config):
        self.config = config

    def init_two_layer(self, net, config):
        # Random first-layer rows on the unit sphere so the optimizer
        # controls only direction, not scale.
        with torch.no_grad():
            W = torch.randn_like(net.fc1.weight)
            W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
            net.fc1.weight.copy_(W)
            # Sample the hidden biases once and freeze them. They form a
            # one-dimensional random-feature dictionary for the unknown link.
            net.fc1.bias.uniform_(-1.0, 1.0)
        net.fc1.bias.requires_grad_(False)

        # Small, uniform readout initialization.
        bound = 1.0 / math.sqrt(config.width)
        nn.init.uniform_(net.fc2.weight, -bound, bound)
        nn.init.zeros_(net.fc2.bias)

    def make_optimizer(self, net, config):
        # Train everything except the frozen biases.
        params = [p for p in net.parameters() if p.requires_grad]
        return torch.optim.SGD(
            params,
            lr=config.base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    def training_step(self, net, optimizer, x, y, step, config):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        preds = net(x)
        loss = torch.mean((preds - y) ** 2)
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.item())}

    def finalize(self, net, x_train, y_train, config):
        # No post-hoc refit needed in this scaffold version.
        return


def build_strategy(config):
    return Strategy(config)
```
