## Research question

Stochastic first-order optimization for non-convex machine-learning models. Each step exposes only a noisy mini-batch gradient, yet parameters differ widely in gradient scale and curvature. The only design freedom is the **update rule** — the auxiliary state it keeps and how it turns the raw gradient into a parameter step. Everything else — model, dataset, loss, batch size, epoch budget, and learning-rate schedule — is fixed.

## Prior art / Background / Baselines

- **SGD.** Take a fixed-size step in the negative mini-batch gradient direction using a single global step size.
- **Classical momentum.** Maintain a running average of past gradients to smooth the trajectory and accelerate along consistent directions.
- **AdaGrad.** Accumulate per-coordinate squared gradients and normalize each update by the square root of that history, giving each coordinate its own effective step size.
- **RMSprop.** Replace AdaGrad's cumulative sum with an exponential moving average of squared gradients, adapting the per-coordinate step size to recent gradient scale.

## Fixed substrate / Code framework

A single training driver is frozen and must not be touched. It builds a small convnet, loads CIFAR-10 with standard augmentation, and runs a fixed epoch budget with a fixed batch size and cosine learning-rate schedule. The optimizer is the only editable object; the driver calls `optimizer.zero_grad()` before the backward pass and `optimizer.step()` after.

## Editable interface

Exactly one region is editable: the `Optimizer` class in `custom_optimizer.py`. The contract is three methods:

- `__init__(self, params, lr, **kwargs)` — store the parameters and any hyperparameters the update rule needs.
- `step(self)` — read `p.grad` for each parameter and apply one in-place update.
- `zero_grad(self)` — clear the gradients stored on the parameters.

Every method on the ladder is a fill of this same contract. The starting point is the scaffold default: **vanilla SGD**.

```python
# EDITABLE region of custom_optimizer.py -- default fill: vanilla SGD
class Optimizer:
    """Stochastic optimizer. Default: vanilla SGD. Replace with your method."""

    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p.data.add_(p.grad, alpha=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

## Evaluation settings

CIFAR-10 classification with a small convnet, fixed training budget of 100 epochs, batch size 128, and cosine annealing with linear warmup. Primary metric: **best test accuracy** (higher is better) over seeds {42, 123, 456}. Secondary metric: final validation loss (lower is better). All settings share the fixed model and training recipe above.
