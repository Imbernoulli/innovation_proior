## Research Question

Standard deep-network training follows noisy gradients while steadily shrinking the learning rate, then returns the final weight vector. The practical question is whether the training tail can be used to return a single network that performs better at test time, without changing the architecture, loss, dataset, or optimizer family, and without maintaining multiple networks for evaluation.

## Geometric Clues

Several facts are already on the table. First, width is plausibly tied to generalization: large-batch training can land in sharp regions that generalize worse, while small-batch noise tends to avoid some sharp basins; local-entropy methods explicitly try to bias optimization toward wide valleys. Second, sharpness is not simply "steep in every direction"; a solution can be flat in most directions but have a few directions of sharp ascent.

Third, neural-network nonconvexity does not force the useful region to be a collection of isolated points. Independently trained optima can be joined by simple curves of near-constant loss, and nearby points on such paths can still make meaningfully different predictions. This makes it plausible that a training trajectory can move through a connected low-loss region rather than only converge to a single isolated endpoint.

Fourth, a non-vanishing learning rate changes the late-training picture. Under simplifying stochastic-process assumptions, constant-step SGD behaves like a stationary sampler around a local optimum, with covariance controlled by learning rate, minibatch size, curvature, and gradient noise. In high dimension, typical samples from such a distribution lie near a shell of an ellipsoid rather than at its center.

## Existing Baselines

Conventional decayed-learning-rate training returns the last iterate.

Classical trajectory averaging in stochastic approximation averages iterates to improve convergence-rate or asymptotic variance. In neural-network practice, a related exponential moving average is often paired with a decaying learning rate.

Fast geometric ensembling uses a cyclical learning rate to collect nearby but diverse checkpoints, then averages their predictions at test time.

Cyclical learning-rate training alternates exploration and lower-rate refinement.

## Evaluation Setting

The relevant empirical setting is image classification on CIFAR-10, CIFAR-100, and ImageNet, with architectures that were already standard in this line of work: VGG, Preactivation ResNets, Wide ResNets, PyramidNets, DenseNets, and Shake-Shake networks. Comparisons are against conventional SGD training and against fast geometric ensembling under matched or clearly accounted training budgets.

The diagnostic geometry is as important as accuracy. One should inspect train loss and test error along random rays from a candidate solution and along the line connecting the candidate to the conventional endpoint.

## Code Scaffold

The fixed substrate is an ordinary PyTorch training loop. The open slots are the late learning-rate schedule, the rule for collecting or summarizing late-training weights, and the batch-normalization treatment for whatever final weights are returned.

```python
import torch

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr

def tail_lr(step, cycle_len, high_lr, low_lr=None):
    # TODO: choose the late-training schedule.
    raise NotImplementedError

def train_tail(model, loader, loss_fn, optimizer, tail_epochs,
               high_lr, low_lr=None, cycle_len=None, device=None):
    model.train()
    step = 0
    for _ in range(tail_epochs):
        for x, y in loader:
            if device is not None:
                x, y = x.to(device), y.to(device)
            step += 1
            set_lr(optimizer, tail_lr(step, cycle_len or 1, high_lr, low_lr))
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()

    # TODO: decide which single set of weights should be returned.
    return model
```
