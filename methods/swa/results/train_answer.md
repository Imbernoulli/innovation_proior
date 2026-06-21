I propose the method Stochastic Weight Averaging, abbreviated SWA. The core observation is that the last iterate of a conventionally trained deep network is usually chosen to minimize the regularized training objective, but the surface that really matters at deployment is the test-error surface. Those two surfaces are close in weight space, yet they are not perfectly aligned. Because of this mismatch, the training-loss minimum can sit near the edge of the broad region where test error is low, and a small shift can push it into worse territory. SWA therefore does not try to find a deeper training-loss point. Instead, it tries to locate a wider, more central point inside the same good basin.

The strategy is easiest to understand from the geometry of late-training SGD. When the learning rate is decayed toward zero, the optimizer loses its exploration and collapses onto a single point. Averaging those late iterates then only smooths noise around that same point, which rarely helps generalization much. In contrast, if the learning rate stays non-negligible, SGD keeps moving around inside a connected low-loss region. Under the usual stochastic-process simplifications, constant-step SGD near a local optimum behaves like a stationary sampler whose covariance is shaped by the learning rate, batch size, curvature, and gradient noise. In high dimensions, typical samples from such a distribution lie near the periphery of an ellipsoid, while their arithmetic mean sits closer to the center. That means the average of an energetic trajectory is not merely a denoised last iterate; it is a qualitatively different point that can move inward toward a flatter, more representative location.

SWA exploits this fact deliberately. Starting from pretrained weights, which can come from a conventional training run or from a reduced-budget run, the algorithm continues training with either a constant high learning rate or a cyclical schedule that repeatedly jumps from a high rate to a low rate and back. The constant schedule gives the broadest exploration, while the cyclical schedule trades some exploration for checkpoints that are individually a bit better refined. Checkpoints are collected at the end of each epoch under the constant schedule, or at the bottom of each cycle under the cyclical schedule. An equal running average of these checkpoints is maintained, so storing every checkpoint is unnecessary. Algebraically, when a new checkpoint arrives, the average is updated by adding a fraction of the difference between the live weights and the current average.

A subtlety arises with networks that use batch normalization. The averaged weights never produced the activations that filled the running mean and variance buffers during training, so those buffers are not consistent with the averaged model. SWA fixes this with a single additional forward pass over the training data in training mode, recomputing the batch-normalization statistics for the averaged weights. After that pass, the returned model is a single network with updated normalization buffers, ready for inference at the cost of one model, not an ensemble.

The method can also be viewed as a one-model approximation to fast geometric ensembling. Suppose the collected checkpoints are nearby weights, and let the averaged weight be their centroid. If each checkpoint is expanded around the centroid, the first-order correction in a prediction averages to zero because the offsets sum to zero. The gap between averaging predictions and evaluating the averaged weights is therefore second order in the checkpoint spread. Meanwhile, different checkpoints can still disagree at first order, which is the source of ensemble benefit. As long as the explored region is small enough that second-order effects remain minor, SWA captures much of the ensemble gain without paying the storage or inference cost of multiple networks.

The train-to-test shift is exactly the scenario where this geometry pays off. The conventional endpoint may have marginally lower training loss but sit near a boundary of the good region, making it vulnerable to curvature directions that matter for test error. The SWA point lives in the same basin but closer to its center, so random perturbations or a modest shift between train and test surfaces are less damaging. Empirically, this translates into improved test accuracy on image-classification benchmarks such as CIFAR-10, CIFAR-100, and ImageNet across architectures including VGG, Preactivation ResNets, Wide ResNets, PyramidNets, DenseNets, and Shake-Shake networks.

The implementation is lightweight. It requires one extra copy of the model weights, a simple running-average update, a deliberately chosen late-learning-rate schedule, and a batch-normalization recomputation step. No architectural change, no loss change, and no multi-model inference are needed. The canonical name for the method is Stochastic Weight Averaging, and it is best applied as a short, high-learning-rate tail appended to ordinary training, followed by the equal average of the visited weights.

```python
import torch
import torch.nn as nn

def swa_lr(step, cycle_len, high_lr, low_lr=None):
    if low_lr is None:
        return high_lr
    t = ((step - 1) % cycle_len + 1) / cycle_len
    return (1.0 - t) * high_lr + t * low_lr

def train_swa_tail(model, loader, loss_fn, optimizer, tail_epochs,
                   high_lr, low_lr=None, cycle_len=None, device=None):
    if low_lr is not None and cycle_len is None:
        raise ValueError("cycle_len is required for cyclical SWA")

    swa_model = torch.optim.swa_utils.AveragedModel(model, device=device)
    step = 0
    model.train()

    for _ in range(tail_epochs):
        for x, y in loader:
            if device is not None:
                x, y = x.to(device), y.to(device)
            step += 1
            for group in optimizer.param_groups:
                group["lr"] = swa_lr(step, cycle_len or 1, high_lr, low_lr)
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()

            if low_lr is not None and step % cycle_len == 0:
                swa_model.update_parameters(model)

        if low_lr is None:
            swa_model.update_parameters(model)

    if int(swa_model.n_averaged) == 0:
        swa_model.update_parameters(model)

    torch.optim.swa_utils.update_bn(loader, swa_model, device=device)
    return swa_model


# Small self-contained illustration on a noisy quadratic bowl.
torch.manual_seed(0)
w_true = torch.tensor([2.0, -1.0])
w = torch.zeros(2, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)
swa = None
n_averaged = 0

for epoch in range(30):
    for _ in range(50):
        optimizer.zero_grad()
        noise = torch.randn(2) * 0.3
        loss = ((w - w_true + noise) ** 2).mean()
        loss.backward()
        optimizer.step()
    if epoch >= 15:
        with torch.no_grad():
            if swa is None:
                swa = w.detach().clone()
            else:
                swa += (w.detach() - swa) / (n_averaged + 1)
            n_averaged += 1

print("last iterate:", w.detach().tolist())
print("SWA average: ", swa.tolist())
print("true value:  ", w_true.tolist())
```
