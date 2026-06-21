# Stochastic Weight Averaging

Stochastic Weight Averaging (SWA) keeps SGD exploring a good region with a high constant or cyclical learning rate, then returns the equal average of the visited weights. The point of the high-learning-rate tail is essential: it creates a spread of nearby, high-performing checkpoints whose centroid lies in a wider, more central part of the same basin.

## Method

Start from pretrained weights `w_hat`. Continue training with either:

```text
constant: alpha(i) = alpha_1

cyclical: alpha(i) = (1 - t(i)) alpha_1 + t(i) alpha_2
          t(i) = (mod(i - 1, c) + 1) / c
```

For the cyclical schedule, jump directly from `alpha_2` back to `alpha_1` at the next cycle. Collect a checkpoint at each cycle minimum; for the constant schedule, collect once per epoch. Maintain the running equal average:

```text
w_swa <- (w_swa * n_models + w) / (n_models + 1)
```

After training, recompute batch-normalization running statistics with one forward pass over the training loader using `w_swa`.

## Why It Works

With a non-vanishing learning rate, SGD continues to traverse the periphery of a good region. Under the constant-LR sampling picture, high-dimensional samples concentrate near an ellipsoid shell, while their mean moves inward. This makes the average a central point rather than just a smoothed last iterate.

SWA is also a one-model approximation to fast geometric ensembling. For nearby checkpoints `w_i`, set `w_swa = (1/n) sum_i w_i` and `Delta_i = w_i - w_swa`, so `sum_i Delta_i = 0`. Linearizing a scalar prediction `f` at `w_swa`,

```text
(1/n) sum_i f(w_i) - f(w_swa) = O(max_i ||Delta_i||^2)
```

because the first-order terms cancel. The checkpoints themselves can still differ at first order, so SWA keeps much of the ensemble benefit while returning one model.

## Code

```python
import torch

def _set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr

def swa_lr(step, cycle_len, high_lr, low_lr=None):
    if low_lr is None:
        return high_lr
    if cycle_len <= 0:
        raise ValueError("cycle_len must be positive")
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
            _set_lr(optimizer, swa_lr(step, cycle_len or 1, high_lr, low_lr))
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
```
