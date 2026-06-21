## Research question

Train deep convolutional image classifiers with plain SGD. The optimizer, model, augmentation, weight decay, and 200-epoch budget are all fixed. The only remaining design choice is the per-epoch learning rate: a function `eta(epoch)` over the run. The question is which `eta(epoch)` curve gives the best test accuracy across three different settings at once: a shallow residual net on an easy 10-class problem, a deep residual net on a hard 100-class problem, and an inverted-residual mobile net on a grayscale 10-class problem.

## Prior art / Background / Baselines

The standard ways people currently set `eta(epoch)` are listed below. Each fills the same `eta(epoch)` slot.

- **Constant rate.** Hold `eta = base_lr` for the whole run.
- **Step / multi-step decay.** Hold a high plateau and divide the rate by a fixed factor at hand-picked epochs (for example, multiply by 0.2 at epochs 60/120/160 of a 200-epoch run). The milestones and drop factor are free numbers tied to the pre-fixed budget.
- **Cyclical (triangular) learning rates.** Ramp the rate linearly up to a `max_lr` and back down, repeatedly, between two fixed bounds.

## Fixed substrate / Code framework

A single training file is frozen and must not be touched. It builds the model by architecture name (CIFAR-adapted ResNet-20/56 with `BasicBlock`, VGG-16-BN, MobileNetV2 with inverted-residual blocks), applies Kaiming-normal init, loads CIFAR-10/100/FashionMNIST with `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`, and runs a plain SGD loop. The optimizer is created once as `SGD(lr=base_lr=0.1, momentum=0.9, weight_decay=5e-4)` and there is **no** PyTorch scheduler object: each epoch the loop calls the editable function, writes the returned float into every parameter group's `lr`, trains one epoch, evaluates, and tracks the best test accuracy seen. Momentum (0.9) and weight decay (5e-4) are fixed and never re-set by the loop. The loop passes the schedule only `epoch`, `total_epochs`, `base_lr`, and a `config` dict.

## Editable interface

Exactly one region is editable — the body of `get_lr(epoch, total_epochs, base_lr, config)` in `pytorch-vision/custom_schedule.py` (the bracketed region, lines 246–269). The contract: it is called once per epoch with the 0-indexed `epoch` (0 … `total_epochs`−1), the budget `total_epochs` (200), the reference rate `base_lr` (0.1), and `config = {'arch': ..., 'dataset': ...}` (`arch` ∈ {`resnet20`, `resnet56`, `vgg16bn`, `mobilenetv2`}, `dataset` ∈ {`cifar10`, `cifar100`, `fmnist`}); it returns the single float used as the learning rate for that whole epoch. The float is free — it may exceed `base_lr` or fall to 0 — but it is the *only* lever: momentum and weight decay stay at their fixed values, and the rate is constant within an epoch (the loop sets it once per epoch, not per batch). `math` is imported at module top, so `math.cos`/`math.pi` are available.

The starting point is the scaffold default: a **constant** rate. Each method replaces only this function body and nothing else.

```python
# EDITABLE region of pytorch-vision/custom_schedule.py (lines 246-269) -- default fill
def get_lr(epoch, total_epochs, base_lr, config):
    """Compute learning rate for the given epoch.

    Called once per epoch to set the learning rate for all parameter groups.

    Args:
        epoch: current epoch (0-indexed, ranges from 0 to total_epochs-1)
        total_epochs: total number of training epochs
        base_lr: initial learning rate (from --lr flag, default 0.1)
        config: dict with keys:
            - arch: str ('resnet20', 'resnet56', 'vgg16bn', 'mobilenetv2')
            - dataset: str ('cifar10', 'cifar100', 'fmnist')

    Returns:
        float: learning rate to use for this epoch
    """
    return base_lr  # constant LR (no schedule)
```

## Evaluation settings

Three fixed (architecture, dataset) pairs, each trained for 200 epochs from Kaiming init with the frozen SGD loop, on a single seed (42): **ResNet-20 on CIFAR-10** (shallow net, easy 10-class), **ResNet-56 on CIFAR-100** (deep net, hard 100-class), and **MobileNetV2 on FashionMNIST** (inverted-residual mobile net, grayscale 10-class, replicated to 3 channels). The same `get_lr` is applied to all three. The metric is **best test accuracy (%, higher is better)** reached during training, reported per setting; the schedule must not modify model code, augmentation, loss, optimizer type, momentum, weight decay, or evaluation.
