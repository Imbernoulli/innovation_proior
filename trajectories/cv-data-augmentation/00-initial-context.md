## Research question

Train-time data augmentation is a cheap regularizer for image classification: it costs nothing at inference, touches no weights, and on small natural-image datasets it often improves accuracy more than architectural changes of comparable effort. The design object is the **training-image transform** — the label-preserving perturbation applied to each example as it is loaded. Everything else is frozen: the model, the optimizer, the schedule, the initialization, and the no-augmentation test transform. The question is which transform pipeline, returned as a single `torchvision.transforms.Compose`, generalizes best across three regimes at once — a small ResNet on CIFAR-10, a deeper ResNet on CIFAR-100, and a MobileNetV2 on FashionMNIST.

## Prior art / Background / Baselines

- **Standard geometric augmentation (Krizhevsky et al. 2012; the CIFAR convention).** It pads the image by 4 pixels and crops back a random 32×32 window, then mirrors with probability 0.5. Gap: it only perturbs geometry, so it covers only a narrow slice of plausible image variations and leaves achievable regularization on the table.
- **AutoAugment (Cubuk et al. 2019).** It searches a policy — five sub-policies, each two operations, each operation a (transform type, probability, magnitude) triple — by training an RNN controller with reinforcement learning on a small proxy task, then transfers the winning policy to the target. Gap: it adds a full second optimization loop, a search space near 10^32, and the proxy task gives no direct way to set augmentation strength for the target model and dataset.
- **Population-Based Augmentation / Fast AutoAugment (Ho et al. 2019; Lim et al. 2019).** They use cheaper search procedures — population training or density matching — for the same policy family. Gap: they still require a search phase and per-transform probabilities and magnitudes; the cost drops but the mismatch between the proxy search and the target task remains.

## Fixed substrate / Code framework

The training loop, models, and optimizer are frozen. The models are textbook small-image networks: a CIFAR ResNet (3×3 stem, no max-pool, global average pool; ResNet-20 uses `[3,3,3]` blocks, ResNet-56 uses `[9,9,9]`) and a CIFAR-adapted MobileNetV2 (stride-1 stem, width 1.0). Weights use Kaiming-normal initialization. Optimization is SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`, cosine-annealed over 200 epochs, batch size 128, cross-entropy loss; the reported number is the best test accuracy seen during training.

Data loading is fixed: the loop builds the train transform by calling `build_train_transform(config)`, builds a fixed test transform (`ToTensor` + `Normalize`, no augmentation), and serves minibatches. For FashionMNIST the loop wraps the returned pipeline: it prepends `Resize(32)` and, after the pipeline's `ToTensor`, repeats the single channel to three channels. So any tensor-space transform placed after `ToTensor` sees a 1-channel tensor on FashionMNIST, and any PIL-space transform sees a resized grayscale image.

## Editable interface

Only one region is editable — the `build_train_transform(config)` function in `pytorch-vision/custom_augment.py` (lines 246–275). The contract is rigid: it receives a `config` dict with `img_size` (int, `32`), `mean` and `std` (per-channel tuples), and `dataset` (`'cifar10'`/`'cifar100'`/`'fmnist'`), and it must return a `transforms.Compose`. The returned pipeline **must** end with `transforms.ToTensor()` and `transforms.Normalize(config['mean'], config['std'])`. It may not read test labels, change the split, or alter any model/optimizer code. Dataset-specific behavior is allowed.

PIL-space ops go before `ToTensor`; tensor-space ops go after `ToTensor` and before `Normalize`. Custom transform classes may be defined inside the function. The starting point is the scaffold default: the minimal geometric baseline.

```python
# EDITABLE region of custom_augment.py (lines 246-275) -- default fill
def build_train_transform(config):
    """Build training data transform pipeline.

    Must return a complete transforms.Compose pipeline including
    ToTensor() and Normalize(config['mean'], config['std']).
    config: {img_size:int, mean:tuple, std:tuple, dataset:str}
    """
    return transforms.Compose([
        transforms.RandomCrop(config['img_size'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']),
    ])
```

## Evaluation settings

Three (architecture, dataset) pairs span the regime: **ResNet-20 on CIFAR-10** (small model, 10 classes, 50k train), **ResNet-56 on CIFAR-100** (deeper model, 100 classes, the hardest of the three — least data per class, most room for a regularizer to help), and **MobileNetV2 on FashionMNIST** (a different architecture family on a grayscale dataset, served through the resize-and-repeat wrapper above). One seed (`42`).

The single metric, on every pair, is **best test accuracy** (%, higher is better) achieved during the 200-epoch run. A good augmentation must hold up across all three at once; a transform tuned to one regime that hurts another is not a win.
