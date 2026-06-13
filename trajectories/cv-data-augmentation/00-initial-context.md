## Research question

Train-time data augmentation is the cheapest regularizer that exists for image classification: it costs nothing at inference, touches no weights, and on small natural-image datasets it routinely buys more accuracy than any architectural change of comparable effort. The single thing being designed here is the **training-image transform** — the label-preserving perturbation applied to each example as it is loaded. Everything else about the run is frozen: the model, the optimizer, the schedule, the initialization, and the (no-augmentation) test transform. The question is which transform pipeline, returned as a single `torchvision.transforms.Compose`, generalizes best across three different (architecture, dataset) regimes at once — a small ResNet on CIFAR-10, a deeper ResNet on the harder CIFAR-100, and a MobileNetV2 on FashionMNIST.

## Prior art before the first rung (the augmentation lineage)

The first rung reacts to a specific line of work — automated, per-image augmentation policies — and to the minimal geometric baseline they were built to beat. These precede the ladder; each carries the gap that the next move is meant to close.

- **Standard geometric augmentation (Krizhevsky et al. 2012; the CIFAR convention).** `RandomCrop(32, padding=4)` plus `RandomHorizontalFlip` — pad the image by 4 pixels and crop back a random 32×32 window, then mirror with probability 0.5. Cheap, label-preserving, the de-facto floor for every CIFAR recipe. Gap: it only perturbs *geometry* (position and chirality); it injects no photometric variation and no occlusion, so it leaves a large amount of achievable regularization on the table.
- **AutoAugment (Cubuk et al. 2019).** Search for a *policy* — five sub-policies, each two operations, each operation a (transform type, probability, magnitude) triple — by training an RNN controller with reinforcement learning on a small proxy task, then transfer the winning policy to the target. Real accuracy gains, zero inference cost. Gap: a whole second optimization loop (controller + RL + proxy), a search space near 10^32, and the proxy is structurally wrong for the one thing that matters most — augmentation *strength* scales with model and dataset size, so a strength pinned on a small proxy is mis-set on the real target, with no handle to re-tune it.
- **Population-Based Augmentation / Fast AutoAugment (Ho et al. 2019; Lim et al. 2019).** Cheaper searches for the same kind of policy — schedules of magnitudes, density matching. Gap: still a search phase, still per-transform probabilities and magnitudes to fit; the cost comes down but the structural objection (you are searching for knobs that may not carry the benefit) remains.

The common thread: these methods spend enormous machinery learning *which* operations to apply and *with what probability*, on the bet that the specific learned probabilities are what help. The ladder below tests the opposite hypothesis — that the benefit is *diversity*, not the learned policy — and then asks whether a structurally different regularizer (deleting information rather than diversifying it) does better still.

## The fixed substrate

A complete CIFAR-style training loop is frozen and must not be touched. The models are textbook small-image networks: a CIFAR ResNet (3×3 stem, no max-pool, global average pool — ResNet-20 is `[3,3,3]` blocks, ResNet-56 is `[9,9,9]`) and a CIFAR-adapted MobileNetV2 (stride-1 stem, width 1.0). Weights get Kaiming-normal init. Optimization is SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`, cosine-annealed over `200` epochs, batch size 128, cross-entropy loss; best test accuracy seen during training is the reported number. Data loading is fixed: the loop builds the *train* transform by calling `build_train_transform(config)`, builds a fixed *test* transform (just `ToTensor` + `Normalize`, no augmentation), downloads nothing, and serves minibatches. One harness detail is load-bearing for FashionMNIST: it is grayscale, so the loop wraps the returned train pipeline — it inserts `Resize(32)` at the front and, immediately after the pipeline's `ToTensor`, a channel-repeat to 3 channels — meaning any tensor-space transform in the returned pipeline that runs *after* `ToTensor` sees a 1-channel tensor on FashionMNIST, and any PIL-space transform sees a resized grayscale PIL image.

## The editable interface

Exactly one region is editable — the `build_train_transform(config)` function in `pytorch-vision/custom_augment.py` (lines 246–275). The contract is rigid: it receives a `config` dict with `img_size` (int, `32`), `mean` and `std` (per-channel tuples), and `dataset` (`'cifar10'`/`'cifar100'`/`'fmnist'`), and it must return a `transforms.Compose`. The returned pipeline **must** end with `transforms.ToTensor()` and `transforms.Normalize(config['mean'], config['std'])` so the downstream models receive normalized tensors; it may not read test labels, change the split, or alter any model/optimizer code. Dataset-specific behavior is allowed. Every rung on the ladder is a different fill of this one function — PIL-space ops (geometric, photometric, automated policies) go before `ToTensor`, tensor-space ops (cutout, random erasing) go after it and before `Normalize`. Custom transform classes may be defined inside the function.

The starting point is the scaffold default: the minimal geometric baseline.

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

Three (architecture, dataset) pairs span the regime: **ResNet-20 on CIFAR-10** (small model, 10 classes, 50k train), **ResNet-56 on CIFAR-100** (deeper model, 100 classes, the hardest of the three — least data per class, most room for a regularizer to help), and **MobileNetV2 on FashionMNIST** (a different architecture family on a grayscale dataset, served through the resize-and-repeat wrapper above). One seed (`42`). The single metric, on every pair, is **best test accuracy** (%, higher is better) achieved during the 200-epoch run. A good augmentation must hold up across all three at once; a transform tuned to one regime that hurts another is not a win.
