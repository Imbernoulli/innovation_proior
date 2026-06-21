## Research question

Before the first gradient step, every weight in a deep convolutional network is a number drawn from a distribution I choose. That choice governs whether the forward signal stays usable through many layers, whether backpropagated gradients vanish or explode, and — empirically — where the optimizer ends up after the full training schedule. The object of study is an **initialization strategy**: a data-independent rule that fills `Conv2d`, `BatchNorm2d`, and `Linear` parameters in place, given only the architecture and its depth. The rule must work across three network families — a deep residual net, a plain VGG-style stack, and an inverted-residual mobile net — so any assumption tied to one topology has to prove itself on the other two.

## Prior art / Background / Baselines

- **Small fixed-variance Gaussian (the scaffold default).** Draw every weight independently from `N(0, 0.01²)`. Gap: the variance is not scaled by fan-in, so signals and gradients shrink geometrically with depth; deep stacks start almost dead.
- **LeCun / Xavier scaling (LeCun 1998; Glorot & Bengio 2010).** Set `Var(W) = 1/fan_in` or `2/(fan_in+fan_out)` to keep activation/gradient variance stable across linear or `tanh`-like layers. Gap: the derivation assumes a symmetric nonlinearity with unit derivative at zero; with ReLU, which zeroes half its inputs, forward variance halves at every layer and the signal still decays.
- **Kaiming / He scaling (He et al. 2015).** Set `Var(W) = 2/fan` to account for ReLU zeroing half its inputs. Gap: it calibrates per-layer variance but does not address how variance accumulates across many layers or through the full network.
- **Norm-preservation via orthogonality (Saxe et al. 2014).** Initialize each weight matrix to be orthogonal so it preserves vector norms exactly in deep linear networks. Gap: the isometry breaks under ReLU and non-square conv reshapes, and the construction does not account for batch normalization.

## Fixed substrate / Code framework

The training loop, data pipeline, model definitions, and evaluation are frozen. Three architectures are built by `build_model(arch, num_classes)`: a CIFAR **ResNet-56** (`BasicBlock` ×27 across three stages, each block = two 3×3 convs with BN, an additive shortcut, final `Linear` head), a **VGG-16-BN** (plain `Conv-BN-ReLU` stack, no shortcuts, two-layer classifier with `Dropout`), and a **MobileNetV2** (inverted-residual blocks with depthwise 3×3 convs, pointwise 1×1 expansions/projections, BN throughout, additive shortcuts only when stride 1 and channels match). Every conv followed by BN has `bias=False`. Optimization is fixed: SGD `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`, cosine-annealed over 200 epochs, cross-entropy loss, batch 128, with `RandomCrop(32, pad=4)` + `RandomHorizontalFlip` augmentation. The metric is the best test accuracy reached during those 200 epochs.

## Editable interface

Exactly one function is editable — `initialize_weights(model, config)` in `pytorch-vision/custom_init.py` (lines 228–261). It receives the constructed model and a `config` dict and must initialize **all** parameters in place, with no return value. The contract:

- `config['arch']` — `'resnet56'`, `'vgg16bn'`, or `'mobilenetv2'`; `config['num_classes']` — 10 or 100; `config['depth']` — total count of `Conv2d` + `Linear` layers.
- I may iterate `model.modules()` / `model.named_modules()` / `model.named_parameters()`, branch on module type or on `arch`, treat shortcut projections differently from main-path convs, set `BatchNorm2d` weight/bias as desired, and use any depth-dependent arithmetic. The module classes `BasicBlock`, `InvertedResidual`, `nn.Conv2d`, `nn.BatchNorm2d`, `nn.Linear` are in scope.
- **Hard constraints:** the rule must be **data-independent** — no access to training data, no calibration forward passes — and must not alter the model graph, change optimizer hyperparameters, or touch evaluation.

The starting point is the scaffold default: a tiny fixed-variance Gaussian on every weight, BN at `(weight=1, bias=0)`. Each method replaces exactly this function body and nothing else.

```python
# EDITABLE region of pytorch-vision/custom_init.py (lines 228-261) — default fill
def initialize_weights(model, config):
    """Initialize all weights in the vision model.

    config: {'arch': str, 'num_classes': int, 'depth': int}
    Layer types: nn.Conv2d (bias=False under BN), nn.BatchNorm2d, nn.Linear.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
```

## Evaluation settings

Three (architecture, dataset) pairs, each trained from scratch for the full 200-epoch schedule on seed 42: **ResNet-56 on CIFAR-100**, **VGG-16-BN on CIFAR-100**, and **MobileNetV2 on FashionMNIST** (grayscale repeated to three channels, resized to 32×32). One metric, higher is better: best test accuracy (%) reached during training, reported per (arch, dataset).
