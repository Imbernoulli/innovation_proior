## Research question

Before the first gradient step, every weight in a deep convolutional network is just a number drawn from
some distribution I choose. That choice alone decides whether the forward pass keeps a usable signal as it
passes through dozens of layers, whether the backward pass delivers a gradient that is neither vanishing nor
exploding, and — empirically — where 200 epochs of SGD finally land. The single thing being designed here is
the **initialization strategy**: a data-independent rule that fills `Conv2d`, `BatchNorm2d`, and `Linear`
parameters in place, given only the architecture and its depth. Everything else about training is frozen.
The rule has to hold across three different network families at once — a deep residual net, a plain
VGG-style stack, and an inverted-residual mobile net — so a strategy that quietly assumes one topology has
to earn its keep on the other two.

## Prior art before the first rung

The first rung reacts to the standard fan-scaling lineage that every modern initializer descends from. These
are the methods the ladder starts above; each fixed one failure of the last.

- **Small fixed-variance Gaussian (the scaffold default).** Draw every weight from `N(0, 0.01²)`. Simple,
  symmetry-breaking, depth-blind. Gap: the variance is unrelated to fan-in/out, so the per-layer signal
  shrinks geometrically with depth — a deep stack initialized this way starts almost dead, and on a network
  without BatchNorm to rescue it, never recovers.
- **LeCun / Xavier scaling (LeCun 1998; Glorot & Bengio 2010).** Set `Var(W) = 1/fan_in` (LeCun) or
  `2/(fan_in+fan_out)` (Xavier) to hold activation/gradient variance roughly constant across a *linear* or
  `tanh`-like layer. Gap: the derivation assumes a symmetric, unit-derivative-at-zero nonlinearity; with
  ReLU, which zeroes half its inputs, the forward variance is halved at every layer, so Xavier still decays
  with depth in a ReLU net.
- **Kaiming / He scaling (He et al. 2015, arXiv:1502.01852).** Repair the ReLU factor of two:
  `Var(W) = 2/fan`. This is the variance the residual and orthogonal rungs both fall back to as their
  per-layer scale. Gap: it controls *per-layer* variance but says nothing about how variance *accumulates*
  down a residual network's main path, nor about the conditioning of the backward Jacobian beyond its scale.
- **Norm-preservation via orthogonality (Saxe, McClelland & Ganguli 2014, arXiv:1312.6120).** Instead of
  matching only the second moment, make each weight matrix *orthogonal* so it preserves the entire norm of
  whatever vector passes through it, forward and backward — exact isometry in a deep linear net. Gap: the
  clean theory is for linear/square maps; ReLU and non-square conv reshapes perturb the isometry, and the
  scheme is silent about residual accumulation and about BatchNorm's own rescaling.

## The fixed substrate

The training loop, data pipeline, model definitions, and evaluation are frozen and must not be touched. Three
architectures are built by `build_model(arch, num_classes)` and handed to me fully constructed: a CIFAR
**ResNet-56** (`BasicBlock` ×27 across three stages, each block = two 3×3 convs with BN, an additive
shortcut, final `Linear` head), a **VGG-16-BN** (a plain `Conv-BN-ReLU` stack, no shortcuts, a two-layer
classifier with `Dropout`), and a **MobileNetV2** (inverted-residual blocks with depthwise 3×3 convs,
pointwise 1×1 expansions/projections, BN throughout, additive shortcuts only when stride 1 and channels
match). Every conv that is followed by BN has `bias=False`. Optimization is fixed: SGD `lr=0.1`,
`momentum=0.9`, `weight_decay=5e-4`, cosine-annealed over 200 epochs, cross-entropy loss, batch 128, with
`RandomCrop(32, pad=4)` + `RandomHorizontalFlip` augmentation. The metric is the best test accuracy reached
during those 200 epochs.

## The editable interface

Exactly one function is editable — `initialize_weights(model, config)` in `pytorch-vision/custom_init.py`
(lines 228–261). It receives the constructed model and a `config` dict and must initialize **all**
parameters in place, with no return value. The contract:

- `config['arch']` — `'resnet56'`, `'vgg16bn'`, or `'mobilenetv2'`; `config['num_classes']` — 10 or 100;
  `config['depth']` — total count of `Conv2d` + `Linear` layers.
- I may iterate `model.modules()` / `model.named_modules()` / `model.named_parameters()`, branch on module
  type or on `arch`, treat shortcut projections differently from main-path convs, set `BatchNorm2d`
  weight/bias however I like, and use any depth-dependent arithmetic. The module classes `BasicBlock`,
  `InvertedResidual`, `nn.Conv2d`, `nn.BatchNorm2d`, `nn.Linear` are all in scope.
- **Hard constraints:** the rule must be **data-independent** — no access to training data, no calibration
  forward passes — and must not alter the model graph, change optimizer hyperparameters, or touch evaluation.

The starting point is the scaffold default: a tiny fixed-variance Gaussian on every weight, BN at
`(weight=1, bias=0)`. Each method on the ladder replaces exactly this function body and nothing else.

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

Three (architecture, dataset) pairs, each trained from scratch for the full 200-epoch schedule on seed 42:
**ResNet-56 on CIFAR-100**, **VGG-16-BN on CIFAR-100**, and **MobileNetV2 on FashionMNIST** (grayscale
repeated to three channels, resized to 32×32). One metric, higher is better: best test accuracy (%) reached
during training, reported per (arch, dataset).
