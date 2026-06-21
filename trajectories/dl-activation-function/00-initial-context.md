## Research question

Design the pointwise nonlinearity for deep convolutional networks so that test accuracy rises across three architecture/dataset pairs while everything else is frozen. The only designed object is the activation curve — the element-wise map applied after each Conv–BatchNorm pair and in the classifier head. The optimizer, learning-rate schedule, weight initialization, data augmentation, and model definitions are fixed; only the shape of the nonlinearity may change.

## Prior art / Background / Baselines

Current pointwise nonlinearities:

- **ReLU** (`max(x,0)`): cheap, sparse, and has unit gradient for positive inputs. Outputs zero on the negative half-line.
- **Leaky ReLU / PReLU** (`max(x, px)` with a small constant or learned per-channel slope `p`): allows a small gradient on the negative side; piecewise-linear and monotonic.
- **ELU** (`x` for `x≥0`, `α(eˣ−1)` for `x<0`): negative saturation pushes mean activation toward zero; monotonic.
- **Self-gated smooth units** (SiLU/Swish `x·σ(βx)`, GELU `x·Φ(x)`, Mish `x·tanh(softplus(x))`): multiply the input by a smooth nonlinear function of itself, so they are differentiable everywhere, take small negative values, and are non-monotonic with a shallow negative bump.

## Fixed substrate / Code framework

The training pipeline and model definitions are frozen:

- **Optimizer:** SGD, `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`.
- **Schedule:** cosine annealing over 200 epochs (`CosineAnnealingLR`, `T_max=200`).
- **Initialization:** Kaiming-normal (`fan_out`, `nonlinearity='relu'`) on conv weights, `fan_in` on linear weights; BatchNorm weight 1, bias 0. Fixed regardless of activation.
- **Data:** `Resize(32)` → `RandomCrop(32, padding=4)` → `RandomHorizontalFlip` → `ToTensor` → per-dataset `Normalize`; FashionMNIST is repeated to 3 channels. Test transform is resize + normalize. Batch size 128, `CrossEntropyLoss`.
- **Architectures:** CIFAR ResNet-20 (BasicBlock `[3,3,3]`, 3×3 stem, global avg pool); VGG-16-BN (Conv–BN–activation blocks + 512→512 classifier with activation and dropout 0.5); MobileNetV2 (inverted residuals, activation replacing ReLU6). The activation module is instantiated in every place ReLU would appear.

The metric is best test accuracy reached during the 200 epochs, so the activation is judged on the peak of the fixed cosine run.

## Editable interface

Exactly one region is editable — the `CustomActivation` class in `pytorch-vision/custom_activation.py` (the `nn.Module` used everywhere a ReLU would be). The contract is a shape-preserving element-wise (or channel-wise) `forward(x)`; learnable parameters may be registered in `__init__`.

The starting point is the scaffold default: plain ReLU.

```python
# EDITABLE region of pytorch-vision/custom_activation.py (lines 32-49) -- default fill (ReLU)
class CustomActivation(nn.Module):
    """Custom activation function. Drop-in replacement for ReLU.

    Input/Output: tensor of any shape, element-wise operation.

    Design considerations:
        - Monotonicity vs non-monotonicity tradeoffs
        - Smoothness (differentiability everywhere vs piecewise)
        - Negative-domain behavior (zero, linear, bounded)
        - Computational cost (simple ops vs learned parameters)
        - Interaction with BatchNorm and residual connections
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x)  # default: ReLU
```

(`torch`, `torch.nn as nn`, and `torch.nn.functional as F` are already imported at module top.)

## Evaluation settings

Three architecture/dataset pairs, one seed, evaluated by best test accuracy (%, higher is better):

- **resnet20-cifar10** — ResNet-20 on CIFAR-10.
- **vgg16bn-cifar100** — VGG-16-BN on CIFAR-100.
- **mobilenetv2-fmnist** — MobileNetV2 on FashionMNIST (grayscale repeated to 3 channels).

Seed `42`. Each run is the full fixed pipeline above (SGD + cosine, 200 epochs) with only `CustomActivation` changed.
