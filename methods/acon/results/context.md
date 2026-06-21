## research question

Swish improves consistently over ReLU but was discovered by architecture search rather than derived. How should the shape of a scalar activation function be chosen for a deep network, and can that shape be made to depend on the network's input? Both small mobile models and large optimized backbones are of interest.

## background

The standard activations are ReLU and its piecewise-linear relatives Leaky ReLU and PReLU. Maxout generalizes this family by taking the maximum of two linear functions, at the cost of extra parameters and non-smoothness. Softplus is a smooth convex approximation to ReLU. Swish is a strong smooth activation, found by search; GELU and Mish are similar fixed smooth curves. On the adaptive side, SENet reweights channels and DY-ReLU conditions piecewise-linear coefficients on the input, both making the nonlinearity input-aware through a routing module. Candidates are typically checked on both small mobile models and large optimized backbones.

## baselines

- **ReLU** `max(x,0)`: hard sign gate with zero on the negative side.
- **Leaky ReLU / PReLU** `max(x, px)`: adds a small fixed or learned negative slope.
- **Maxout** `max(η_a(x), η_b(x))`: takes the maximum of two learned linear functions.
- **Swish / SiLU** `x·σ(βx)`: a smooth self-gated curve found by architecture search.
- **Softplus** `log(1+eˣ)`: a smooth convex approximation to ReLU.
- **GELU / Mish** `x·Φ(x)`, `x·tanh(softplus(x))`: other smooth self-gated curves.
- **SENet** and **DY-ReLU**: channel reweighting, or input-conditioned piecewise-linear coefficients.

## evaluation settings

Drop-in activation swaps on ImageNet-1k (224×224, top-1 error) across MobileNetV1/V2, ShuffleNetV2, ResNet-18/50/101/152, and SENet-154. Lightweight models follow the ShuffleNetV2 recipe; large models use a linear-decay LR from 0.1, weight decay 1e-4, batch 256, ~600k iterations. Comparisons include static activations and adaptive modules, with ablations over routing granularity and candidate activations on ShuffleNetV2 0.5×. Transfer tasks are COCO detection with RetinaNet (ResNet-50) and CityScapes segmentation with PSPNet (ResNet-50). Metrics are top-1 error, AP, mIoU, plus parameter/FLOP counts and gate histograms for conditioned gates.

## code framework

```python
import torch
import torch.nn as nn


class Activation(nn.Module):
    """Drop-in activation, swapped in for ReLU/Swish with the net otherwise fixed."""

    def __init__(self, num_channels):
        super().__init__()
        # TODO
        pass

    def forward(self, x):
        # TODO
        pass
```
