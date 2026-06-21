## Research question

Train a neural net from scratch to a fixed mean test accuracy on CIFAR-10 as fast as possible on a single GPU. The accuracy bar is **94% mean test accuracy**, with documented **95%** and **96%** variants as harder targets. Hardware is fixed at one **NVIDIA A100**. Timing starts when the method first receives the training data and ends when it emits test-set predictions; a method is valid only if its mean accuracy over repeated runs clears the bar. Warmup runs on dummy data and arbitrary test-time augmentation are allowed.

The train set (50,000 images), test set (10,000 images), accuracy bar, and single-A100 budget are frozen. The only free variable is the **training method**: network architecture, initialization, optimizer, data augmentation, and inference procedure. The ranking metric is **A100-seconds to reach the accuracy bar** (lower is better).

## Prior art / Background / Baselines

- **David Page / Myrtle "How to train your ResNet" (2019).** A carefully engineered ResNet-style training pipeline reaches 94% in ~10 A100-seconds. Gap: it relies on a deep residual network and a long schedule, leaving headroom for a faster single-GPU result.
- **tysam-code/hlb-CIFAR10 (2023).** A compact from-scratch network reaches 94% in **6.3 A100-seconds**. Gap: its unwhitened variant falls back to **18.3 A100-seconds**, so the same network without the published extras is much slower.

The starting point is the **unwhitened baseline**: the hlb-CIFAR10 network trained with Nesterov SGD, a triangular learning-rate schedule, label smoothing 0.2, and horizontal-flip plus random-translation augmentation. It reaches 94% mean accuracy in **45 epochs / 18.3 A100-seconds**.

## Fixed substrate / Code framework

The network is a VGG-like convolutional net of ~1.97M parameters: a 2×2-stride first convolution with no padding, three blocks of two 3×3 convolutions with `MaxPool2d(2)`, BatchNorm, and GELU, a final max-pool, and a bias-free linear head whose output is scaled by 1/9. Convolutional and linear biases are disabled; BatchNorm scale is frozen at 1 and only its biases train.

Training uses Nesterov SGD at batch size 1024 with label smoothing 0.2 and a triangular learning-rate schedule (start at 0.2× the max, peak 20% of the way through, decay to zero). Augmentation is horizontal flip plus 2-pixel reflection-padded random translation. Evaluation uses horizontal-flip test-time augmentation.

## Editable interface

The editable parts are the network definition, initialization, optimizer, data augmentation, and inference procedure. The dataset, accuracy bar, hardware, and evaluation protocol are fixed. The scaffold below is the baseline; the marked regions are the editable slots.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('bias', False)
        super().__init__(*args, **kwargs)

def make_network():
    """EDITABLE: return the trainable network."""
    net = nn.Sequential(
        Conv(3, 24, kernel_size=2, stride=2, padding=0),  # first conv
        nn.GELU(),
        Conv(24, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
        Conv(64, 64, kernel_size=3, padding=1), nn.MaxPool2d(2), nn.BatchNorm2d(64), nn.GELU(),
        Conv(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        Conv(128, 128, kernel_size=3, padding=1), nn.MaxPool2d(2), nn.BatchNorm2d(128), nn.GELU(),
        Conv(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
        Conv(256, 256, kernel_size=3, padding=1), nn.MaxPool2d(2), nn.BatchNorm2d(256), nn.GELU(),
        nn.Flatten(),
        nn.Linear(256, 10, bias=False),  # head output scaled by 1/9
    )
    # Fixed substrate: BN scale frozen at 1, conv/linear biases disabled.
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.weight.data.fill_(1.0)
    return net

def make_optimizer(model, lr, momentum, weight_decay):
    """EDITABLE: return the optimizer."""
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                           weight_decay=weight_decay, nesterov=True)

def train_augment(images):
    """EDITABLE: training-time augmentation."""
    return images

def train(model, loader, optimizer, epochs):
    """EDITABLE: training loop."""
    for epoch in range(epochs):
        for inputs, targets in loader:
            outputs = model(train_augment(inputs))
            loss = F.cross_entropy(outputs, targets, label_smoothing=0.2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def infer(model, images):
    """EDITABLE: test-time augmentation and inference."""
    with torch.no_grad():
        return model(images)
```

## Evaluation settings

The ranking metric is **A100-seconds to clear the accuracy bar** (lower is better), reported as the mean over many repeated runs. All runs are on the same 50,000-image train set and 10,000-image test set. The primary target is 94% mean test accuracy; the 95% and 96% variants use the same protocol with a higher bar.
