**Problem.** ResNet-50's published 76.13% top-1 and Swin-T's 81.30% top-1 were
measured under two different training procedures (original SGD/90-epoch
recipe vs. AdamW/300-epoch/heavy-augmentation recipe), so the raw gap
confounds architecture with optimization procedure. Before changing the
network, isolate the training-procedure term by retraining the *unmodified*
ResNet-50 architecture under a Transformer-family recipe.

**Recipe (frozen from here on for every subsequent rung, architecture-only
comparisons going forward).**

- Optimizer: AdamW, base learning rate 4e-3, weight decay 0.05
- Schedule: 300 epochs, cosine decay, 20-epoch linear warmup, batch size 4096
- Augmentation: RandAugment (9, 0.5), Mixup (0.8), CutMix (1.0), Random
  Erasing (0.25)
- Regularization: Stochastic Depth, Label Smoothing (0.1)
- LayerScale (init 1e-6) wrapping every residual branch
- No Exponential Moving Average (EMA is left out because it interacts badly
  with BatchNorm's running statistics, and the network is still BN-based)

**Architecture.** Completely unmodified ResNet-50: 7x7 stride-2 conv + maxpool
stem, stage depths (3, 4, 6, 3), standard bottleneck blocks (1x1 reduce ->
3x3 conv -> 1x1 expand), BatchNorm + ReLU throughout.

```python
# rung 1: standard torchvision ResNet-50, architecture completely unmodified.
import torch
import torch.nn as nn
from torchvision.models import resnet50

model = resnet50(weights=None)  # architecture only; weights trained fresh below

# --- LayerScale wrapper for every residual branch (CaiT-style, init 1e-6) ---
class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        # x: (N, C, H, W) -- scale per output channel before the residual add
        return x * self.gamma[None, :, None, None]

def add_layerscale_to_bottleneck(block: "torchvision.models.resnet.Bottleneck"):
    out_dim = block.conv3.out_channels
    block.layerscale = LayerScale(out_dim, init_value=1e-6)
    orig_forward = block.forward

    def forward(x):
        identity = x
        out = block.conv1(x); out = block.bn1(out); out = block.relu(out)
        out = block.conv2(out); out = block.bn2(out); out = block.relu(out)
        out = block.conv3(out); out = block.bn3(out)
        out = block.layerscale(out)                 # <-- only addition vs. stock ResNet
        if block.downsample is not None:
            identity = block.downsample(x)
        out = out + identity
        return block.relu(out)

    block.forward = forward

for module in model.modules():
    if module.__class__.__name__ == "Bottleneck":
        add_layerscale_to_bottleneck(module)

# --- training recipe (fixed for every rung from here on) ---
training_recipe = dict(
    optimizer="AdamW", base_lr=4e-3, weight_decay=0.05,
    epochs=300, lr_schedule="cosine", warmup_epochs=20, warmup_schedule="linear",
    batch_size=4096,
    randaugment=(9, 0.5), mixup=0.8, cutmix=1.0, random_erasing=0.25,
    label_smoothing=0.1, stochastic_depth=True,
    layer_scale_init=1e-6, ema=None,  # EMA off: BatchNorm-incompatible
)
```

**Test.** Train this exact architecture under this recipe, three random
seeds, ImageNet-1K, 224x224. Read off mean +/- std top-1 accuracy and GFLOPs.
GFLOPs must equal the untouched baseline's 4.09G exactly, since nothing
architectural changed — that is the sanity check that no accidental
architecture edit crept into the recipe change.
