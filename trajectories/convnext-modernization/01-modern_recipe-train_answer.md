The 76.13% reported for ResNet-50 and the 81.30% reported for Swin-T are not
a clean architecture comparison, because they were trained under two
different procedures at once: ResNet-50 under the original 90-epoch SGD
recipe, Swin-T under AdamW with hundreds of epochs and a heavy augmentation
bundle. Before I touch a single convolution I want that confound gone, so
the first experiment holds architecture at exactly zero and moves only the
training recipe: retrain the completely unmodified ResNet-50 — same stem,
same stage depths (3, 4, 6, 3), same bottleneck blocks — under a recipe
matching what the Transformer-family papers use as a package: AdamW instead
of SGD, 300 epochs instead of 90, RandAugment/Mixup/CutMix/Random Erasing for
augmentation, Stochastic Depth and Label Smoothing for regularization, and a
LayerScale gate (initialized near zero, CaiT-style) wrapping every residual
branch. I take the recipe as a bundle rather than tuning each ingredient
separately, because the question this step answers is coarse — is the
training-procedure gap large or small — not which single augmentation knob
matters most. Every piece here is architecture-agnostic: augmentation and
regularization act on the input or the loss, not on whether the blocks are
convolutional or attention-based, and LayerScale is a strict generalization
of "no extra gate" that only changes how a residual branch's contribution is
weighted during optimization, not what operations exist in the block. The
one thing I deliberately leave out is Exponential Moving Average, since
EMA-smoothed weights and BatchNorm's running statistics can drift out of
sync, and the network is still BatchNorm-based here — normalization itself
isn't up for revision yet.

This recipe, once measured, is frozen for every step that follows. The whole
point of isolating training procedure now is wasted if a later step quietly
changes the optimizer or augmentation strength too, so whatever this rung
measures becomes the fixed training-procedure term, and every later delta
is attributable to architecture by construction. I don't have a confident
guess for the exact size of the recipe-alone gain — it could be a large
fraction of the 5.2-point ResNet-50-to-Swin-T gap, in which case the coming
architecture edits are a lighter-touch story than they look, or it could be
small, in which case nearly all of that gap is architectural and the coming
redesign carries the real burden. What I do expect, since this is the same
recipe bundle credited with lifting plain ResNet-50 elsewhere, is a real,
non-trivial improvement. The one hard check regardless of the accuracy
number: since nothing in the network itself changed, GFLOPs must come out
identical to the untouched baseline's 4.09G — any drift there would mean an
architectural edit snuck into the recipe change by accident, and would need
to be found before trusting the accuracy number at all.

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
