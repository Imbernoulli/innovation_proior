# Context

## Research question

By 2021 the prevailing wisdom in visual recognition had flipped. For most of the 2010s, convolutional networks were the unquestioned backbone for image classification, detection, and segmentation. Then Vision Transformers arrived, and within a year hierarchical Transformers such as Swin became the state of the art across classification, detection, and segmentation alike. The community's explanation crystallized quickly: self-attention has some intrinsic representational advantage that convolution lacks, and that is why Transformers scale better.

The problem is that this comparison is confounded. A modern Swin model differs from a textbook ResNet on many axes at once — the training recipe (optimizer, epochs, augmentation, regularization), the macro layout (how compute is distributed across stages, how the input is tokenized, how resolution is reduced between stages), and a dozen micro choices (which normalization, which activation, how many of each per block). When a Swin beats a ResNet, which of those differences is doing the work? The precise question is: if we hold FLOPs roughly fixed and one by one transplant the design and training decisions that Transformers brought, how much of the gap is attention itself versus everything else bundled around it? A satisfying answer would either isolate self-attention as the irreplaceable ingredient, or show that a pure ConvNet, modernized with the same non-attention choices, can close the gap — which would mean the gap was never about attention.

## Background

The field state rests on a long line of ConvNet design and a short, sharp disruption from Transformers.

Convolution carries built-in inductive biases that suit images: translation equivariance, locality, and — when used in a sliding-window, fully-convolutional manner — shared computation across spatial positions, which is what makes detection and segmentation on high-resolution inputs tractable. Through the 2010s a sequence of architectures refined how to spend convolutional compute: VGG established that stacking small 3×3 kernels is both expressive and hardware-friendly; ResNet (He et al. 2016) made very deep networks trainable with residual connections and a bottleneck block; ResNeXt (Xie et al. 2017) showed grouped convolution improves the FLOPs/accuracy trade-off; MobileNet (Howard et al. 2017) and Xception popularized depthwise separable convolution; MobileNetV2 (Sandler et al. 2018) introduced the inverted residual with a linear bottleneck; EfficientNet and RegNet pushed scaling rules and design-space search.

The disruption: Vision Transformers (Dosovitskiy et al. 2021) split an image into non-overlapping patches, embed them as a token sequence, and apply standard NLP Transformer blocks — multi-head self-attention plus a two-layer MLP with a 4× hidden expansion, each preceded by LayerNorm, with GELU as the activation. Apart from the initial patchify step, ViT injects almost no image-specific prior, yet with enough data and model size it beats ResNets on classification. But vanilla ViT's global self-attention is quadratic in the number of tokens, which becomes intractable at the high resolutions detection and segmentation need. Swin Transformer (Liu et al. 2021) fixed this by reintroducing ConvNet priors: a multi-stage hierarchy with feature maps that halve in resolution and double in width across four stages, self-attention restricted to local windows (window size at least 7×7) with a shifted-window scheme to allow cross-window communication, and a separate patch-merging layer to downsample between stages. Swin's stage compute ratio is 1:1:3:1 (1:1:9:1 for larger variants), concentrating computation in the third stage.

A few load-bearing concepts and diagnostic facts about existing systems:
- A modern training recipe alone is known to lift a plain ResNet-50 well above its original reported accuracy — recent studies (Bello et al. 2021; Wightman et al. 2021) show this. The original torchvision ResNet-50 reports 76.1% top-1 on ImageNet-1K trained for 90 epochs.
- Depthwise convolution mixes information only across spatial positions, per channel; a 1×1 convolution mixes only across channels. Self-attention likewise computes a spatially-varying weighted sum applied per channel, and its MLP mixes channels — the two families share a "separate spatial and channel mixing" structure.
- BatchNorm, while standard in ConvNets, has known intricacies — dependence on batch statistics, train/test discrepancy, and interactions with running averages — that can hurt models in some regimes; LayerNorm, used throughout Transformers, is per-sample and sidesteps these. Directly swapping LayerNorm for BatchNorm in a vanilla ResNet is known to degrade accuracy (Wu & Johnson 2021).
- GELU (Hendrycks & Gimpel 2016), x·Φ(x) with Φ the standard Gaussian CDF, is a smooth variant of ReLU used in BERT, GPT, and ViT.
- LayerScale (Touvron et al. 2021, CaiT): a per-channel learnable diagonal applied to each residual branch's output, initialized to a small value so each block begins close to identity, which eases optimization of deep stacks.
- Stochastic depth (Huang et al. 2016): during training, randomly drop whole residual branches (replace with identity), a regularizer for deep residual nets.

## Baselines

The natural points of comparison are the strong models a new backbone would be measured against.

**ResNet / ResNeXt.** The standard ResNet-50 uses a stem of a 7×7 stride-2 convolution followed by a 3×3 max pool (4× downsampling), then four stages of bottleneck blocks with block counts (3,4,6,3); each bottleneck is 1×1 (reduce) → 3×3 → 1×1 (expand) with BatchNorm and ReLU after each conv, and a residual add; downsampling happens inside the first block of each stage via stride-2 3×3 conv and a stride-2 1×1 conv on the shortcut. ResNeXt replaces the 3×3 with a grouped convolution and widens to compensate, following "use more groups, expand width." Gap: these are pre-Transformer designs trained with short schedules and light augmentation; their macro layout (stage ratio, stem) and micro choices (per-conv ReLU/BN, small kernels) were never revisited against the Transformer recipe, leaving open how much of their deficit is architectural versus training.

**MobileNetV2 inverted residual.** The block goes narrow → wide → narrow: a 1×1 expansion (expansion factor 6) lifts channels, a depthwise 3×3 filters spatially, a 1×1 projection contracts, with the residual connecting the narrow ends and no activation on the final projection (linear bottleneck). Gap: designed for mobile efficiency, not as a high-accuracy backbone, and not paired with the Transformer training recipe or macro layout.

**ViT.** Patchify (e.g. 16×16 non-overlapping) → token embeddings → repeated blocks of LayerNorm → multi-head self-attention → residual, then LayerNorm → MLP (hidden 4×, GELU) → residual. No hierarchy, no convolution beyond the patch embedding. Gap: global attention is quadratic in token count, so it does not scale to dense prediction at high resolution; it also needs large-scale pretraining to shine.

**Swin Transformer.** A four-stage hierarchy with patch-merging downsampling, window-based multi-head self-attention with relative position bias, shifted windows for cross-window mixing, stage ratio 1:1:3:1, channel widths starting at 96 and doubling per stage, window size 7. Gap: it achieves its generality by re-importing convolutional priors through increasingly intricate machinery (cyclic-shift windows, relative position biases), and its success is still credited to attention rather than to those reintroduced priors — leaving unexamined whether a plain ConvNet equipped with the same priors and recipe would match it.

## Evaluation settings

- **ImageNet-1K**: 1000 classes, ~1.2M training images; report top-1 accuracy on the 50K validation set. Standard testing crop ratio 0.875 at 224², 1.0 at 384².
- **ImageNet-22K**: ~14M images, 21841 classes (superset of the 1K classes), used for large-scale pretraining followed by fine-tuning on ImageNet-1K.
- **Two FLOP regimes** for controlled comparison: ResNet-50 / Swin-T (~4.5×10⁹ FLOPs) and ResNet-200 / Swin-B (~15×10⁹ FLOPs).
- **Modern training recipe** (the yardstick the Transformer models use): AdamW optimizer, 300-epoch schedule with linear warmup and cosine decay, batch size 4096, weight decay 0.05; data augmentation via Mixup, Cutmix, RandAugment, Random Erasing; regularization via Stochastic Depth and Label Smoothing; EMA of weights. Accuracies on the small regime averaged over three seeds.
- **Downstream**: COCO object detection/instance segmentation with Mask R-CNN / Cascade Mask R-CNN; ADE20K semantic segmentation with UperNet. Metrics: box/mask AP for COCO, mIoU for ADE20K.

## Code framework

The primitives that already exist: PyTorch `nn.Conv2d` (including grouped/depthwise via `groups`), `nn.Linear`, `nn.GELU`, normalization layers, and the stochastic-depth `DropPath` and truncated-normal init from `timm`. A staged backbone is built from a stem, a sequence of resolution stages each made of repeated residual blocks, optional downsampling between stages, a global pool, and a linear classifier head. The contribution will live inside the residual block and the stem/downsampling structure.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

class Block(nn.Module):
    """A single residual block. The internal structure — which convs,
    where normalization and activation sit, what mixes spatial vs channel
    information — is the open design question."""
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # TODO: spatial mixing op, channel-mixing ops, norm, activation,
        #       residual-branch scaling
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # TODO: transform x through the block body
        x = input + self.drop_path(x)
        return x

class Backbone(nn.Module):
    """Generic 4-stage hierarchical ConvNet: stem -> [stage -> downsample] -> pool -> head."""
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = self._make_stem(in_chans, dims[0])  # TODO: how to tokenize/downsample the input
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(self._make_downsample(dims[i], dims[i+1]))  # TODO

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = None   # TODO: final normalization choice
        self.head = nn.Linear(dims[-1], num_classes)

    def _make_stem(self, in_chans, dim):
        pass  # TODO

    def _make_downsample(self, in_dim, out_dim):
        pass  # TODO

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x  # TODO: pool + normalize

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
```
