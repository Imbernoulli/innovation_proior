# Research question

Semantic segmentation labels every pixel of an image with a class. The natural design is encoder–decoder: a backbone produces features, a decoder turns them into a per-pixel map. The question is how to build a segmentation model that is simultaneously **accurate**, **efficient**, and **robust** — in particular robust to the resolution gap between training and testing, because segmentation is routinely trained on fixed crops (e.g. 512²) but deployed at arbitrary, often much larger, image sizes.

Two structural tensions make this hard.

First, **a single feature scale is not enough.** Semantic segmentation needs both high-resolution detail (to get object boundaries right) and large effective receptive field / coarse semantic context (to know *what* a region is). A backbone that emits one low-resolution feature map can recover semantics but loses the fine detail; the field's answer has always been a *multi-scale* feature hierarchy (strides 4, 8, 16, 32) feeding a decoder that fuses scales.

Second, **positional encodings tie a Transformer to its training resolution.** A Transformer's attention is permutation-invariant, so it needs position information injected somehow. The standard trick — a learned, fixed-length positional embedding added to the tokens — assumes the number and arrangement of tokens at test time matches training. When the test image is a different size, that embedding has to be interpolated, and the interpolation degrades accuracy. For a task whose whole point is to run at varied, large resolutions, this is a real problem.

A satisfactory answer must (a) emit a multi-scale hierarchical feature map like a convnet, so the decoder gets both detail and context; (b) keep attention affordable at the large token counts segmentation demands; (c) avoid any positional encoding that breaks under resolution change; and (d) pair the encoder with a decoder simple enough not to become the bottleneck.

# Background

**Fully-convolutional segmentation.** Since the fully-convolutional network (Long et al., 2015) reframed segmentation as dense per-pixel classification, the dominant designs have been convolutional encoder–decoders. The recurring problem is that a classification backbone downsamples aggressively (to stride 32), which is good for semantics but destroys spatial precision. The field's fixes all aim to recover resolution and enlarge receptive field: U-Net (Ronneberger et al., 2015) adds skip connections from encoder to decoder to bring detail back; the DeepLab family (Chen et al., 2017) uses *dilated/atrous* convolutions to enlarge receptive field without downsampling, and atrous spatial pyramid pooling (ASPP) to gather multi-scale context; PSPNet (Zhao et al., 2017) uses a pyramid pooling module for the same end. These decoders are heavy — ASPP and PPM add substantial compute — and the receptive field of stacked convolutions still grows only slowly with depth.

**The Transformer and self-attention.** In NLP the dominant architecture (Vaswani et al., 2017) is built from scaled dot-product attention, `Attention(Q,K,V) = softmax(QKᵀ/√d) V`, in parallel heads, wrapped in residuals and layer normalization, with a position-wise MLP feed-forward sublayer. Attention models long-range dependencies in a single layer — exactly the global context segmentation wants — but it is global, so for `N` tokens it costs Θ(N²) in both the score matrix and the value aggregation. A feature map tokenized at a fine stride has thousands of tokens, so plain global attention is intractable for dense prediction.

**Vision Transformer (Dosovitskiy et al., 2020).** The first convincing pure-Transformer image classifier: cut the image into non-overlapping 16×16 patches, linearly embed each into a token, prepend a class token, add a *learned absolute positional embedding*, and run a standard Transformer encoder with global attention. Two properties matter here and both are liabilities for segmentation. It emits a *single* low-resolution feature map (stride 16 throughout — no hierarchy), so a segmentation decoder gets no fine-scale features. And its fixed-length learned positional embedding is tied to the training token grid: at a different test resolution the embedding must be interpolated, which the diagnostic measurements show *drops accuracy* the more the test resolution departs from training.

**Segmentation atop a plain ViT (SETR, Zheng et al., 2020).** A demonstration that a ViT encoder can do segmentation, by attaching a progressive-upsampling or multi-level decoder to the single-scale ViT features. It confirms Transformers are competitive for segmentation but inherits the ViT liabilities: single-scale features (recovered only by a comparatively complex decoder), quadratic attention, and the resolution-fragile positional embedding. Its decoders are also non-trivial.

**Pyramid vision Transformers.** A concurrent line builds a *hierarchical* Transformer by shrinking the spatial resolution stage by stage so the backbone emits strides 4/8/16/32 like a convnet (Wang et al., 2021), often combined with a *spatial-reduction* attention that shrinks the key/value sequence to control cost. This solves the no-hierarchy problem and softens the quadratic cost. But these designs still carry positional encodings, so the resolution-robustness problem persists, and they are typically evaluated with heavy decoder heads.

**Position information in convnets.** A separate observation (Chu et al., 2021; Islam et al., 2020) is that a convolution applied to a feature map is not as position-agnostic as it appears: its zero-padding at the borders gives border locations a systematically different response, so convnets can encode some absolute spatial information even without an explicit positional signal.

# Baselines

**FCN / U-Net / DeepLabv3+ / PSPNet (convolutional encoder–decoders).** A convnet backbone (often ResNet) emits a feature hierarchy; the decoder enlarges receptive field and recovers resolution via dilated convolutions (DeepLab), pyramid pooling (PSPNet), or skip connections (U-Net). *Gap:* convolution's receptive field grows slowly and its weights are content-independent; achieving global context requires heavy modules (ASPP/PPM) that dominate the compute, and accuracy at high efficiency is limited.

**ViT / SETR.** Patchify → linear embed → +class token → +learned absolute positional embedding → L global-attention encoder blocks; for segmentation, a decoder upsamples the single-scale features. *Gap:* one feature scale (stride 16), so the decoder must fabricate multi-scale information; global attention is Θ(N²), intractable at dense resolutions; the fixed positional embedding must be interpolated at test time and accuracy drops as test resolution diverges from training; the SETR decoder is comparatively heavy.

**Pyramid vision Transformer (PVT).** A hierarchical Transformer backbone emitting strides 4/8/16/32, with spatial-reduction attention to cut the key/value length. *Gap:* it still uses positional encodings (resolution-fragile), tokenizes the image into non-overlapping patches, and is usually paired with conventional heavy decoders.

# Evaluation settings

- **ADE20K** (Zhou et al., 2017): 20,210 training / 2,000 validation images, 150 semantic categories — a hard, fine-grained benchmark. Metric: mean Intersection-over-Union (mIoU), single- and multi-scale.
- **Cityscapes** (Cordts et al., 2016): 5,000 finely annotated urban-scene images (2,975 train / 500 val / 1,525 test), 19 classes, 1024×2048 native resolution — the canonical high-resolution segmentation benchmark. Metric: mIoU.
- **COCO-Stuff** (Caesar et al., 2018): 164K images, 172 categories, for testing on a large, diverse label set.
- Protocol: backbone pre-trained on ImageNet-1K classification, then the full encoder–decoder fine-tuned on the segmentation set. Training with AdamW, polynomial learning-rate decay with warm-up, random-scale/crop/flip augmentation, fixed crop size (e.g. 512² on ADE20K, 1024×1024 on Cityscapes). Efficiency reported alongside accuracy: parameter count, FLOPs, and measured FPS/latency. Robustness probed by training at one resolution and testing at others.

# Code framework

A generic semantic-segmentation harness can already be written with PyTorch modules: a backbone turns an image into feature maps, a decode head fuses those features into stride-reduced logits, and the training step optimizes per-pixel cross-entropy. The unresolved part is the concrete backbone/head design that makes dense prediction accurate, efficient, and stable across image sizes.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone(nn.Module):
    # Image -> list of feature maps. The architecture is the open design question.
    def __init__(self, in_chans=3):
        super().__init__()
        # TODO: build a backbone that returns spatial features useful for dense prediction.
        pass

    def forward(self, x):
        # TODO: return feature maps ordered from finer to coarser resolution.
        pass


class DecodeHead(nn.Module):
    # Feature maps -> per-pixel class logits at a stride suitable for final upsampling.
    def __init__(self, num_classes):
        super().__init__()
        # TODO: fuse the backbone features without making the head the compute bottleneck.
        pass

    def forward(self, features):
        # TODO: return logits with shape (B, num_classes, h, w).
        pass


class SegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Backbone()
        self.decode_head = DecodeHead(num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.decode_head(features)
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)


def build_optimizer(model, lr=6e-5, weight_decay=1e-2):
    # AdamW + polynomial decay with warm-up (scheduler omitted here).
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_step(model, images, targets, optimizer):
    optimizer.zero_grad()
    logits = model(images)
    loss = F.cross_entropy(logits, targets, ignore_index=255)
    loss.backward()
    optimizer.step()
    return loss.item()
```
