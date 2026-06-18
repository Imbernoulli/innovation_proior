# Context

## Research question

Computer vision has a single workhorse backbone — the convolutional network — that is trained once on classification and then reused, almost unchanged, as the feature extractor for object detection, instance segmentation, and semantic segmentation. The question is whether a Transformer can play that same role: a *general-purpose* visual backbone that serves every recognition task, the way the Transformer already dominates natural language processing.

Two properties of images make a direct port of a language Transformer hard.

First, **scale**. In language, the token (a word/subword) is the atomic unit and is essentially of one scale. In images, the objects of interest span enormous scale ranges — a distant pedestrian and a foreground bus live in the same frame — and the standard remedy in detection and segmentation is a *feature pyramid*: feature maps at several resolutions (typically strides 4, 8, 16, 32 relative to the input). A backbone that emits only a single-resolution feature map cannot feed the pyramid-based heads (FPN, U-Net, the FPN inside Mask R-CNN) that the field relies on.

Second, **resolution and cost**. Self-attention compares every token with every other token, so its cost grows quadratically in the number of tokens. A dense-prediction image tokenized at a fine stride has thousands of tokens; at detection/segmentation resolutions the token count is far larger still. Quadratic attention is then intractable.

A satisfactory answer must therefore (a) produce a *hierarchical, multi-scale* feature map at the same strides a convnet produces, so it drops into existing dense-prediction heads without modification; (b) have computational complexity that is *linear*, not quadratic, in the number of pixels; and (c) keep enough modeling power to match or beat strong convolutional backbones across all three tasks.

## Background

**Convolutional backbones and the pyramid.** Since AlexNet, image models have been deep convolutional networks (VGG, GoogLeNet, ResNet, DenseNet, HRNet, EfficientNet), refined through depth, residual connections, grouped/depthwise/deformable convolutions, and width/resolution scaling. A defining structural property is the *stage hierarchy*: a stem downsamples to stride 4, then each of a few stages halves the spatial resolution and roughly doubles the channels, yielding feature maps at strides 4, 8, 16, 32. This pyramid is exactly what the dense-prediction machinery consumes. Feature Pyramid Networks (Lin et al., 2017) and U-Net (Ronneberger et al., 2015) combine the multi-resolution maps so that small objects are localized on the fine, high-resolution maps and large objects on the coarse ones. Any backbone that wants to serve detection and segmentation has to supply this pyramid.

**The Transformer.** In NLP the dominant architecture (Vaswani et al., 2017) is built from scaled dot-product attention, `Attention(Q,K,V) = softmax(QKᵀ/√d) V`, run in parallel heads, wrapped in residual connections and layer normalization, with a two-layer position-wise MLP (GELU activation, ~4× hidden expansion) as the feed-forward sublayer. Attention models long-range dependencies directly, but it is global — every token attends to every other — so for n tokens it costs Θ(n²) in both the score matrix and the value aggregation.

**Vision Transformer (Dosovitskiy et al., 2020).** The first convincing demonstration that a near-pure Transformer can do image classification. It cuts the image into non-overlapping 16×16 patches, linearly embeds each patch into a token, prepends a learnable class token, adds a learned absolute position embedding, and runs a standard Transformer encoder with *global* self-attention. It reaches an excellent speed/accuracy trade-off on classification — but only when pre-trained on a very large dataset (JFT-300M); on ImageNet-1K alone it underperforms convnets. Structurally it has two limitations that matter here: it emits a *single* low-resolution feature map (stride 16 throughout — no pyramid), and its global attention is *quadratic* in image size, so it cannot scale to the token counts of dense prediction.

**Training Transformers on moderate data (Touvron et al., 2020).** A data-efficient recipe shows the same Transformer can be trained on ImageNet-1K with heavy augmentation and regularization (AdamW; RandAugment; Mixup; CutMix; random erasing; stochastic depth / drop-path), plus distillation. This removes the JFT-300M requirement but leaves the single-scale, quadratic structure untouched.

**Self-attention as a local operator.** A line of work replaces spatial convolutions in a ResNet with self-attention computed in a local window *around each pixel* — a sliding window (Ramachandran et al., 2019; Hu et al., 2019; Zhao et al., 2020). This makes attention local and, in principle, cheaper, and it reaches slightly better accuracy/FLOPs than the convolutional counterpart. The reported pain point is *real* latency: because each query pixel sits at the center of its own window, different queries have *different* key sets, which defeats batched matrix multiplication and produces poor memory-access patterns on general hardware — so wall-clock time is far worse than the FLOP count suggests. This is the diagnostic finding that distinguishes "local attention is a good idea" from "*sliding-window* local attention is hardware-friendly."

**Relative position encoding.** Rather than (or in addition to) an absolute position embedding, several works (Shaw et al., 2018; Raffel et al., 2019; Hu et al., 2018; Hu et al., 2019) add a learned bias to the attention logits that depends only on the *relative* offset between the query and key positions. This injects a translation-equivariant geometric prior, which matches the statistics of images better than absolute coordinates.

**Efficient attention.** A separate strand attacks the quadratic cost by approximating the softmax kernel to obtain linear-time attention (e.g., Performer, Choromanski et al., 2020). It reduces cost but pays in accuracy, and it does not by itself produce a feature pyramid.

A concurrent observation: a multi-resolution Transformer pyramid can be built by progressively reducing spatial resolution (Wang et al., 2021), but if the attention within each stage remains global-like, the cost stays quadratic in image size even though the pyramid is present.

## Baselines

**Vision Transformer (ViT).** Patchify (16×16) → linear embed → +class token → +absolute position embedding → L identical Transformer encoder blocks with global MSA → classify from the class token. Block: `x ← x + MSA(LN(x)); x ← x + MLP(LN(x))`, pre-norm, GELU MLP at 4× width, head dimension 64, scaling `1/√d`. *Gap:* one feature scale (stride 16) so it cannot feed pyramid heads; global attention is Θ((hw)²) so it is intractable at dense-prediction resolutions; data-hungry without huge pre-training.

**DeiT.** Same architecture as ViT, trained on ImageNet-1K with strong augmentation/regularization and distillation. *Gap:* fixes the data requirement, not the structure — still single-scale and quadratic; cannot be dropped into detection frameworks without bolt-on deconvolution to fabricate multiple scales.

**ResNet / ResNeXt.** The standard convolutional backbones: a stem to stride 4 then four residual stages at strides 4/8/16/32, channels doubling per stage; ResNeXt adds grouped convolutions. They naturally produce the pyramid and have linear cost in pixels, so they slot directly into FPN/Mask-R-CNN/UperNet. *Gap:* convolution's receptive field grows only slowly with depth and its weights are content-independent; it lacks the data-dependent long-range interaction that attention provides.

**Sliding-window self-attention backbones.** Replace ResNet's spatial convolutions with per-pixel local self-attention. *Gap:* good FLOPs/accuracy but poor real latency because the per-query key set varies, blocking efficient batched computation on general hardware.

**Linear/efficient attention (Performer).** Kernel-approximated softmax for linear-time global attention. *Gap:* approximation costs accuracy, and it offers no hierarchy on its own.

## Evaluation settings

- **Image classification — ImageNet-1K** (1.28M training, 50K validation, 1000 classes); optionally pre-train on **ImageNet-22K** (14.2M images, 21,841 classes) and fine-tune on ImageNet-1K. Metric: top-1 (and top-5) accuracy, single center crop. Training follows the data-efficient Transformer recipe: AdamW, cosine schedule with linear warm-up, ~300 epochs from scratch, RandAugment/Mixup/CutMix/random-erasing/stochastic-depth, weight decay, gradient clipping; default input 224², with fine-tuning to larger inputs (e.g., 384²).
- **Object detection & instance segmentation — COCO 2017** (118K train, 5K val, 20K test-dev). The backbone is plugged into standard detection frameworks (Mask R-CNN / Cascade Mask R-CNN, ATSS, RepPoints v2, Sparse R-CNN, HTC) via FPN, in mmdetection. Multi-scale training, AdamW, 1×/3×/6× schedules. Metrics: box AP and mask AP (and AP₅₀, AP₇₅).
- **Semantic segmentation — ADE20K** (25K images, 150 categories; 20K train / 2K val / 3K test). The backbone feeds UperNet in mmsegmentation; AdamW, linear-decay schedule with warm-up, 160K iterations, 512² (or 640²) crops, multi-scale test. Metric: mIoU.
- Efficiency is reported alongside accuracy: parameter count, theoretical FLOPs, and measured throughput / FPS / per-stage attention latency on a V100 GPU.

## Code framework

The pre-method scaffold below is a generic Transformer-backbone-for-vision harness: the pieces that already exist (patchify stem, a stack of pre-norm Transformer blocks whose attention sublayer is a replaceable slot, a downsampling step between stages, a classification head, the DeiT-style training loop). Some sublayers are left as empty slots to be filled in.

```python
import torch
import torch.nn as nn

class Mlp(nn.Module):
    # Standard position-wise FFN: Linear -> GELU -> Linear, ~4x hidden width.
    def __init__(self, dim, hidden_dim, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class PatchEmbed(nn.Module):
    # Split image into non-overlapping patches and linearly embed each (conv with stride=kernel=patch).
    def __init__(self, patch_size=4, in_chans=3, embed_dim=128, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, (H/ps * W/ps), C
        if self.norm is not None:
            x = self.norm(x)
        return x


class Attention(nn.Module):
    # The attention sublayer to be designed. Pre-method: only the generic interface is known.
    def __init__(self, dim, num_heads):
        super().__init__()
        # TODO: design the attention sublayer.
        pass

    def forward(self, x):
        # TODO: return attended features, same shape as x
        pass


class Block(nn.Module):
    # Pre-norm Transformer block: residual around (attention) and around (MLP).
    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads)          # the slot above
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop_path)
        # TODO: stochastic-depth (drop-path) on each residual branch

    def forward(self, x):
        # TODO: x = x + drop_path(attn(norm1(x))); x = x + drop_path(mlp(norm2(x)))
        pass


class Downsample(nn.Module):
    # Inter-stage spatial downsample that builds the pyramid (stride 4 -> 8 -> 16 -> 32).
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        # TODO: halve spatial resolution, increase channels, in token form
        pass

    def forward(self, x):
        pass


class VisionTransformerBackbone(nn.Module):
    # Patchify -> several stages of Blocks with a Downsample between stages -> pooled head.
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=128, depths=(2, 2, 2, 2), num_heads=(4, 8, 16, 32),
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim, norm_layer)
        # TODO: build len(depths) stages; each stage is `depth` Blocks then a Downsample
        #       (except the last); channels double and resolution halves each stage.
        self.stages = nn.ModuleList()  # TODO
        self.norm = norm_layer(embed_dim * 2 ** (len(depths) - 1))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim * 2 ** (len(depths) - 1), num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        # TODO: run stages
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2)).flatten(1)
        return self.head(x)


def build_optimizer(model, lr=1e-3, weight_decay=0.05):
    # DeiT-style optimization: AdamW + cosine schedule w/ warm-up (scheduler omitted here).
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_step(model, images, targets, optimizer, criterion):
    optimizer.zero_grad()
    loss = criterion(model(images), targets)   # CE with label smoothing + Mixup/CutMix targets
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()
```
