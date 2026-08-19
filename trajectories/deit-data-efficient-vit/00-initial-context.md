## Research question

A pure-attention image classifier — a Transformer that ingests an image as a sequence of patch tokens
with no convolutions at all — has been shown to match strong convolutional networks on ImageNet-1k
top-1 accuracy, but only after pre-training on a private dataset of hundreds of millions of labeled
images with heavy compute. Trained from scratch on ImageNet-1k alone (~1.28M training images, no extra
data), the same architecture underperforms: a ViT-B/16 trained with the original ViT recipe (AdamW,
weight decay 0.3, dropout 0.1, gradient clipping, no strong augmentation) reaches 77.91% top-1. An
existing engineering pipeline (the timm training library) already lifts this same architecture, still on
ImageNet-1k only, to 79.35% top-1 through unspecified training-procedure tweaks — the starting bar to
beat. The question: can a convolution-free image Transformer be trained, on a single 8-GPU node in a few
days, to be competitive with convnets of similar size and throughput, using *only* ImageNet-1k? A second,
related question: once such a model exists, how should it be distilled from a strong teacher classifier?

## Background

**Inductive priors in convnets and transformers.** Convolutions bake in strong priors — locality and
translation equivariance — that match natural images, so a convnet needs comparatively little data to
generalize. A Transformer's self-attention has almost no such built-in spatial prior; it learns the
structure of images from data, which is the presumed reason it needs so much of it.

**The Transformer block.** Self-attention computes, for queries Q, keys K, values V (all linear
projections of the same input sequence X, so K holds N key vectors and the attention is among all N
input vectors), Attention(Q,K,V) = softmax(QKᵀ/√d)·V; the √d normalization keeps the softmax from
saturating as the head dimension grows. Multi-head self-attention runs h such heads in parallel, each
producing an N×d output, concatenates them to N×dh, and reprojects to N×D. A full block adds, on top of
the attention sublayer, a feed-forward network (FFN) of two linear layers with a GeLU between them that
expands D→4D then contracts 4D→D; both sublayers are residual (skip connections) and layer-normalized.
There is no batch normalization, so batch size can be reduced without hurting the layer statistics.

**Treating an image as tokens.** A fixed-size RGB image is cut into N non-overlapping patches of
16×16 pixels (N = 14×14 = 196 at 224² input); each patch (3·16·16 = 768 numbers) is linearly projected
to the model width D. Because the block is permutation-invariant, a positional embedding (one vector per
patch position, learned) is added to the patch tokens before the first block. A special trainable
**class token** is appended to the patch sequence, passes through all the layers interacting with the
patches purely via self-attention (the supervision signal at training time comes only from this token;
the patch tokens are the model's only variable input), and its final output is linearly projected to the
class logits — replacing the global pooling a convnet would use. The network thus processes N+1 tokens
but reads out only the class token. Three sizes are on the table, all with head dimension fixed at 64 so
only width and head count change: a Tiny size (D=192, 3 heads, 12 layers, 5M params, the ResNet-18-scale
counterpart), a Small size (D=384, 6 heads, 12 layers, 22M params, the ResNet-50-scale counterpart), and
a Base size (D=768, 12 heads, 12 layers, 86M params, architecturally identical to ViT-B).

**Train-low, fine-tune-high resolution (FixRes).** Training at a lower resolution and fine-tuning at a
higher one is faster and, under strong augmentation, more accurate than training at the high resolution
directly. With a fixed patch size, raising the input resolution increases the number of patches N; the
transformer blocks and the classifier head handle the longer token sequence unchanged (nothing in a
Transformer block depends on sequence length), but the N positional embeddings — one per patch position
— must be resized to the new grid. A naive bilinear interpolation of a positional-embedding vector from
its neighbors reduces its ℓ2-norm relative to its neighbors; a pre-trained transformer is not adapted to
these artificially shrunk vectors, producing a large accuracy drop if used without further training.
Bicubic interpolation approximately preserves vector norm and is the safer default before fine-tuning.

**Knowledge distillation.** A student can be trained to imitate a teacher classifier's output rather
than (or in addition to) the ground-truth label. Soft distillation (Hinton et al.) minimizes the
Kullback–Leibler divergence between the temperature-softened softmax outputs of teacher and student,
typically blended with the ground-truth cross-entropy:

  L = (1−λ)·L_CE(ψ(Z_s), y) + λ·τ²·KL(ψ(Z_s/τ), ψ(Z_t/τ)),

where Z_s, Z_t are student/teacher logits, ψ is softmax, τ is the temperature, and the τ² factor
compensates for the 1/τ scaling of the softened gradients so their magnitude stays comparable as τ
varies. Separately, it is known from the distillation literature that a teacher's soft labels act
somewhat like label smoothing, and — more relevant under heavy data augmentation — that a teacher
re-evaluated on the *same augmented crop* the student sees can correct for cases where an aggressive crop
or mix has changed what is actually visible in the image relative to its dataset label (e.g. a "cat"
image whose crop no longer contains the cat).

**The augmentation / regularization toolbox.** Available training ingredients include AdamW (decoupled
weight decay), cosine learning-rate schedules with warmup, AutoAugment and RandAugment (learned
augmentation policies), Mixup and CutMix (mixing images and labels), random erasing, label smoothing,
stochastic depth (randomly dropping residual blocks, which eases convergence of deep transformer stacks),
dropout, repeated augmentation (multiple augmented views of the same image within a batch), and
exponential moving averages of weights. Learning rate is commonly scaled with batch size. Transformers
are known to be sensitive to weight initialization; a truncated normal distribution (Hanin & Rolnick) is
a documented safe choice after several other initializations failed to converge in preliminary tests.

**Teacher candidates.** Convolutional classifiers of varying capacity are available as potential teachers
for distillation, spanning a range of accuracies (roughly 80–83% top-1 on ImageNet-1k for the RegNetY
family at 4–16 GFLOPs), as well as a Transformer of comparable accuracy to the student itself. For every
distillation experiment below, the default teacher is fixed to a RegNetY-16GF convnet (84M params),
trained with the same data and the same data augmentation as the student, reaching 82.9% top-1 itself.

## Baselines

**The large-scale-pretrained image Transformer (ViT; Dosovitskiy et al.).** The architecture this
exploration keeps unchanged: patches → linear embedding → class token + positional embedding → stack of
Transformer blocks → class-token classifier. Trained on ~300M private images it reaches strong ImageNet
accuracy; trained on ImageNet-1k alone with its own recipe (weight decay 0.3, gradient clipping, dropout,
no strong augmentation) it reaches only 77.91% top-1, and the stated conclusion was that such
transformers "do not generalize well when trained on insufficient amounts of data."

**Convolutional networks (EfficientNet, RegNet, ResNet).** The accuracy/throughput yardstick on
ImageNet-1k. They are data-efficient (strong architectural priors), train well with SGD, and currently
define the accuracy-vs-speed frontier at every parameter count considered here.

## Evaluation settings

The benchmark is ImageNet-1k classification (1,281,167 training images, 1000 classes), top-1 accuracy on
the standard 50,000-image validation set, with companion test sets ImageNet-Real and ImageNet-V2 used
elsewhere to check for overfitting to the original validation set (not the primary tracked metric here).
Models are also compared on throughput (images/second on a single 16GB V100 GPU, largest batch size per
model, averaged over 30 runs) and parameter count. Training is at 224² with optional fine-tuning at 384²
(or other sizes); a full 300-epoch training run at the Base size takes 53 hours on a single 8-GPU node. No external
training data is used at any point in this exploration.

## Code framework

The available primitives are: a Transformer-block library (multi-head self-attention, GeLU FFN, layer
norm, residual connections); a patch-embedding projection; learnable class-token and positional-embedding
parameters; AdamW with a cosine schedule and warmup; the augmentation/regularization toolbox above; and a
softmax cross-entropy loss. The scaffold assembles a patch-token transformer with a class token and a
classifier head, and leaves empty the slots this exploration must fill: how to read out the prediction(s),
how a teacher is brought to bear during training, and how the positional embeddings are handled when
fine-tuning at higher resolution.

```python
import torch
from torch import nn


class TransformerBlock(nn.Module):
    """Standard pre-existing block: residual MSA + residual GeLU-FFN, layer-normed."""
    def __init__(self, dim, heads, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, act=nn.GELU)
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchTokenTransformer(nn.Module):
    def __init__(self, img=224, patch=16, in_ch=3, dim=768, depth=12, heads=12, num_classes=1000):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        n_patches = (img // patch) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)            # (B, N, dim)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.norm(self.blocks(x))
        # TODO: read out prediction(s) from the relevant token(s)
        raise NotImplementedError


def training_loss(student_outputs, labels, teacher=None, inputs=None):
    # TODO: supervised loss, plus a teacher-based signal and how to combine them
    raise NotImplementedError


def resize_pos_embed(pos_embed, old_grid, new_grid):
    # TODO: resize the N positional embeddings to a new patch grid for higher-res fine-tuning
    raise NotImplementedError
```
