## Research question

A pure-attention image classifier — a Transformer that ingests an image as a sequence of patch tokens
with no convolutions at all — had just been shown to match or beat strong convolutional networks on
ImageNet. But the demonstration came with a heavy caveat: it required pre-training on a giant private
dataset of hundreds of millions of labeled images and large compute, and its authors concluded that
such transformers "do not generalize well when trained on insufficient amounts of data." The precise
question is whether a convolution-free image Transformer can be trained to be competitive with
convnets of similar size and speed using *only* a mid-sized public dataset (ImageNet-1k, ~1.2M images),
on a single machine in a few days — i.e. whether the data-hunger is intrinsic to the architecture or an
artifact of the training recipe. A second, related question: once such a transformer can be trained,
how should it be *distilled* from a strong teacher, given that the transformer's token-based structure
differs fundamentally from a convnet's?

## Background

**Why convnets were data-efficient and transformers were not.** Convolutions bake in strong priors —
locality and translation equivariance — that match natural images, so a convnet needs comparatively
little data to generalize. A Transformer's self-attention has almost no such built-in spatial prior;
it must learn the structure of images from data, which is why, with a mid-sized dataset, it tended to
underperform. The hypothesis worth testing is that this gap can be closed not by adding convolutions
back, but by supplying the missing "data" through aggressive augmentation and regularization, and by
distilling a teacher's inductive bias into the transformer.

**The Transformer block.** Self-attention computes, for queries Q, keys K, values V (all linear
projections of the same input sequence X, so K = N queries attend over all N inputs),
Attention(Q,K,V) = softmax(QKᵀ/√d)·V; the √d normalization keeps the softmax from saturating as the
head dimension grows. Multi-head self-attention runs h such heads in parallel, concatenates their N×d
outputs to N×dh, and reprojects to N×D. A full block adds, on top of the attention sublayer, a
feed-forward network of two linear layers with a GeLU between them that expands D→4D then contracts
4D→D; both sublayers are residual (skip connections) and layer-normalized. There is no batch
normalization, so batch size can be reduced without hurting statistics.

**Treating an image as tokens.** A fixed-size RGB image is cut into N non-overlapping patches of
16×16 pixels (N = 14×14 = 196 at 224² input); each patch (3·16·16 = 768 numbers) is linearly projected
to the model width D. Because the block is permutation-invariant, a positional embedding (fixed
sinusoidal or learned) is added to the patch tokens before the first block. A special trainable
**class token** is appended to the patch sequence, passes through all the layers interacting with the
patches via self-attention, and its final output is linearly projected to the class logits — replacing
the global pooling a convnet would use. The network thus processes N+1 tokens but reads out only the
class token. Larger and smaller width/depth configurations give models of varying size (e.g. an 86M-
parameter base model, and smaller variants comparable in size to ResNet-50 and ResNet-18).

**Train-low, fine-tune-high resolution.** Training at a lower resolution and fine-tuning at a higher
one (FixRes; Touvron et al.) is faster and more accurate under strong augmentation. With a fixed patch
size, raising the input resolution increases the number of patches N; the transformer and classifier
handle the longer token sequence unchanged, but the N positional embeddings must be resized to the new
grid by interpolation.

**Knowledge distillation.** A student can be trained to imitate a teacher classifier. Soft
distillation (Hinton et al.) minimizes the Kullback–Leibler divergence between the temperature-softened
softmax outputs of teacher and student, typically combined with the ground-truth cross-entropy:

  L = (1−λ)·L_CE(ψ(Z_s), y) + λ·τ²·KL(ψ(Z_s/τ), ψ(Z_t/τ)),

where Z_s, Z_t are student/teacher logits, ψ is softmax, τ is the temperature, and the τ² factor
compensates for the 1/τ scaling of the softened gradients so their magnitude stays comparable as τ
varies.

**The augmentation / regularization toolbox.** Available training ingredients include AdamW (decoupled
weight decay), cosine learning-rate schedules with warmup, AutoAugment and RandAugment (learned
augmentation policies), Mixup and CutMix (mixing images and labels), random erasing, label smoothing,
stochastic depth (randomly dropping residual blocks, which eases convergence of deep transformers),
dropout, repeated augmentation (multiple augmented views of the same image within a batch), and
exponential moving averages of weights. Learning rate is commonly scaled with batch size.

## Baselines

**The large-scale-pretrained image Transformer (ViT; Dosovitskiy et al.).** The architecture this work
adopts: patches → linear embedding → class token + positional embedding → stack of Transformer blocks →
class-token classifier. It reached strong ImageNet accuracy but *only* after pre-training on ~300M
private images, with a training recipe (e.g. weight decay 0.3, gradient clipping, dropout, no strong
augmentation) tuned for that large-scale regime. The gap it leaves: trained on ImageNet-1k alone with
that recipe, it underperforms convnets — its authors' stated conclusion was that such transformers need
very large training sets. Whether a *different recipe* closes this gap on public data is exactly the
open question.

**Convolutional networks (EfficientNet, RegNet, ResNet).** The accuracy/throughput yardstick on
ImageNet-1k. They are data-efficient (strong priors), train well with SGD, and define the
accuracy-vs-speed frontier a convolution-free model must reach. A strong convnet is also the natural
candidate *teacher* for distillation.

## Evaluation settings

The benchmark is ImageNet-1k classification (~1.2M training images, 1000 classes), top-1 accuracy,
with companion test sets ImageNet-Real and ImageNet-V2 to check for overfitting to the original
validation set. Models are compared on accuracy versus throughput (images/second on a V100 GPU) and
parameter count, and on total training cost (GPU-days on a single 8-GPU node). Transfer to downstream
classification (CIFAR-10/100, Oxford Flowers-102, Stanford Cars, iNaturalist) is a secondary yardstick.
Training is at 224² with optional fine-tuning at 384² (and other sizes).

## Code framework

The available primitives are: a Transformer-block library (multi-head self-attention, GeLU FFN, layer
norm, residual connections); a patch-embedding projection; learnable token and positional-embedding
parameters; AdamW with a cosine schedule and warmup; the augmentation/regularization toolbox above; and
a softmax cross-entropy loss. The scaffold assembles a patch-token transformer with a class token and a
classifier head, and leaves empty the slots the method must fill: how to read out the prediction(s),
what extra learning signal a teacher provides and through what mechanism it enters the token sequence,
how the two are combined into one loss, and how the positional embeddings are resized when fine-tuning
at higher resolution.

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
        # TODO: any additional learned token the method introduces?
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        # TODO: any additional classifier head?

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
