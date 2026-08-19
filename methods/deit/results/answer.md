# DeiT: Data-efficient image Transformers (with distillation through attention)

## Problem
A convolution-free image Transformer (patches → tokens → class-token classifier) had only been shown to
match convnets after pre-training on hundreds of millions of private images, with the conclusion that
such models need very large training sets. Train the same architecture to convnet-competitive ImageNet
accuracy using ImageNet-1k *only*, on a single node in a few days — and devise a distillation procedure
suited to transformers.

## Key ideas
1. **Data-efficient training recipe.** The transformer's data-hunger is hypothesized to be a recipe
   artifact, not intrinsic: strong augmentation + regularization (substituting for missing data) plus
   AdamW, tuned specifically for ImageNet-1k scale, is predicted to close most of the gap to convnet
   accuracy with no convolutions and no external data — validated by comparing the resulting model
   directly against convnets of similar size and speed.
2. **Distillation token.** A new learnable token, used like the class token but supervised by the
   teacher's label instead of the true label, gives the transformer a dedicated distillation pathway.
3. **Hard-label distillation.** Treat the teacher's top-1 prediction as a label — parameter-free, and
   expected to beat soft distillation here because the label stays valid for exactly the augmented crop
   the student sees, unlike a softened target computed once. Expected to work best with a **convnet
   teacher** rather than a transformer teacher: distillation should transfer more than the label, it
   should transfer the teacher's inductive bias, and a convnet's locality/translation-equivariance prior
   is exactly what the transformer structurally lacks.

## Architecture (unchanged from ViT, plus one token)
Image (224²) → 16×16 patches → linear/Conv2d patch embedding (D = 768 for the base model) → prepend a
**class token** and a **distillation token** → add positional embeddings (now N + 2) → stack of
Transformer blocks (residual multi-head self-attention + residual GeLU FFN expanding D→4D→D, layer
norm, no batch norm) → read out the class token (→ class head, true-label target) and the distillation
token (→ distillation head, teacher-label target). Base/Small/Tiny configurations parallel
ViT-B/ResNet-50/ResNet-18 in scale.

## Distillation losses
Soft (baseline):
  L = (1−λ)·L_CE(ψ(Z_s), y) + λ·τ²·KL(ψ(Z_s/τ), ψ(Z_t/τ)),   τ = 3.0, λ = 0.1.
Hard-label (used):
  L = ½·L_CE(ψ(Z_s), y) + ½·L_CE(ψ(Z_s), y_t),   y_t = argmax_c Z_t(c).
The τ² factor in the soft loss compensates the 1/τ² gradient scaling of softened logits. The teacher is
re-evaluated on the *augmented* crop the student sees, so y_t is consistent with the student's input.
The class head is trained against y, the distillation head against y_t.

Why a separate distillation token rather than a second class token: two tokens trained on the same
(true-label) target have nothing in the objective pulling them apart, so they are predicted to collapse
toward each other and add nothing beyond a single class token. The class and distillation tokens instead
have *different* targets (true label vs. teacher label), which should keep them meaningfully distinct
rather than converging to copies — though some convergence toward each other with depth is expected too,
since the two targets usually agree. That prediction — a same-target control token collapsing while the
distillation token stays distinct — is the check that validates the design before relying on it.

## Test-time prediction
Late fusion: add the softmax outputs of the class head and the distillation head, predict the argmax.

## Training recipe (base model, ImageNet-1k)
- Optimizer AdamW rather than SGD: with no batch norm and a loss landscape less forgiving of a single
  global step size, per-parameter adaptive scaling with decoupled weight decay is predicted to train
  this architecture far better — validated by a matched-budget comparison against SGD before committing.
  lr = 5e-4 × batch/512, cosine decay, warmup 5 epochs, weight decay 0.05 (the large-scale recipe's 0.3
  is calibrated for 300M images and is expected to over-regularize on 1.2M under heavy augmentation),
  300 epochs, batch ~1024.
- Truncated-normal initialization (transformers are sensitive to initialization scale).
- Augmentation: RandAugment (favored over AutoAugment as the stronger of the two automatic policies),
  Mixup (p = 0.8), CutMix (p = 1.0), random erasing (p = 0.25); **repeated augmentation** (×3 views per
  batch) is expected to be a key contributor, since it should give a better-conditioned gradient than a
  single noisy view. Label smoothing ε = 0.1.
- Regularization: stochastic depth 0.1 in place of dropout — regularizing whole residual paths should
  suit a transformer stack better than dropping activations, and should avoid the redundant pressure
  dropout would add on top of heavy augmentation. No batch norm, so batch size should be reducible
  without the accuracy penalty a batch-normed model would take.
- Validation: matched-budget knockout ablations (remove one ingredient at a time, everything else held
  fixed) are how each of these calls gets checked before the recipe is locked in — Mixup+CutMix together
  and repeated augmentation are the two ingredients expected to cost the most accuracy if removed.

## Resolution
Train at 224², fine-tune at 384² (FixRes). Patch size fixed, so N grows; the N positional embeddings are
resized to the new grid with **bicubic** interpolation. Bilinear interpolation is provably norm-reducing
when averaging non-collinear vectors, and feeding the pretrained model positional vectors at a shrunk
magnitude is expected to throw off attention logits it was tuned to; bicubic's negative side-lobe weights
let the interpolated vector overshoot toward its neighbors instead, which should approximately preserve
the norm. Keep strong augmentation and the teacher during fine-tuning, rather than dropping either
mid-fine-tune and discarding signal the model still benefits from.

## Code
```python
import torch
from torch import nn
import torch.nn.functional as F


class DistilledTransformer(nn.Module):
    def __init__(self, img=224, patch=16, in_ch=3, dim=768, depth=12, heads=12,
                 num_classes=1000, drop_path=0.1):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        n_patches = (img // patch) ** 2
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, dim))            # new token
        self.pos_embed  = nn.Parameter(torch.zeros(1, n_patches + 2, dim))
        dpr = torch.linspace(0, drop_path, depth)
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads, drop_path=float(dpr[i]))
                                      for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head      = nn.Linear(dim, num_classes)                     # true-label head
        self.head_dist = nn.Linear(dim, num_classes)                     # teacher-label head
        for p in (self.cls_token, self.dist_token, self.pos_embed):
            nn.init.trunc_normal_(p, std=0.02)

    def forward_features(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls  = self.cls_token.expand(x.size(0), -1, -1)
        dist = self.dist_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, dist, x], dim=1) + self.pos_embed
        x = self.norm(self.blocks(x))
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x_cls, x_dist = self.forward_features(x)
        y, y_dist = self.head(x_cls), self.head_dist(x_dist)
        if self.training:
            return y, y_dist
        return (y.softmax(-1) + y_dist.softmax(-1)) / 2                   # late fusion


class HardDistillationLoss(nn.Module):
    def __init__(self, teacher):                                         # convnet teacher
        super().__init__()
        self.teacher = teacher.eval()

    def forward(self, inputs, outputs, labels):
        out_cls, out_dist = outputs
        with torch.no_grad():
            teacher_labels = self.teacher(inputs).argmax(dim=1)
        return 0.5 * F.cross_entropy(out_cls, labels) \
             + 0.5 * F.cross_entropy(out_dist, teacher_labels)


class SoftDistillationLoss(nn.Module):
    def __init__(self, teacher, tau=3.0, lam=0.1):
        super().__init__()
        self.teacher, self.tau, self.lam = teacher.eval(), tau, lam

    def forward(self, inputs, outputs, labels):
        out_cls, out_dist = outputs
        with torch.no_grad():
            t = self.teacher(inputs)
        T = self.tau
        kd = F.kl_div(F.log_softmax(out_dist / T, 1), F.log_softmax(t / T, 1),
                      reduction='sum', log_target=True) * (T * T) / out_dist.numel()
        return (1 - self.lam) * F.cross_entropy(out_cls, labels) + self.lam * kd


def resize_pos_embed(pos_embed, old_grid, new_grid):
    cls_dist, patch_pe = pos_embed[:, :2], pos_embed[:, 2:]
    D = patch_pe.size(-1)
    patch_pe = patch_pe.reshape(1, old_grid, old_grid, D).permute(0, 3, 1, 2)
    patch_pe = F.interpolate(patch_pe, size=(new_grid, new_grid),
                             mode='bicubic', align_corners=False)        # norm-preserving
    patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, D)
    return torch.cat([cls_dist, patch_pe], dim=1)
```
