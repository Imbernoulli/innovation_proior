**Problem.** ViT-B trained on ImageNet-1k alone, with its own recipe, reaches only 77.91% top-1 (79.35%
after known off-the-shelf training-pipeline tweaks) — well short of convnets at similar size. The
architecture has almost no built-in spatial prior; a convnet's locality/translation-equivariance is
supplied by the operator itself, a Transformer's has to come from the data it is shown. Before touching
the architecture, isolate how much of the gap is a training-procedure gap: swap in a bundle of
augmentation and regularization that manufactures, on the data side, some of the invariance a convolution
would give away structurally.

**Recipe (unmodified ViT-B/16/S/Ti architecture).**

- Optimizer: AdamW, lr = 0.0005 × batch_size/512, cosine decay, 5-epoch warmup, weight decay 0.05
  (down from the as-published 0.3 — tuned for a much larger pretraining set, likely too strong a
  regularizer here)
- Batch size 1024, 300 epochs, truncated-normal initialization (transformers are init-sensitive; several
  untested initializations are known not to converge)
- Augmentation: RandAugment(9, 0.5) chosen over AutoAugment, Mixup(0.8), CutMix(1.0), Random
  Erasing(0.25)
- Regularization: Stochastic Depth(0.1), Label Smoothing(0.1), Repeated Augmentation (3 views/batch); no
  dropout (stacking it on top of stochastic depth + heavy augmentation risks under-fitting a network with
  no architectural head start)
- Optional fine-tune at higher resolution: bicubic-interpolated positional embeddings (preserves vector
  norm, unlike bilinear, which shrinks it and can otherwise sharply hurt a pre-trained network)

**Architecture.** Completely unmodified: patch embed (16×16 conv/linear) → class token + learned
positional embedding → 12 pre-norm Transformer blocks (MSA + GeLU FFN, both residual) → LayerNorm → linear
classifier on the class token only. No distillation signal of any kind.

```python
import torch
from torch import nn

# --- classifier readout: class token only, no distillation ---
class DataEfficientViT(nn.Module):
    def __init__(self, img=224, patch=16, in_ch=3, dim=768, depth=12, heads=12, num_classes=1000,
                 drop_path=0.1):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        n_patches = (img // patch) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        # stochastic depth: linearly increasing drop rate across the stack
        dpr = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, drop_path=dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)            # (B, N, dim)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])                                     # class-token logits only


def training_loss(student_outputs, labels, teacher=None, inputs=None):
    # no teacher this rung: plain label-smoothed cross-entropy against (possibly Mixup/CutMix-mixed) labels
    logp = torch.log_softmax(student_outputs, dim=-1)
    eps = 0.1
    n_classes = student_outputs.size(-1)
    with torch.no_grad():
        smooth = torch.full_like(logp, eps / n_classes)
        smooth.scatter_add_(1, labels.argmax(-1, keepdim=True) if labels.dim() > 1 else labels[:, None],
                             torch.full((labels.size(0), 1), 1.0 - eps, device=logp.device))
    return -(smooth * logp).sum(-1).mean()


def resize_pos_embed(pos_embed, old_grid, new_grid):
    # bicubic (not bilinear): approximately preserves vector norm across the interpolation
    cls_pe, patch_pe = pos_embed[:, :1], pos_embed[:, 1:]
    dim = patch_pe.shape[-1]
    grid = patch_pe.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
    grid = torch.nn.functional.interpolate(grid, size=new_grid, mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_grid[0] * new_grid[1], dim)
    return torch.cat([cls_pe, grid], dim=1)


training_recipe = dict(
    optimizer="AdamW", base_lr=0.0005, lr_base_batch=512, batch_size=1024, weight_decay=0.05,
    schedule="cosine", warmup_epochs=5, epochs=300, init="truncated_normal",
    randaugment=(9, 0.5), mixup=0.8, cutmix=1.0, random_erasing=0.25,
    stochastic_depth=0.1, label_smoothing=0.1, repeated_augmentation=3, dropout=None, ema=None,
)
```

**Test.** Train Ti/S/B under this recipe, ImageNet-1k only, 224²; read off top-1. Then fine-tune the
resulting Base-size model at 384² with bicubic-resized positional embeddings and read off top-1 again, alongside a
sweep to intermediate sizes to see how top-1 moves with resolution alone, holding the recipe fixed.
