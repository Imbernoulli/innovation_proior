The problem is whether a pure-attention image Transformer can compete with convnets on ImageNet-1k without any external data. The original Vision Transformer result suggested such models were intrinsically data-hungry, because it required pre-training on hundreds of millions of private images. That conclusion was too strong: the gap came from the training recipe, not the architecture. Convnets win with less data because convolution hard-codes locality and translation equivariance, priors a Transformer must learn from examples. But those missing priors can be supplied synthetically through stronger augmentation and regularization, and the missing data-efficiency can be recovered through distillation from a convnet teacher. The remaining question is how to adapt distillation to the Transformer token structure.

The method is DeiT, short for Data-efficient image Transformers. It keeps the exact ViT architecture but retrains it with a recipe tuned for ImageNet-1k. The optimizer is AdamW, not SGD, because SGD undertrains the Transformer by a wide margin. The learning rate is linearly scaled with batch size from a base of 5e-4 and decayed with a cosine schedule, with a short warmup. Weight decay is lowered from the large-scale ViT value of 0.3 down to 0.05, because heavy augmentation already provides enough regularization on only 1.2M images. Initialization uses truncated normal, which is important for stable Transformer training. The augmentation stack is aggressive: RandAugment, Mixup, CutMix, random erasing, and repeated augmentation, where each training batch contains three augmented views of the same images. Label smoothing at 0.1 is applied to the ground-truth labels. Stochastic depth at rate 0.1 replaces dropout, since dropout actually hurts here; the no-batch-norm design also lets the model train well at smaller batch sizes.

The key architectural addition is a distillation token. Besides the standard class token, a second learnable token is prepended to the patch sequence. It passes through every Transformer block, attends to the patch tokens and the class token, and is read out by its own classification head. The class token is trained on the true labels, while the distillation token is trained on hard labels produced by a teacher. The class token and distillation token are distinct only because their targets differ: a control experiment with two class tokens shows them collapsing to cosine similarity 0.999 and giving no benefit, whereas the distillation token stays different enough to provide real signal. At test time the final prediction is the late fusion of the two heads, averaging their softmax outputs.

The distillation loss is parameter-free hard-label distillation. Instead of matching the full softened teacher distribution, the student simply matches the teacher's top-1 prediction. Because the teacher is evaluated on the same augmented crop seen by the student, its hard label is consistent with the student's input. This is simpler than soft distillation and works better in this setting. Importantly, the teacher should be a strong convnet rather than another Transformer: the convnet's inductive biases are exactly what the Transformer lacks, so distilling its decisions transfers those biases into the Transformer's behavior. For higher resolution, the model is trained at 224 by 224 and fine-tuned at 384 by 384. The positional embeddings are resized with bicubic interpolation, because bilinear interpolation shrinks the vector norms and breaks the pretrained representation; fine-tuning then adapts the model to the new grid while keeping the teacher signal.

```python
import torch
from torch import nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Pre-norm residual block: MSA + GeLU FFN."""
    def __init__(self, dim, heads, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.drop_path = drop_path

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class DeiT(nn.Module):
    """Data-efficient image Transformer with class and distillation tokens."""
    def __init__(self, img=224, patch=16, in_ch=3, dim=768, depth=12, heads=12,
                 num_classes=1000, drop_path=0.1):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        n_patches = (img // patch) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 2, dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim, heads, drop_path=dpr[i]) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.head_dist = nn.Linear(dim, num_classes)
        for p in (self.cls_token, self.dist_token, self.pos_embed):
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        dist = self.dist_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, dist, x], dim=1) + self.pos_embed
        x = self.norm(self.blocks(x))
        out_cls = self.head(x[:, 0])
        out_dist = self.head_dist(x[:, 1])
        if self.training:
            return out_cls, out_dist
        return (out_cls.softmax(-1) + out_dist.softmax(-1)) / 2


class HardDistillationLoss(nn.Module):
    """Hard-label distillation: 0.5 CE(true) + 0.5 CE(teacher argmax)."""
    def __init__(self, teacher):
        super().__init__()
        self.teacher = teacher.eval()

    def forward(self, inputs, outputs, labels):
        out_cls, out_dist = outputs
        with torch.no_grad():
            teacher_labels = self.teacher(inputs).argmax(dim=1)
        loss_cls = F.cross_entropy(out_cls, labels)
        loss_dist = F.cross_entropy(out_dist, teacher_labels)
        return 0.5 * loss_cls + 0.5 * loss_dist


def resize_pos_embed(pos_embed, old_grid, new_grid):
    """Bicubic resize of patch positional embeddings for higher-res fine-tuning."""
    cls_dist = pos_embed[:, :2]
    patch_pe = pos_embed[:, 2:]
    d = patch_pe.size(-1)
    patch_pe = patch_pe.reshape(1, old_grid, old_grid, d).permute(0, 3, 1, 2)
    patch_pe = F.interpolate(patch_pe, size=(new_grid, new_grid),
                             mode='bicubic', align_corners=False)
    patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, d)
    return torch.cat([cls_dist, patch_pe], dim=1)
```
