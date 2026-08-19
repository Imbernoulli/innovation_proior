The problem is whether a pure-attention image Transformer can compete with convnets on ImageNet-1k without any external data. The original Vision Transformer result suggested such models were intrinsically data-hungry, because it required pre-training on hundreds of millions of private images. That conclusion may have been too strong: the hypothesis pursued here is that the gap came from the training recipe, not the architecture. Convnets win with less data because convolution hard-codes locality and translation equivariance, priors a Transformer must learn from examples. But those missing priors should be supplyable synthetically through stronger augmentation and regularization, and the missing data-efficiency should be recoverable through distillation from a convnet teacher. The remaining question is how to adapt distillation to the Transformer token structure.

The method is DeiT, short for Data-efficient image Transformers. It keeps the exact ViT architecture but retrains it with a recipe tuned for ImageNet-1k. The optimizer is AdamW rather than SGD: with no batch norm and a loss landscape less forgiving of a single global step size, per-parameter adaptive scaling with decoupled weight decay should suit this architecture much better, a prediction checked with a matched-budget comparison between the two before committing to AdamW for the full run. The learning rate is linearly scaled with batch size from a base of 5e-4 and decayed with a cosine schedule, with a short warmup. Weight decay is lowered from the large-scale ViT value of 0.3 down to 0.05, because heavy augmentation should already provide enough regularization on only 1.2M images, and the extra 0.3 is expected to over-constrain convergence at this scale. Initialization uses truncated normal, which keeps activations controlled and is important for stable Transformer training. The augmentation stack is aggressive: RandAugment, Mixup, CutMix, random erasing, and repeated augmentation, where each training batch contains three augmented views of the same images; knockout ablations at matched budget are the check for whether each ingredient earns its place, and Mixup+CutMix together plus repeated augmentation are the two expected to matter most. Label smoothing at 0.1 is applied to the ground-truth labels. Stochastic depth at rate 0.1 replaces dropout: on a model already under heavy augmentation and with no batch norm, dropout is expected to add redundant regularization pressure that slows convergence without helping, while stochastic depth regularizes at the level of whole residual paths, a better match for a stack of residual transformer blocks; the no-batch-norm design should also let the model train well at smaller batch sizes, since there is no batch-dependent normalization statistic to degrade.

The key architectural addition is a distillation token. Besides the standard class token, a second learnable token is prepended to the patch sequence. It passes through every Transformer block, attends to the patch tokens and the class token, and is read out by its own classification head. The class token is trained on the true labels, while the distillation token is trained on hard labels produced by a teacher. The class token and distillation token are predicted to be distinct precisely because their targets differ, not because of any structural difference between them: a control that swaps in a second class token with the same true-label target has nothing pulling it away from the first, so it should collapse toward the first token and add no benefit, whereas the distillation token, trained against the teacher's label — a different target from the true label even though the two frequently agree — should stay measurably distinct and provide real signal. That comparison is the check that validates the design before relying on it. At test time the final prediction is the late fusion of the two heads, averaging their softmax outputs.

The distillation loss is parameter-free hard-label distillation. Instead of matching the full softened teacher distribution, the student simply matches the teacher's top-1 prediction. Because the teacher is evaluated on the same augmented crop seen by the student, its hard label is consistent with the student's input. This is simpler than soft distillation, and is predicted to work better in this setting: a hard label recomputed for exactly the crop the student sees should be a better guide under aggressive augmentation than a softened target computed once, a prediction settled by a head-to-head comparison at matched budget with teacher and student held fixed. Importantly, the teacher is chosen to be a strong convnet rather than another Transformer: distillation should transfer more than the label, it should transfer the teacher's inductive bias, and a convnet's locality/translation-equivariance prior is exactly what the Transformer structurally lacks — a hypothesis checked by comparing a convnet teacher against a same-accuracy transformer teacher on an otherwise identical student and recipe. For higher resolution, the model is trained at 224 by 224 and fine-tuned at 384 by 384. The positional embeddings are resized with bicubic interpolation: bilinear interpolation is provably norm-reducing when averaging non-collinear vectors, and feeding the pretrained model positional vectors at a shrunk magnitude is expected to throw off the attention logits it was tuned to; bicubic's negative side-lobe weights let the interpolated vector overshoot toward its neighbors instead of strictly averaging them, which should approximately preserve the norm. Fine-tuning then adapts the model to the new grid while keeping the teacher signal, rather than dropping to true labels alone mid-fine-tune.

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
