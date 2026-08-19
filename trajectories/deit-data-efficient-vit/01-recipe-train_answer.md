The architecture is not what I want to touch first. A ViT-B trained on ImageNet-1k alone, with the
recipe it was introduced with, sits at 77.91% top-1, and an off-the-shelf training pipeline — without
changing a single weight of the network — has already pushed that same architecture to 79.35% just by
adjusting how it is trained. A two-point jump from procedure alone, with zero architectural change, is
reason enough to test a full hypothesis before I let myself touch a single Transformer block: the gap
between this convolution-free classifier and a data-efficient convnet might not be an architecture gap at
all. It might be a training-procedure gap, dressed up to look architectural because nobody has yet run
this network through the kind of tuning convnets received over a decade.

The reason procedure should matter this much for this particular architecture is structural. Convolutions
bake locality and translation equivariance directly into the weight-sharing pattern of the operator — a
convnet does not have to be shown that a cat shifted three pixels left is still a cat, the convolution
guarantees it. Self-attention carries no such constraint: every patch token can attend to every other
patch token under a fully learned weighting, so whatever spatial structure the network respects has to be
induced from the training distribution itself. On a genuinely huge dataset that induction happens on its
own; on 1.28M images it may not. That reframes the task: rather than add locality to the architecture, I
want the training procedure to manufacture, through the data the network actually sees, some of the
invariance a convolution would have supplied structurally — and augmentation and regularization are
exactly the tools that can do that job without touching a single weight-sharing pattern in the model.

I propose a coherent bundle rather than a single trick, chosen so that each piece substitutes for a
specific missing prior without either starving the network of fitting capacity or leaving it so
unconstrained that it memorizes. AdamW stays, since it is already the as-published optimizer and nothing
here motivates a regression to SGD on a network with no batch normalization to smooth its landscape. Weight
decay drops from the as-published 0.3 to 0.05: that coefficient was tuned against a training set two
orders of magnitude larger, and a decay that strong is a bet that overfitting is the dominant risk, which
is backwards for a model with almost no structural prior fitting a comparatively small, fixed image set
directly. Batch size comes down to 1024 (from 4096), scaling the learning rate proportionally, purely a
compute fit to a single 8-GPU node — nothing in the architecture depends on a large batch for stable
statistics once batch normalization is gone. Truncated-normal initialization is a prerequisite, not a
lever: transformers are known to be sensitive to init, and several untested options are known outright
not to converge.

For augmentation the guiding principle is the same throughout: manufacture the invariances the
convolution does not supply. RandAugment composes randomized photometric and geometric transforms,
directly diversifying what a "cat" looks like beyond any single dataset crop; I take it over AutoAugment
because AutoAugment's policy was searched against a convnet's biases already in place, and there is no
reason to inherit a search target tuned for a different inductive bias when this architecture starts with
none. Mixup and CutMix convexly or spatially combine two images and their labels — beyond the usual
boundary-smoothing effect, a CutMix-composited image literally contains two objects, so the network
cannot get away with keying on one dominant local cue, which directly counters an unconstrained attention
pattern collapsing onto a shortcut. Random erasing knocks out rectangular patches outright, a blunter
version of the occlusion robustness a convolution's local pooling gives away for free. Stochastic depth,
which randomly drops whole residual sub-blocks, is documented specifically for easing convergence of deep
transformer stacks — as distinct from ordinary dropout's per-unit masking — and twelve residual blocks
with no batch-norm to keep gradients well-scaled is exactly that setting, so I include it and, on the
same reasoning, leave plain dropout out: layering both regularizers on top of an already-heavy
augmentation stack risks under-fitting a network with no architectural head start to spare. Label
smoothing at 0.1 softens the hard-label target, consistent with training under transformations that
already make the "true" label somewhat approximate for a given augmented view. Repeated augmentation
draws multiple independently augmented views of the same source image within a batch rather than one
view per distinct image, raising gradient-signal diversity per batch without requiring more distinct
images than the dataset offers — exactly the kind of data multiplication a data-limited, prior-free
architecture should want.

I am applying this bundle unchanged across all three sizes, Ti, S, and B, because nothing in the
diagnosis above is size-specific: if a training-procedure fix is the right explanation, it should help a
5M-parameter model for the same underlying reason it helps an 86M-parameter one. Independently of the
recipe question, I also want to know what a completed 224²-trained model gains from testing or
fine-tuning at higher resolution, since the number of patch tokens changes with resolution while nothing
else in a Transformer block depends on sequence length. The one piece that does need adaptation is the
positional embedding, one vector per patch position: I interpolate it with a bicubic kernel rather than
bilinear, because bilinear interpolation systematically shrinks a vector's norm relative to its
un-interpolated neighbors, and a vector far outside the norm range the network was pre-trained on is not
a safe input to hand a model that has not had a chance to adapt to it — bicubic approximately preserves
that norm and should be the gentler default for fine-tuning at a new size.

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
