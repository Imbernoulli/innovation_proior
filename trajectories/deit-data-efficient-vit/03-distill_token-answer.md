**Problem.** Hard-label distillation beat soft distillation cleanly at every size, but it was implemented
by asking one token — the class token — to satisfy two only loosely correlated targets out of the same
768-dimensional embedding: the true label and the teacher's (frequently different) hard decision. Give
the teacher signal its own token instead, so the two objectives no longer have to compromise inside a
single shared vector.

**Design.** Add a second special token, the **distillation token**, alongside the class token and the
patch tokens, before the first Transformer block. It is a free parameter with no fixed spatial content,
exactly like the class token, and passes through the same self-attention layers, attending to and being
attended by every patch token and the class token. The only asymmetry between the two special tokens is
their target: the class token's final-layer output is read by a linear classifier trained against the
true label; the distillation token's final-layer output is read by a *separate* linear classifier trained
against the teacher's hard pseudo-label (the loss from the previous rung, unchanged: y_t =
argmax_c Z_t(c), teacher re-evaluated on the same augmented crop). Both tokens, and both classifiers, are
learned purely by backpropagation through the shared attention stack — nothing about the mechanism is
different from how the class token alone came to specialize on its target in every prior rung.

**Control (to isolate signal from bare capacity).** Run a duplicate-class-token variant alongside the
real design: a second free token added the same way, but its classifier is trained against the *same*
true label as the class token, not the teacher's pseudo-label. Same architecture, same recipe, same
teacher, same three sizes. Track cosine similarity between the two tokens' embeddings at the input (right
after the tokens are formed) and at the final layer (right before each classifier reads its own token),
to distinguish "the two tokens converged to the same representation" from "the two tokens specialized on
different information."

**Readout.** Three options evaluated at every size, none requiring new machinery: class-token classifier
alone, distillation-token classifier alone, or late fusion — sum the two classifiers' softmax outputs and
take the argmax. Working default is the fused readout.

```python
import torch
from torch import nn
import torch.nn.functional as F


class DistilledViT(nn.Module):
    """Adds a distillation token alongside the class token; two independent linear heads."""
    def __init__(self, img=224, patch=16, in_ch=3, dim=768, depth=12, heads=12, num_classes=1000,
                 drop_path=0.1):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        n_patches = (img // patch) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, dim))          # new: independent free parameter
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 2, dim))   # +2: cls + dist
        dpr = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, drop_path=dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)                         # trained on TRUE label
        self.head_dist = nn.Linear(dim, num_classes)                    # trained on TEACHER pseudo-label
        for p in (self.cls_token, self.dist_token, self.pos_embed):
            nn.init.trunc_normal_(p, std=0.02)

    def forward_features(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)              # (B, N, dim)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        dist = self.dist_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, dist, x], dim=1) + self.pos_embed           # order: cls, dist, patches
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1]                                         # (class embed, distill embed)

    def forward(self, x):
        x_cls, x_dist = self.forward_features(x)
        y, y_dist = self.head(x_cls), self.head_dist(x_dist)
        return y, y_dist                                                # both heads at train time


def readout(y, y_dist, mode="fused"):
    if mode == "class_only":
        return y.argmax(-1)
    if mode == "dist_only":
        return y_dist.argmax(-1)
    if mode == "fused":                                                 # working default
        return (y.softmax(-1) + y_dist.softmax(-1)).argmax(-1)
    raise ValueError(mode)


def training_loss(student_outputs, labels, teacher=None, inputs=None):
    y, y_dist = student_outputs
    with torch.no_grad():
        y_t = teacher(inputs).argmax(dim=-1)                            # same augmented crop, per rung 2
    ce_true = F.cross_entropy(y, labels)
    ce_teacher = F.cross_entropy(y_dist, y_t)
    return 0.5 * ce_true + 0.5 * ce_teacher                             # hard-label objective, unchanged


class DuplicateClassTokenControl(DistilledViT):
    """Isolates capacity from signal: second token has the SAME target as the class token."""
    def training_loss(self, student_outputs, labels, teacher=None, inputs=None):
        y, y_second = student_outputs
        return 0.5 * F.cross_entropy(y, labels) + 0.5 * F.cross_entropy(y_second, labels)


def token_cosine_similarity(embed_a, embed_b):
    # diagnostic: run at the token-formation point (input) and after the final block (output)
    return F.cosine_similarity(embed_a, embed_b, dim=-1).mean()
```

**Test.** Train Ti/S/B with the two-token architecture and the fixed hard-label objective, teacher and
recipe otherwise unchanged from rungs 1–2. Report all three readouts (class only, distillation only,
fused) at every size, alongside the duplicate-class-token control run under identical conditions, plus
the input- and final-layer cosine similarity between the two tokens in both the real design and the
control — to tell whether any gain traces to the distinct target or merely to the added free token.
