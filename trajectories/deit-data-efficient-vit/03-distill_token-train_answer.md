Hard-label distillation beat soft distillation cleanly and by the same margin at every size: a real gain
from the teacher signal, no extra hyperparameters, same architecture and recipe underneath. That settles
the loss question for now — hard pseudo-labels, recomputed on the same augmented crop the student sees,
are the objective I carry forward. But the way I implemented that objective was architecturally lazy on
reflection: I took one token, the class token, whose single linear classifier already had a job — predict
the true label — and asked the same embedding to simultaneously satisfy a second, only loosely correlated
target, the teacher's hard decision. The teacher is not omniscient; at 82.9% top-1 it disagrees with the
ground truth on roughly one image in six, and on exactly the images where it disagrees, the class token's
shared representation is being pulled toward two different answers for the same image, out of the same
768-dimensional vector, read by the same linear head. That is not obviously the most a single distillation
signal can buy.

I propose giving the teacher signal its own token. Nothing about the class token is architecturally
special — it is one more entry in the sequence, passing through the same self-attention layers as every
patch token, distinguished only by being a free parameter rather than a projected patch, and by having a
classifier trained to read a specific target off its final-layer output. That is the entire mechanism by
which it comes to represent "whatever the classification objective needs." If that mechanism is what lets
one token specialize toward one target purely through attention with the shared patch sequence, a second
token of exactly the same kind — injected alongside the class token before the first block, attending to
and attended by every patch and every other special token — should let two objectives diverge into two
roles instead of fighting for one. Concretely: add a **distillation token**, a new free parameter treated
identically to the class token by the attention mechanism, whose own linear classifier is trained only
against the teacher's hard pseudo-label from the previous rung, while the class token's classifier keeps
training on the true label, undisturbed.

Before spending a training run on this, I want to rule out the boring alternative in advance: maybe any
gain from a second token comes from the extra parameters and extra attention slot alone, regardless of
what its classifier is trained on. If that were the whole story, a second token trained on the *same*
target as the class token — a duplicate class token, initialized independently but pointed at the
identical true label — should do just as well. I am running that control alongside the real design, both
under identical conditions (hard-label objective, same teacher, same recipe, same three sizes), so that
any observed gain from the real design can be attributed to the distinct target rather than to bare
capacity. Beyond raw accuracy I am tracking one diagnostic to help interpret either outcome: the cosine
similarity between the two tokens' learned embeddings, both near the input (right after the tokens are
formed) and at the final layer (right before each classifier reads its own token). A second token that
adds nothing beyond capacity has no incentive to diverge from the first and should converge toward it
regardless of target; a token doing real, distinct work should stay measurably less similar to the class
token than the duplicate-target control's second token does to its own twin.

The last open design choice is how to read a prediction back out with two token-specific classifiers
instead of one. Three options require no new machinery: use only the class token's classifier and treat
the distillation token as a purely training-time auxiliary; use only the distillation token's classifier,
the mirror case; or combine both. My working default for the primary readout is late fusion — sum the two
classifiers' softmax outputs and take the argmax, on the reasoning that if the two tokens really have
specialized on complementary rather than redundant information, each should be able to contribute
whatever the other is missing. I am reporting all three readouts at every size rather than committing to
one in advance, since I do not yet know whether fusion beats either token alone, whether one token already
captures most of the combined benefit, or whether a single-token readout wins for some size.

So the full proposal for this rung: add a distillation token beside the class token, keep the hard-label
loss from the previous rung unchanged but now split across two independent classifiers reading two
independent tokens, evaluate all three possible readouts, and run the duplicate-class-token control in
parallel with token-similarity diagnostics to separate a genuinely useful distinct signal from mere added
capacity.

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
        return self.head(x_cls), self.head_dist(x_dist)                 # both heads at train time


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
    return 0.5 * F.cross_entropy(y, labels) + 0.5 * F.cross_entropy(y_dist, y_t)


class DuplicateClassTokenControl(DistilledViT):
    """Isolates capacity from signal: second token has the SAME target as the class token."""
    def training_loss(self, student_outputs, labels, teacher=None, inputs=None):
        y, y_second = student_outputs
        return 0.5 * F.cross_entropy(y, labels) + 0.5 * F.cross_entropy(y_second, labels)


def token_cosine_similarity(embed_a, embed_b):
    # diagnostic: run at the token-formation point (input) and after the final block (output)
    return F.cosine_similarity(embed_a, embed_b, dim=-1).mean()
```
