# DINO: self-DIstillation with NO labels

## Problem

Learn strong visual representations from unlabeled images, with a method simple enough to run unchanged on both Vision Transformers and convnets — no negative pairs, no large batches / memory banks, no clustering / optimal-transport machinery, and no architectural asymmetry. The central difficulty in any negative-free, label-free scheme is **representation collapse**: a degenerate solution where the network outputs the same thing for every image.

## Key idea

Cast self-supervised learning as **knowledge distillation without labels and without an a-priori teacher**. A student network is trained to match the output distribution of a teacher network that is built online as an exponential moving average of the student itself. Both networks share the *same* architecture and turn an image into a `K`-dimensional probability distribution via a temperature softmax; the training signal is the cross-entropy between the teacher's distribution on one view and the student's distribution on a *different* view of the same image. Collapse is prevented not by negatives or a predictor, but by two cheap, complementary operations applied only to the teacher: **centering** (subtract an EMA of teacher logits) and **sharpening** (a low teacher temperature).

## The method

Let `g = h ∘ f` be a backbone `f` plus a projection head `h`, producing a `K`-dim logit vector. Student `g_θs`, teacher `g_θt`, same architecture, different weights. Softmax with temperatures:

```
P_s(x)^(i) = softmax( g_θs(x) / τ_s )_i
P_t(x)^(i) = softmax( (g_θt(x) − c) / τ_t )_i
```

**Multi-crop views.** From an image build a set `V` of two global crops `{x_1^g, x_2^g}` (224², large area) and several local crops (96², small area). All crops go through the student; only the two global crops go through the teacher. This enforces "local-to-global" correspondences.

**Objective** (cross-entropy `H(a,b) = −Σ a log b`, stop-gradient on the teacher):

```
min_θs   Σ_{x ∈ {x_1^g, x_2^g}}   Σ_{x' ∈ V, x' ≠ x}   H( P_t(x), P_s(x') )
```

**Teacher = EMA of student** (momentum encoder), with `λ` on a cosine schedule from 0.996 → 1:

```
θ_t ← λ θ_t + (1 − λ) θ_s
```

**Centering** (EMA bias on teacher logits, batch size `B`, rate `m`):

```
c ← m c + (1 − m) (1/B) Σ_i g_θt(x_i)
```

**Sharpening**: a low teacher temperature `τ_t` (≈0.04–0.07, warmed up from 0.04 over the first epochs), with `τ_s = 0.1` so the teacher target is sharper than the student.

**Why it does not collapse.** There are two collapse modes: the teacher output becomes constant either by one dimension dominating, or by going uniform. Decompose the loss as `H(P_t, P_s) = h(P_t) + D_KL(P_t ‖ P_s)`; a constant teacher output drives `D_KL → 0`. Centering subtracts the mean logit, so no single dimension can dominate — but centering alone flattens the output toward uniform. Sharpening (low `τ_t`) concentrates the distribution and prevents the uniform mode — but alone it lets one dimension dominate. The two push in opposite directions; applied together they balance and hold `D_KL` away from zero. (Without centering, the teacher entropy collapses to 0; without sharpening, to `log K`.)

## Working code

The loss and the training step are the heart.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DINOHead(nn.Module):
    """Backbone -> 3-layer MLP (GELU) -> l2-normalize bottleneck -> weight-normed K-dim layer."""
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        # weight-normalized last layer; gain fixed to 1 (and frozen) stabilizes early training
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)   # l2 bottleneck: stabilizes deep heads
        return self.last_layer(x)


class MultiCropWrapper(nn.Module):
    """One backbone forward per distinct resolution, then the head on concatenated features."""
    def __init__(self, backbone, head):
        super().__init__()
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]), return_counts=True)[1], 0)
        start_idx, output = 0, torch.empty(0, device=x[0].device)
        for end_idx in idx_crops:
            out = self.backbone(torch.cat(x[start_idx:end_idx]))
            output = torch.cat((output, out))
            start_idx = end_idx
        return self.head(output)


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs,
                 student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # warm up the teacher temperature: a too-high temp collapses training early
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
        ))

    def forward(self, student_output, teacher_output, epoch):
        # student: divide by temperature, split per crop (all crops)
        student_out = (student_output / self.student_temp).chunk(self.ncrops)

        # teacher: center then sharpen, softmax, stop-gradient; only the 2 global crops
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss, n_terms = 0, 0
        for iq, q in enumerate(teacher_out):          # over teacher global views
            for v in range(len(student_out)):         # over all student views
                if v == iq:
                    continue                          # skip same-view pairs
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_terms += 1
        total_loss /= n_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)   # first-order stat only
        self.center = self.center * self.center_momentum \
            + batch_center * (1 - self.center_momentum)           # EMA -> batch-size robust


def train_one_epoch(student, teacher, dino_loss, data_loader, optimizer,
                    lr_schedule, wd_schedule, momentum_schedule, epoch):
    for it, (images, _) in enumerate(data_loader):
        it_global = len(data_loader) * epoch + it
        for i, g in enumerate(optimizer.param_groups):
            g["lr"] = lr_schedule[it_global]
            if i == 0:
                g["weight_decay"] = wd_schedule[it_global]

        images = [im.cuda(non_blocking=True) for im in images]
        teacher_output = teacher(images[:2])   # only the 2 global views
        student_output = student(images)        # all crops
        loss = dino_loss(student_output, teacher_output, epoch)

        optimizer.zero_grad()
        loss.backward()
        clip_gradients(student, clip=3.0)
        cancel_last_layer_grads(epoch, student, freeze_last_layer=1)  # freeze head 1st epoch
        optimizer.step()

        # EMA teacher update (momentum encoder); no backprop through teacher
        with torch.no_grad():
            m = momentum_schedule[it_global]    # cosine 0.996 -> 1
            for ps, pt in zip(student.parameters(), teacher.parameters()):
                pt.data.mul_(m).add_((1 - m) * ps.detach().data)


def clip_gradients(model, clip):
    for p in model.parameters():
        if p.grad is not None:
            norm = p.grad.data.norm(2)
            coef = clip / (norm + 1e-6)
            if coef < 1:
                p.grad.data.mul_(coef)


def cancel_last_layer_grads(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None
```

Build (student gets the same head as teacher; teacher starts as a copy, gradients off):

```python
student = MultiCropWrapper(backbone(), DINOHead(embed_dim, out_dim=65536))
teacher = MultiCropWrapper(backbone(), DINOHead(embed_dim, out_dim=65536))
teacher.load_state_dict(student.state_dict())
for p in teacher.parameters():
    p.requires_grad = False

dino_loss = DINOLoss(out_dim=65536, ncrops=2 + n_local,
                     warmup_teacher_temp=0.04, teacher_temp=0.07,
                     warmup_teacher_temp_epochs=30, nepochs=nepochs)
optimizer = torch.optim.AdamW(get_params_groups(student))  # lr = 0.0005 * bs / 256
```

Data augmentation produces the crops: two global crops at 224² (large scale range) and `n_local` local crops at 96² (small scale range), with color jitter, Gaussian blur, and solarization (BYOL-style), bicubic interpolation.

## Defaults

`τ_s = 0.1`, `τ_t` warmed 0.04→0.07 over 30 epochs (keep ≤ ~0.06 once warm), center momentum `m = 0.9`, teacher momentum `λ` cosine 0.996→1, `K = 65536`, bottleneck `d = 256`, head = 3-layer MLP (GELU, hidden 2048) + ℓ2 norm + weight-normed linear, BN-free for ViT, AdamW with `lr = 0.0005·bs/256` (10-epoch warmup, cosine decay), weight decay cosine 0.04→0.4, gradient clip 3.0, freeze last layer for the first epoch. Multi-crop: 2×224² + (e.g.) 8×96². Evaluate frozen features with linear probe or hyperparameter-free weighted k-NN (k=20).
