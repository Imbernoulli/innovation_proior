# iBOT - Image BERT Pre-Training with Online Tokenizer

## Problem

Masked image modeling needs a target for each hidden patch. Pixel regression is too low-level, while BEiT's offline discrete-VAE tokenizer makes MIM a two-stage pipeline with a fixed tokenizer. iBOT's move is to notice that BEiT's masked-patch loss and DINO's `[CLS]` self-distillation loss are both cross-entropies from a teacher distribution to a student distribution. Replace BEiT's fixed tokenizer with DINO's EMA teacher, but read it on patch tokens.

## Method

For two augmented global views `u` and `v`, form masked views `u_hat` and `v_hat` for the student. The EMA teacher receives the clean global views. The teacher backbone plus patch projection head is the online tokenizer.

The masked image modeling loss for one global view:

`L_MIM(u) = - sum_{i=1}^N m_i P_{theta'}^patch(u_i)^T log P_theta^patch(u_hat_i)`.

The other global view adds the symmetric term `L_MIM(v)`. In the reference code, each image's patch loss is normalized by `mask.sum().clamp(min=1)`, then averaged across the batch and across same-view global-crop pairs.

The class-token loss keeps DINO's cross-view self-distillation, except the student global crops may be masked:

`L_[CLS] = - P_{theta'}^[CLS](view_q)^T log P_theta^[CLS](view_v)`, for teacher global crops `q` and all student crops `v != q`.

Teacher logits are centered and sharpened, then detached:

`P_t = softmax((t - C) / tau_t)`, and `log P_s = log_softmax(s / tau_s)`.

There are separate centers and teacher temperatures for `[CLS]` and patch tokens: `C` has shape `[1, K]`, `C_patch` has shape `[1, 1, K]`. Defaults used in the reference code path include `tau_s = 0.1`, center momenta `0.9`, output dimension `K = 8192`, AdamW, learning-rate warmup for 10 epochs then cosine decay, weight decay `0.04 -> 0.4`, and teacher EMA momentum scheduled from `0.996` to `1`.

The losses are summed without an extra scale:

`L = lambda1 L_[CLS] + lambda2 L_MIM`, with `lambda1 = lambda2 = 1`.

For multi-crop runs, iBOT uses random MIM: per image the prediction ratio is sampled from `{0, Uniform(0.1, 0.5)}` through `--pred_ratio 0 0.3 --pred_ratio_var 0 0.2`. A zero mask gives a clean DINO-style image for the `[CLS]` loss and zero contribution for that image's patch loss.

## Reference-Faithful Core

```python
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class iBOTHead(nn.Module):
    def __init__(self, in_dim, out_dim, patch_out_dim=8192, nlayers=3,
                 hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True,
                 shared_head=True):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers += [nn.Linear(hidden_dim, bottleneck_dim)]
        self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

        if shared_head:
            self.last_layer2 = self.last_layer
        else:
            self.last_layer2 = nn.utils.weight_norm(
                nn.Linear(bottleneck_dim, patch_out_dim, bias=False)
            )
            self.last_layer2.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer2.weight_g.requires_grad = False

    def forward(self, x):
        if x.ndim == 2:
            z = F.normalize(self.mlp(x), dim=-1, p=2)
            return self.last_layer(z)

        z = F.normalize(self.mlp(x), dim=-1, p=2)
        return self.last_layer(z[:, 0]), self.last_layer2(z[:, 1:])


class iBOTLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ngcrops, nlcrops,
                 warmup_teacher_temp=0.04, teacher_temp=0.07,
                 warmup_teacher_patch_temp=0.04, teacher_patch_temp=0.07,
                 warmup_epochs=30, nepochs=800, student_temp=0.1,
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0):
        super().__init__()
        self.student_temp = student_temp
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_epochs),
            np.ones(nepochs - warmup_epochs) * teacher_temp,
        ))
        self.teacher_patch_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_patch_temp, teacher_patch_temp, warmup_epochs),
            np.ones(nepochs - warmup_epochs) * teacher_patch_temp,
        ))

    def forward(self, student_output, teacher_output, student_local_cls,
                student_mask, epoch):
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output
        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        student_cls_chunks = (student_cls / self.student_temp).chunk(self.ncrops)
        student_patch_chunks = (student_patch / self.student_temp).chunk(self.ngcrops)

        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_patch_temp_schedule[epoch]
        teacher_cls_chunks = F.softmax(
            (teacher_cls - self.center) / temp, dim=-1
        ).detach().chunk(self.ngcrops)
        teacher_patch_chunks = F.softmax(
            (teacher_patch - self.center2) / temp2, dim=-1
        ).detach().chunk(self.ngcrops)

        cls_loss = patch_loss = 0.0
        n_cls = n_patch = 0
        for q, teacher_cls_q in enumerate(teacher_cls_chunks):
            for v, student_cls_v in enumerate(student_cls_chunks):
                if v == q:
                    per_patch = torch.sum(
                        -teacher_patch_chunks[q]
                        * F.log_softmax(student_patch_chunks[v], dim=-1),
                        dim=-1,
                    )
                    mask = student_mask[v].flatten(-2, -1)
                    per_image = torch.sum(per_patch * mask.float(), dim=-1)
                    per_image = per_image / mask.sum(dim=-1).clamp(min=1.0)
                    patch_loss = patch_loss + per_image.mean()
                    n_patch += 1
                else:
                    per_image = torch.sum(
                        -teacher_cls_q * F.log_softmax(student_cls_v, dim=-1),
                        dim=-1,
                    )
                    cls_loss = cls_loss + per_image.mean()
                    n_cls += 1

        cls_loss = cls_loss / n_cls * self.lambda1
        patch_loss = patch_loss / n_patch * self.lambda2
        self.update_center(teacher_cls, teacher_patch)
        return {"cls": cls_loss, "patch": patch_loss, "loss": cls_loss + patch_loss}

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cls_center)
            dist.all_reduce(patch_center)
            world = dist.get_world_size()
        else:
            world = 1
        cls_center = cls_center / (len(teacher_cls) * world)
        patch_center = patch_center / (len(teacher_patch) * world)
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)
```

The official training loop uses the teacher on clean global crops, the student on masked global crops, and the student alone on unmasked local crops for `[CLS]` multi-crop terms. After the student optimizer step, teacher parameters are updated by EMA over parameter names shared by student and teacher. The official reproduction command passes `--shared_head true`; the parser also exposes separate student/teacher sharing flags, with teacher sharing enabled by default.
