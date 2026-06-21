Masked language modeling works for text because there is already a tokenizer: predicting a missing WordPiece id is a classification problem over meaningful units. Vision Transformers turn an image into a sequence of patch tokens, so the same masking recipe seems natural, but the target is the hard part. Raw pixels are continuous and noisy; regressing them pushes the model toward texture and short-range color rather than semantics. BEiT shows that masked image modeling can work if each patch is first mapped to a discrete visual token by an offline discrete VAE, but that tokenizer is fixed, trained on outside data, and its targets encode low-level reconstruction detail rather than the semantic structure we want the encoder to learn. The missing piece is a visual tokenizer that is semantic, jointly trained, and adapted to the actual training domain. Without it, masked image modeling is forced to choose between pixel regression, which is easy to define but weak in transfer, and a frozen discrete tokenizer, which is strong enough to define a target but carries the biases and reconstruction priorities of a separately trained model. What is needed is a target that can emerge from the data itself while the main encoder is being trained.

The key observation is that BEiT's masked-patch loss and DINO's class-token self-distillation loss are the same kind of object: a cross-entropy from a teacher distribution to a student distribution. In BEiT the teacher is a frozen dVAE tokenizer read off patches; in DINO the teacher is an exponential moving average of the student read off the class token. Replacing the frozen tokenizer with the online EMA teacher, but reading it on patch tokens, gives a masked image modeling target that is learned during the same training run. That is the proposal: iBOT, which stands for Image BERT Pre-Training with Online Tokenizer.

iBOT keeps DINO's global cross-view self-distillation on the class token, because that is what forces the network to build semantic structure. The student sees masked augmented views, while the EMA teacher sees clean augmented views. The teacher's backbone plus its projection head acts as the tokenizer, producing a soft K-dimensional distribution for every patch. At each masked patch, the student is trained to match the teacher's distribution for that patch in the clean view. The patch projection head is shared with the class-token head, so the semantics learned through class-token agreement flow directly into the patch targets. Separate centering and sharpening are used for the class and patch streams, with their own EMA centers and teacher temperature schedules, because patch content is more varied than whole-image content.

Masking is applied in contiguous blocks rather than scattered patches, so the model must reason from larger context instead of borrowing from immediate neighbors. The masking ratio is sampled with some spread, and random MIM is used in multi-crop training: each image is either left clean, behaving like a standard DINO sample, or masked with a sampled ratio. This prevents the instability that comes from mixing masked global crops and clean local crops in the same batch. The two losses, class self-distillation and masked patch distillation, are simply summed with equal weight, and the optimization follows the standard self-supervised recipe of AdamW, cosine learning-rate schedule, and a teacher EMA momentum that rises toward one.

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
