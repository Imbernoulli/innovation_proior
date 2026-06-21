The problem is to learn strong visual representations from a large collection of images with no labels, using one method that works unchanged for both convolutional networks and patch-based Vision Transformers. The obvious label-free idea is a Siamese matching objective: take two augmented views of the same image and make their embeddings agree. But that objective has a catastrophic trivial optimum: map every image to the same constant vector, get zero loss, and learn nothing. This representation collapse is the central difficulty, and every prior approach is defined mainly by the device it uses to avoid collapse.

Existing ideas fall into three families, each with real costs. Contrastive or instance-discrimination methods such as SimCLR and MoCo treat each image as its own class and push views of different images apart. The negatives make collapse impossible, but they require either enormous batches or a large memory bank of past embeddings, plus a momentum encoder whose only role is to keep those stored embeddings consistent. SwAV replaces negatives with online clustering over learned prototypes and forces balanced assignments across the batch through an iterative Sinkhorn-Knopp optimal-transport step; the balancing prevents collapse, but it couples all samples in the batch and adds algorithmic machinery. BYOL removes both negatives and clustering, instead regressing one view onto another view produced by a slow-moving EMA target with an extra predictor head on only one branch. It works, but the mechanism is subtle and partly mysterious, relying on an asymmetric architecture and reported batch-normalization effects. None of these options is as clean or architecture-agnostic as one would like.

The method I would propose is DINO, which stands for self-DIstillation with NO labels. It reframes self-supervised learning as knowledge distillation without a pretrained teacher: the teacher is constructed online as an exponential moving average of the student itself. Both networks share the same architecture, a backbone followed by a projection head, and there is no extra predictor or architectural asymmetry. Each network produces a K-dimensional distribution via a temperature softmax. Given several augmented crops of one image, the teacher receives only the two high-resolution global crops, while the student receives every crop, including the small local ones. The loss is the cross-entropy between the teacher's distribution on one global view and the student's distribution on a different view of the same image. Because local crops must predict the global distribution, the objective enforces a meaningful local-to-global correspondence.

Collapse in this framework means the teacher output becomes independent of the input. DINO prevents it with two cheap, complementary operations applied only to the teacher. Centering subtracts an exponential moving average of the raw teacher logits before the softmax, which stops any single dimension from dominating every image. Sharpening divides the centered teacher logits by a low temperature tau_t before the softmax, which keeps the distribution peaked and prevents it from flattening to uniform. Centering alone would push the output toward uniform, while sharpening alone would let one dimension take over; used together they balance each other and keep the teacher output genuinely input-dependent. Writing the cross-entropy as teacher entropy plus KL divergence from teacher to student makes this diagnosis concrete: a runaway entropy near zero indicates one-hot collapse, while an entropy near log K indicates uniform collapse, and both are avoided when centering and sharpening are combined.

Several implementation details keep training stable. The projection head is a three-layer MLP with GELU activations, followed by an L2-normalized bottleneck and a final weight-normalized linear layer mapping to K dimensions. The bottleneck bounds magnitudes so the head can be deep and K can be very large. The last layer's gain is fixed during the first epoch so the backbone can stabilize before the head chases the moving teacher. The teacher momentum is cosine-annealed from about 0.996 toward 1, turning the teacher into a slowly improving running ensemble that stays ahead of the student. Because the anti-collapse mechanism is centering plus sharpening rather than batch normalization, the head can be entirely BN-free, which is especially natural for Vision Transformers.

```python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np


class DINOHead(nn.Module):
    """Backbone -> 3-layer MLP (GELU) -> l2-normalized bottleneck -> weight-normed K-dim layer."""
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
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
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
            if isinstance(out, tuple):
                out = out[0]
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
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = (student_output / self.student_temp).chunk(self.ncrops)
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss, n_terms = 0, 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_terms += 1
        total_loss /= n_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum \
            + batch_center * (1 - self.center_momentum)


def train_one_epoch(student, teacher, dino_loss, data_loader, optimizer,
                    lr_schedule, wd_schedule, momentum_schedule, epoch):
    for it, (images, _) in enumerate(data_loader):
        it_global = len(data_loader) * epoch + it
        for i, g in enumerate(optimizer.param_groups):
            g["lr"] = lr_schedule[it_global]
            if i == 0:
                g["weight_decay"] = wd_schedule[it_global]

        images = [im.cuda(non_blocking=True) for im in images]
        teacher_output = teacher(images[:2])
        student_output = student(images)
        loss = dino_loss(student_output, teacher_output, epoch)

        optimizer.zero_grad()
        loss.backward()
        clip_gradients(student, clip=3.0)
        cancel_last_layer_grads(epoch, student, freeze_last_layer=1)
        optimizer.step()

        with torch.no_grad():
            m = momentum_schedule[it_global]
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
