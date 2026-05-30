# iBOT — Image BERT Pre-Training with Online Tokenizer

## Problem

Masked language modeling made Transformer pretraining scalable in NLP, but its visual analog — masked image modeling (MIM) — needs a *visual tokenizer* to turn a masked patch into a meaningful supervisory target. Prior MIM (BEiT) used an offline, frozen discrete-VAE tokenizer that captures only low-level detail, is not domain-adaptive, and emits one-hot targets ill-suited to ambiguous patches.

## Key idea

BEiT's masked-patch loss and DINO's `[CLS]` self-distillation loss are the *same* cross-entropy from a teacher distribution to a student distribution; they differ only in the teacher (a frozen dVAE φ vs an online EMA teacher θ′) and the token (patch vs `[CLS]`). iBOT replaces the offline tokenizer with the **online tokenizer**: the EMA teacher network, evaluated on the *clean* image, supplies a soft target distribution for each masked patch. The student sees the masked image and recovers, at each masked patch, the teacher's clean-image distribution. A DINO-style cross-view `[CLS]` self-distillation runs alongside to make that tokenizer semantically meaningful (MIM alone yields near-useless features); the `[CLS]` and patch tokens share one projection head so the earned semantics flows into the patch targets.

## Final objective

Two augmented views u, v of image x; blockwise-mask each into û, v̂. Student processes masked views; teacher (EMA of student) processes clean views.

- Masked image modeling (in-view, masked positions only):
  `L_MIM = − Σ_{i=1}^N m_i · P_θ′^patch(u_i)ᵀ log P_θ^patch(û_i)`, symmetrized over (v̂, v).
- Cross-view `[CLS]` self-distillation:
  `L_[CLS] = − P_θ′^[CLS](v)ᵀ log P_θ^[CLS](û)`, symmetrized.
- Total: `L = L_[CLS] + L_MIM` (summed, no scaling).

Teacher targets are centered then sharpened: `P = softmax((teacher_logit − center)/τ_t)`, detached; student uses `softmax(student_logit/τ_s)`. Separate centers/temperatures for `[CLS]` (C, τ_t) and patches (C′, τ_t′). The teacher = backbone f_t + shared patch head h_t is the online tokenizer.

Key settings: ViT-S/16, B/16, L/16 (224 input, 196 patch tokens). Shared 3-layer-MLP head with ℓ₂ bottleneck, weight-normed last layer, output K = 8192. Soft (not one-hot) targets. Student temp 0.1; teacher temps warmed from low values. Teacher EMA momentum cosine 0.996 → 1. AdamW, batch 1024, lr = 5e−4 × batch/256, 10-epoch warmup then cosine. Blockwise masking, ratio 0.3 ± 0.2. With multi-crop (2×224² + 10×96²), use **random MIM**: per image, ratio 0 (pure DINO) with prob 0.5 or in 0.3 ± 0.2 with prob 0.5, removing the masked-vs-clean distribution mismatch that destabilizes naive multi-crop. Center momenta 0.9.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class iBOTHead(nn.Module):
    """Shared head: same final layer serves [CLS] (x[:,0]) and patch tokens (x[:,1:])."""
    def __init__(self, in_dim, out_dim, patch_out_dim, nlayers=3,
                 hidden_dim=2048, bottleneck_dim=256, shared_head=True):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers += [nn.Linear(hidden_dim, bottleneck_dim)]
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1); self.last_layer.weight_g.requires_grad = False
        self.last_layer2 = self.last_layer if shared_head else \
            nn.utils.weight_norm(nn.Linear(bottleneck_dim, patch_out_dim, bias=False))

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x[:, 0]), self.last_layer2(x[:, 1:])

class iBOTLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ngcrops, ncrops,
                 student_temp=0.1, center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0):
        super().__init__()
        self.student_temp = student_temp
        self.ngcrops, self.ncrops = ngcrops, ncrops
        self.cm, self.cm2 = center_momentum, center_momentum2
        self.lambda1, self.lambda2 = lambda1, lambda2
        self.register_buffer("center",  torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))

    def forward(self, student_out, teacher_out, student_mask, temp, temp2):
        s_cls, s_patch = student_out                 # student: masked views
        t_cls, t_patch = teacher_out                 # teacher: clean views (online tokenizer)
        s_cls   = (s_cls   / self.student_temp).chunk(self.ncrops)
        s_patch = (s_patch / self.student_temp).chunk(self.ngcrops)
        t_cls_c   = F.softmax((t_cls   - self.center)  / temp,  dim=-1).detach().chunk(self.ngcrops)
        t_patch_c = F.softmax((t_patch - self.center2) / temp2, dim=-1).detach().chunk(self.ngcrops)

        L_cls = L_mim = n1 = n2 = 0
        for q in range(len(t_cls_c)):
            for v in range(len(s_cls)):
                if v == q:                           # masked-patch reconstruction
                    l = torch.sum(-t_patch_c[q] * F.log_softmax(s_patch[v], dim=-1), dim=-1)
                    m = student_mask[v].flatten(-2, -1).float()
                    l = torch.sum(l * m, dim=-1) / m.sum(dim=-1).clamp(min=1.0)
                    L_mim += l.mean(); n2 += 1
                else:                                # cross-view [CLS] self-distillation
                    l = torch.sum(-t_cls_c[q] * F.log_softmax(s_cls[v], dim=-1), dim=-1)
                    L_cls += l.mean(); n1 += 1
        L_cls = L_cls / n1 * self.lambda1
        L_mim = L_mim / n2 * self.lambda2
        self.update_center(t_cls, t_patch)
        return L_cls + L_mim

    @torch.no_grad()
    def update_center(self, t_cls, t_patch):
        self.center  = self.center  * self.cm  + t_cls.mean(0, keepdim=True)          * (1 - self.cm)
        self.center2 = self.center2 * self.cm2 + t_patch.mean(1).mean(0, keepdim=True) * (1 - self.cm2)

def train_step(images, masks, student, teacher, loss_fn, opt, ema_m, temp, temp2, n_global):
    t_out = teacher(images[:n_global])                       # clean global crops
    s_out = student(images[:n_global], mask=masks[:n_global])  # masked global crops
    student.backbone.masked_im_modeling = False
    s_local_cls = student(images[n_global:])[0] if len(images) > n_global else None
    student.backbone.masked_im_modeling = True
    if s_local_cls is not None:
        s_out = (torch.cat([s_out[0], s_local_cls]), s_out[1])
    loss = loss_fn(s_out, t_out, masks, temp, temp2)
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(ema_m).add_((1 - ema_m) * ps.detach().data)
```
