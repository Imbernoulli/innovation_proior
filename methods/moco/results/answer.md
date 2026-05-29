# Momentum Contrast (MoCo)

## Problem

Learn transferable visual representations without labels. The view that organizes
the method: contrastive learning is a **dictionary look-up** — an encoded query `q`
must match its positive key `k_+` (another augmented view of the same image) and be
dissimilar to many negative keys. A good dictionary of keys should be **large** (the
InfoNCE mutual-information bound `I ≥ log N − L` tightens with more negatives) and
**consistent** (all keys encoded by nearly the same encoder, so the dot-product
comparisons are commensurable). Prior mechanisms get only one: end-to-end keys are
consistent but capped by batch size; a memory bank is large but its keys come from
encoder states scattered across an entire epoch.

## Key idea

Two pieces together give large *and* consistent:

1. **Dictionary as a queue.** Maintain a FIFO queue of `K` encoded keys; each step
   enqueue the current minibatch's keys and dequeue the oldest. `K` is decoupled
   from the batch size (e.g. batch 256, `K = 65536`). Evicting oldest-first also
   discards the least-consistent keys.
2. **Momentum key encoder.** The keys can't receive gradients (they span many past
   steps). Update the key encoder as a slow exponential moving average of the query
   encoder, so all `K` queued keys come from nearly the same encoder:
   `θ_k ← m θ_k + (1−m) θ_q`, with `m` close to 1 (default 0.999) and only `θ_q`
   trained by back-propagation. `m = 0` degenerates to copying every step; larger
   `m` makes the key encoder track more slowly.

## Final objective and algorithm

InfoNCE realized as a `(1+K)`-way softmax classifier that must pick the positive:

  `L_q = −log( exp(q·k_+ / τ) / Σ_{i=0}^{K} exp(q·k_i / τ) )`

Per step: encode the query view `q = normalize(f_q(x_q))`; momentum-update `f_k`;
encode the key view `k = normalize(f_k(x_k))` (no gradient); form logits
`[q·k_+ , q·queue]/τ`; cross-entropy with target index 0; SGD on `f_q`; enqueue `k`,
dequeue the oldest. Encoder is a standard ResNet with a 128-D L2-normalized output;
`τ = 0.07`. Positives are two random augmentations of one image.

**Shuffling BN.** BN couples samples in a batch, letting the model cheat by reading
the positive's identity off shared batch statistics (pretext accuracy spikes,
validation collapses). Fix: shuffle the key batch across GPUs before encoding (and
unshuffle after), so a query and its positive key fall in different per-GPU BN
sub-batches.

Defaults: `dim=128, K=65536, m=0.999, T=0.07`; SGD lr 0.03 (batch 256, 8 GPUs),
momentum 0.9, weight decay 1e-4.

## Code

```python
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """Query encoder + momentum key encoder + a queue dictionary."""
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07):
        super().__init__()
        self.K, self.m, self.T = K, m, T

        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.copy_(p_q.data)      # key encoder = copy of query encoder
            p_k.requires_grad = False     # key encoder is not updated by gradient

        self.register_buffer("queue", nn.functional.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // x.shape[0]
        idx_shuffle = torch.randperm(batch_size_all, device=x.device)
        torch.distributed.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)
        idx_this = idx_shuffle.view(num_gpus, -1)[torch.distributed.get_rank()]
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        x_gather = concat_all_gather(x)
        num_gpus = x_gather.shape[0] // x.shape[0]
        idx_this = idx_unshuffle.view(num_gpus, -1)[torch.distributed.get_rank()]
        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        q = nn.functional.normalize(self.encoder_q(im_q), dim=1)        # N x C

        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = nn.functional.normalize(self.encoder_k(im_k), dim=1)    # N x C
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)               # N x 1
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # N x K
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T                   # N x (1+K)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)  # positive = 0

        self._dequeue_and_enqueue(k)
        return logits, labels


@torch.no_grad()
def concat_all_gather(tensor):
    gathered = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, tensor, async_op=False)
    return torch.cat(gathered, dim=0)
```

```python
import torchvision.transforms as T
import torchvision.models as models


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


aug = T.Compose([
    T.RandomResizedCrop(224, scale=(0.2, 1.)),
    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transform = TwoCropsTransform(aug)

model = MoCo(models.resnet50).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)

for view_q, view_k in loader:                 # each image -> two augmented views
    logits, labels = model(im_q=view_q.cuda(), im_k=view_k.cuda())
    loss = criterion(logits, labels)          # InfoNCE
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Each query view and key view come from `train_transform` applied to the same
image. After pretraining, the query encoder's backbone is the transferable
representation: linear-probe on frozen features for ImageNet classification, or
fine-tune end-to-end for detection/segmentation.
