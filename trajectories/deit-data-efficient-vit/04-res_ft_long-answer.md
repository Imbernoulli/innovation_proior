**Problem.** The two-token, hard-label, fused-readout design from rung 3 is locked in — architecture,
loss, and teacher are all settled. The one lever never itself tested is schedule length: 300 epochs was
picked at the outset on general grounds, before there was a dual-token architecture or a distillation
objective to converge, and it has not been re-examined since. Two structural reasons suggest it may be an
undertrained point for this specific configuration: repeated augmentation means a nominal epoch only
walks through roughly a third of the dataset's distinct images, and a shared trunk now serves two
partially different objectives (true label via the class token, teacher pseudo-label via the distillation
token) rather than one. A matching reason it may be safe to extend: the augmentation stack (RandAugment,
Mixup, CutMix, erasing, repeated augmentation) keeps synthesizing novel views rather than repeating exact
inputs, which should blunt the usual overfitting risk of training longer.

**Proposal.** Freeze every choice settled through rung 3 — two-token architecture, fused softmax readout
as primary (with class-only and distillation-only still reported), hard-label loss against the fixed
RegNetY-16GF teacher, the full rung-1 augmentation/regularization bundle — and extend only the schedule,
300 → 1000 epochs, rescaling the cosine decay horizon so the learning rate still anneals to zero at the
new endpoint. Run at 224² for all three sizes (Tiny/Small/Base); separately carry the resulting Base-size checkpoint
through the already-established 384² fine-tune (bicubic-interpolated positional embeddings, teacher
retained) to see whether the epoch lever and the resolution lever still compound.

```python
def cosine_schedule(base_lr, total_epochs, warmup_epochs=5):
    # only change from rung 1: total_epochs 300 -> 1000; horizon rescaled so decay still reaches
    # zero at the new endpoint rather than being truncated partway through
    def lr_at(epoch):
        if epoch < warmup_epochs:
            return base_lr * epoch / warmup_epochs
        import math
        t = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * base_lr * (1 + math.cos(math.pi * t))
    return lr_at


training_config_rung4 = dict(
    # unchanged from rung 3: architecture (DistilledViT, two tokens), loss (training_loss, hard-label,
    # fixed RegNetY-16GF teacher), readout (fused primary; class-only and distillation-only also logged),
    # augmentation/regularization bundle (RandAugment, Mixup, CutMix, Random Erasing, Stochastic Depth,
    # Repeated Augmentation x3, Label Smoothing 0.1, no dropout), optimizer (AdamW, wd 0.05,
    # lr = 0.0005*batch/512), batch size 1024.
    epochs=1000,                          # only change: 300 -> 1000
    lr_schedule=cosine_schedule(base_lr=0.0005, total_epochs=1000, warmup_epochs=5),
)

# after the 1000-epoch 224^2 run at the Base size completes, fine-tune the resulting checkpoint at 384^2
# exactly as in rung 1/3: bicubic-interpolated positional embeddings, same augmentation, teacher
# retained (re-evaluated at the fine-tuning resolution), ~25 epochs of fine-tuning.
finetune_config = dict(resolution=384, interp="bicubic", keep_augmentation=True, keep_teacher=True)
```

**Test.** Report 224² top-1 for all three sizes under the 1000-epoch schedule against their rung-3
300-epoch numbers, and the 384²-fine-tuned Base-size number against its own rung-3 counterpart, to see
whether the extra training epochs are a meaningfully better use of the same architecture and recipe, a
wasteful multiplication of compute for marginal return, or something size-dependent in between.
