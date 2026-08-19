The fused two-token readout beat every alternative I tried: the single class-token classifier, the single
distillation-token classifier, and the rung-2 baseline where one token carried both objectives at once.
That settles the architecture and the loss — a dedicated distillation token with a fused readout, trained
against a fixed convnet teacher's hard pseudo-label — and I am not revisiting either this rung. What is
still sitting untouched since the very first rung is the training schedule itself: 300 epochs, chosen at
the outset on general grounds before there was any dual-token architecture or distillation objective to
converge, and never itself re-examined since. I want to check whether that number is a genuine floor or
just a budget I picked once and never revisited.

I have a specific reason to suspect 300 epochs might undertrain exactly this configuration, beyond the
generic "more training helps" intuition. Repeated augmentation, part of the recipe since rung 1, draws
three independently augmented views of the same source image into a batch rather than three distinct
images, so a nominal epoch only walks through roughly a third of the dataset's distinct images before the
schedule calls it complete. And the network I am now training has two classifiers reading two separate
tokens off one shared attention trunk, each pulling it toward a partially different target — the true
label through the class token, the teacher's frequently-different hard decision through the distillation
token. A trunk reconciling two related-but-not-identical objectives has no fewer constraints to satisfy
than one serving a single objective, and there is no guarantee the epoch budget that sufficed for the
single-objective case still suffices once a second, only partially overlapping objective sits on top of
it.

There is a matching reason the extra epochs should be safe to spend rather than actively harmful. The
usual risk of training longer is overfitting: once a model has seen a fixed set of examples enough times,
more epochs mostly memorize rather than generalize. But the recipe already stacks RandAugment, Mixup,
CutMix, random erasing, and repeated augmentation, so the network is essentially never shown the same
exact input twice — every pass synthesizes a fresh augmented view rather than repeating an identical one.
That is a different regime from running extra epochs over literally the same unaugmented images, and if
the augmentation stack is doing the job it was built to do — substituting for data this architecture
doesn't have — it should keep supplying comparatively novel signal well past the point an unaugmented run
would have started memorizing. I state this as a directional argument, not a guarantee: it is entirely
possible the network has already extracted what it can from this particular augmentation distribution by
epoch 300 and further training just spins in place.

The one thing keeping this from being a free lever is cost: a 300-epoch run at the Base size already takes on the
order of two full days on a single 8-GPU node, so more than tripling the epoch count triples that
compute and wall-clock bill for whatever it buys. That tradeoff is exactly why I want to settle this by
measurement rather than by picking whichever of the two directional arguments above I find more
persuasive.

I propose freezing every choice settled through the previous rung — two-token architecture, fused
softmax readout as the primary metric (with the two single-token readouts still logged, since a schedule
change could in principle shift the balance between them even without changing which one wins outright),
hard-label loss against the fixed RegNetY-16GF teacher, the full augmentation and regularization bundle —
and extending only the schedule length, 300 epochs to 1000, with the cosine decay's horizon rescaled so
the learning rate still anneals to zero at the new endpoint rather than being truncated partway through
its planned decay. I run this at 224² for all three sizes, and separately carry the resulting Base-size
checkpoint through the already-established 384² fine-tune — bicubic-interpolated positional embeddings,
teacher retained during fine-tuning — to see whether the epoch lever and the resolution lever still
compound the way they did independently, or whether one of them has already captured most of the
available headroom by the time the other is applied. Whether 1000 epochs is a meaningfully better use of
the same architecture and recipe, a wasteful multiplication of compute for marginal return, or something
size-dependent in between, is exactly what running it will decide — and it is the last lever available
before treating whatever this configuration reaches as the answer to how far a convolution-free,
ImageNet-1k-only Transformer can go.

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
    # unchanged from rung 3: architecture (DistilledViT, two tokens), loss (hard-label, fixed
    # RegNetY-16GF teacher), readout (fused primary; class-only and distillation-only also logged),
    # augmentation/regularization bundle, optimizer (AdamW, wd 0.05, lr = 0.0005*batch/512), batch 1024.
    epochs=1000,                          # only change: 300 -> 1000
    lr_schedule=cosine_schedule(base_lr=0.0005, total_epochs=1000, warmup_epochs=5),
)

# after the 1000-epoch 224^2 run at the Base size completes, fine-tune the resulting checkpoint at 384^2
# exactly as before: bicubic-interpolated positional embeddings, same augmentation, teacher retained
# (re-evaluated at the fine-tuning resolution), ~25 epochs of fine-tuning.
finetune_config = dict(resolution=384, interp="bicubic", keep_augmentation=True, keep_teacher=True)
```
