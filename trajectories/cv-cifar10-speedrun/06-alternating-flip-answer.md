**Problem (from step 5).** At 10.8 epochs / 4.2 s, the only untouched lever is the augmentation. Standard
horizontal-flip flips each image independently with p=0.5 per epoch. Flipping has only two states per image
(2N unique inputs total), but independent coin flips show the *same* view two epochs running for half the
images — so a consecutive epoch-pair sees only ~1.5N unique views instead of 2N. A quarter of the
per-pair augmentation budget is wasted re-showing views the network just saw — the same redundancy that
random reshuffling already eliminates for example *ordering*.

**Key idea.** *Alternating flip* — a derandomized variant of horizontal flipping. Randomly flip 50% of
images in epoch 1; thereafter, on even epochs flip exactly those *not* flipped in epoch 1, on odd epochs
flip exactly those that *were*. Each image strictly alternates original↔mirror, so every consecutive
epoch-pair contains all 2N unique views with zero redundant repeats. Decide flips by a deterministic
per-image hash XOR-ed with epoch parity, so no per-image flip state is stored (no extra memory).

**Why it works.** It maximizes unique inputs seen per unit of training time — the same principle behind
random reshuffling, applied to flipping. Each epoch now delivers ~25% more fresh views per epoch-pair, which
lets the schedule be shortened (to ≈9.9 epochs, the final airbench94 length) at the same 94% accuracy. The
removed randomness was producing redundancy, not useful diversity, so derandomizing only helps. This is the
paper's one novel contribution, accounting for the final ~10% of the speedup over prior work.

**Change / code.** The clean statement of the method, deciding flips by hash + epoch parity:

```python
import hashlib
def hash_fn(n, seed=42):
    k = n * seed
    return int(hashlib.md5(bytes(str(k), 'utf-8')).hexdigest()[-8:], 16)

def alternating_flip(inputs, indices, epoch):
    # Applies alternating flipping to a batch of images
    hashed_indices = torch.tensor([hash_fn(i) for i in indices.tolist()])
    flip_mask = ((hashed_indices + epoch) % 2 == 0).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)
```

In the actual `CifarLoader` this is realized without per-batch hashing by pre-flipping a copy of the images
once and swapping every other epoch:

```python
# pre-flip the whole set once (epoch 0), then alternate every other epoch
if self.aug.get('flip', False):
    images = self.proc_images['flip'] = batch_flip_lr(images)   # random 50% flip, epoch 0
...
if self.aug.get('flip', False):
    if self.epoch % 2 == 1:
        images = images.flip(-1)        # flip *all* together every other epoch -> per-image alternation
```

This completes the **airbench94** training: 94.01% mean accuracy in 9.9 epochs / 3.83 A100-seconds.
(A non-algorithmic `torch.compile` step further reduces this to 3.29 A100-seconds, §3.7, without changing
the math.)
