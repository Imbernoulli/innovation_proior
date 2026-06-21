Multi-crop TTA brought me to 10.8 epochs and 4.2 A100-seconds. I have now squeezed initialization, optimization, and inference; the last untouched lever is the *data* the optimizer sees, specifically the augmentation. Training uses horizontal-flip augmentation — flip each image left-right with 50% probability every epoch — which I had been treating as obviously correct. But in a roughly 10-epoch budget where every epoch is precious, the *statistics* of how that augmentation samples views can waste some of those epochs. Horizontal flipping has exactly two states per image, original and mirrored, so the training distribution holds at most $2N$ unique inputs for $N$ training images, and the augmentation's job over a run is to show the network as many of those unique inputs as it can — diversity is what augmentation buys. Now ask how many *unique* views the network actually sees across two consecutive epochs under standard per-epoch random flipping: each image is flipped independently with $p=0.5$ in each epoch, so for a given image the two epochs show the *same* view (both flipped or both not) with probability $0.5$. On average *half* the images are shown the identical view two epochs running — a redundant repeat carrying no new information. Across a pair of epochs the network sees only about $1.5N$ unique views instead of close to $2N$; a quarter of the per-pair augmentation budget is spent re-showing views the network just saw. This is the same redundancy logic behind random reshuffling (sampling without replacement so all $N$ images appear each epoch, versus with-replacement which shows only ~63% of unique images per epoch), which is already in the recipe for exactly this reason — and random *flipping* violates that same principle.

I propose **alternating flip**, a derandomized variant of horizontal flipping that shows the network all $2N$ unique views across every consecutive epoch-pair with no repeats. The construction: in the first epoch, flip a random 50% of images as usual; then on the *even* later epochs $\{2,4,6,\dots\}$ flip exactly those images that were *not* flipped in epoch 1, and on the *odd* later epochs $\{3,5,7,\dots\}$ flip exactly those that *were*. The effect is that each image strictly alternates between its two states from one epoch to the next — original, mirror, original, mirror — so any two consecutive epochs contain both views of every image, the full $2N$, with zero redundant repeats. It keeps the same two views per image as standard flipping; it just schedules them to eliminate the wasteful coincidental repeats. The implementation has to decide flips per-image consistently across epochs (so I can alternate) without costing extra memory, and I get both by deciding each image's epoch-1 flip with a deterministic hash of its index and then XOR-ing in the epoch parity. The parity $(\text{hash}(i) + \text{epoch}) \bmod 2$ is what makes each image alternate: hold $i$ fixed and step the epoch and the mask flips every epoch; hold the epoch fixed and the per-image hash randomizes which half is flipped. No stored per-image flip state is needed — the pseudorandom function recovers the decision on the fly. In the actual loader I realize this even more cheaply, by pre-flipping a copy of the images once at epoch 0 and swapping the whole set on every other epoch (so the global swap induces the per-image alternation), but the hash form below is the clean statement of the method.

Why this only helps: the inefficiency is that random flip shows ~$1.5N$ unique views per epoch-pair where alternating shows ~$2N$, roughly a 25% boost in fresh views per unit of training. That does not translate one-to-one into a 25% epoch saving, but it is the kind of thing that lets me *shorten the schedule* — if each epoch delivers more new information, the same accuracy arrives in fewer epochs. The one worry is that alternating is *less* random than independent flipping, and if the network had benefited from the extra entropy of truly independent flips, derandomizing could hurt diversity; but the whole argument is that the extra randomness was producing *redundancy*, not useful diversity, so removing it only helps — and the same reasoning is what makes random reshuffling beneficial. This is the rung where I finalize the schedule to its short final length of about 9.9 epochs and call the result the finished airbench94 training, holding 94.01% mean accuracy. This is the one genuinely novel contribution of the ladder, accounting for the final ~10% of the speedup over prior work.

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

In the actual `CifarLoader` this is realized without per-batch hashing by pre-flipping a copy of the images once and swapping every other epoch:

```python
# pre-flip the whole set once (epoch 0), then alternate every other epoch
if self.aug.get('flip', False):
    images = self.proc_images['flip'] = batch_flip_lr(images)   # random 50% flip, epoch 0
...
if self.aug.get('flip', False):
    if self.epoch % 2 == 1:
        images = images.flip(-1)        # flip *all* together every other epoch -> per-image alternation
```
