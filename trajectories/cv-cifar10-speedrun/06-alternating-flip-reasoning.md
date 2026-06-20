10.8 epochs, 4.2 seconds. I have squeezed initialization, optimization, and inference. The last untouched
lever is the *data* the optimizer sees, and specifically the augmentation. The training uses horizontal-
flip augmentation: standard practice, flip each image left-right with 50% probability every epoch. I have
been treating "flip with p=0.5 per epoch" as obviously correct. Let me question it, because in a 10-epoch
budget where every epoch is precious, the *statistics* of how the augmentation samples views might be
wasting some of those precious epochs.

Here is the thing I want to think through carefully. Horizontal flipping has exactly two states per image:
original and mirrored. So if flipping is the augmentation, the training distribution has at most 2N unique
inputs for N training images. The job of the augmentation, over a run, is to show the network as many of
those unique inputs as it can — diversity is what augmentation buys. Now ask: with standard per-epoch
random flipping, how many *unique* views does the network actually see across two consecutive epochs?
Each image is flipped independently with p=0.5 in epoch 1 and again with p=0.5 in epoch 2. For a given
image, the two epochs show the *same* view (both flipped or both not) with probability 0.5. So on average
*half* the images are shown the identical view two epochs running — a redundant repeat that carries no new
information. Across a pair of epochs, instead of seeing close to all 2N unique views, the network on
average sees only about 1.5N. A quarter of the per-pair augmentation budget is spent re-showing views the
network just saw.

This is the same redundancy logic I keep coming back to: a fast run can't afford to spend its few epochs
re-presenting things it has already seen. Random reshuffling (sampling without replacement each epoch so
all N images appear, vs. with-replacement which on average shows only ~63% of unique images per epoch) is
already in the recipe for exactly this reason — maximize unique inputs per unit of training time. Random
*flipping* violates that same principle and I hadn't noticed: independent coin flips each epoch
needlessly repeat half the flip-views.

So derandomize it. I want a flipping scheme where every consecutive pair of epochs shows the network
*all* 2N unique views — every image once original and once mirrored across each pair, with no repeats.
The construction: in the first epoch, flip a random 50% of images as usual. Then on the *even* later
epochs {2, 4, 6, …} flip exactly those images that were *not* flipped in epoch 1, and on the *odd* later
epochs {3, 5, 7, …} flip exactly those that *were* flipped in epoch 1. The effect is that each image
strictly alternates between its two states from one epoch to the next — original, mirror, original,
mirror — so any two consecutive epochs contain both views of every image, the full 2N, with zero
redundant repeats. I call it *alternating flip*. It is a derandomized variant of standard flipping: same
two views per image, but scheduled to eliminate the wasteful coincidental repeats.

The implementation needs to be careful about two things: it must decide flips per-image in a way that's
consistent across epochs (so I can alternate), and it shouldn't cost extra memory. I can get both by
deciding each image's epoch-1 flip with a deterministic hash of its index, then XOR-ing in the epoch
parity:

```python
def hash_fn(n, seed=42):
    k = n * seed
    return int(hashlib.md5(bytes(str(k), 'utf-8')).hexdigest()[-8:], 16)

def alternating_flip(inputs, indices, epoch):
    # Applies alternating flipping to a batch of images
    hashed_indices = torch.tensor([hash_fn(i) for i in indices.tolist()])
    flip_mask = ((hashed_indices + epoch) % 2 == 0).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)
```

The `(hash(i) + epoch) % 2` parity is what makes each image alternate: hold `i` fixed and step `epoch`,
and the mask flips every epoch. Hold `epoch` fixed and the per-image hash randomizes which half is
flipped this epoch. No stored per-image flip state, no extra memory — the pseudorandom function recovers
the decision on the fly. (In the actual loader I realize this even more cheaply with a pre-flipped copy
of the images and an `epoch % 2` swap, but the hash form is the clean statement of the method.)

Now I should be precise about the size of the prize, because I can almost compute it. The inefficiency is
that random flip shows ~1.5N unique views per epoch-pair where alternating shows ~2N — roughly a 25%
boost in fresh views per unit of training. That doesn't translate one-to-one into a 25% epoch saving, but
it is the kind of thing that lets me *shorten the schedule*: if each epoch now delivers more new
information, the same accuracy arrives in fewer epochs. So I expect to be able to trim the training
duration — and this is the rung where I finalize the schedule to its short final length, around 9.9
epochs, and call the result the finished airbench94 training. The accuracy should hold at the 94% bar
(Table 1 will tell me whether alternating flip on its own moves the mean, and whether it composes with
random reshuffling). The risk is subtle: alternating is *less* random than independent flipping, and if
the network somehow benefited from the extra entropy of truly independent flips, derandomizing could hurt
diversity rather than help. But the whole argument is that the "extra randomness" was producing
*redundancy*, not useful diversity, so removing it should only help. The change is the alternating-flip
augmentation; code in the answer.
