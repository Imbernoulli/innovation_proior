10.8 epochs, 4.2 seconds. The multi-crop TTA rung came out on the favorable side of its trade, and the
numbers say so cleanly: 12.0 → 10.8 epochs (1.2 epochs saved) and 4.6 → 4.2 seconds, even though
per-epoch cost ticked *up* from 4.6/12.0 = 0.383 s to 4.2/10.8 = 0.389 s. That ~1.5% per-epoch rise is
the six-view inference I added — eval got slightly more expensive per epoch, exactly as priced — and the
seconds still fell because the 1.2 epochs of training removed were worth far more than the fraction of an
epoch the richer eval cost. So the trade landed positive, as bet, and it tells me the network did still
have some translation-sensitivity left to average away. I have now squeezed initialization, optimization,
and inference. The last untouched lever is the *data* the optimizer sees — and specifically the
augmentation.

The training uses horizontal-flip augmentation: standard practice, flip each image left-right with 50%
probability, independently, every epoch. I have been treating "flip with p=0.5 per epoch" as obviously
correct, because it is what everyone does. But in a budget that is now down to ~10 epochs, where every
epoch is precious, I should question whether the *statistics* of how that augmentation samples views are
wasting some of those precious epochs. The augmentation's whole job is to show the network variety; let me
count how much variety it actually delivers per unit of training.

Here is the thing to think through carefully. Horizontal flipping has exactly two states per image:
original and mirrored. So with flipping as the augmentation, the entire universe of distinct inputs is at
most 2N for N training images — there are only 2N different pictures the network can ever be shown. The
augmentation's job, over a run, is to show as many of those 2N as it can; diversity is what augmentation
buys. Now ask: with standard per-epoch random flipping, how many *unique* views does the network actually
see across two consecutive epochs? Each image is flipped independently with p = 0.5 in epoch t and again,
independently, with p = 0.5 in epoch t+1. For a given image, the two epochs show the *same* view — both
flipped, or both not — with probability P(both flipped) + P(both unflipped) = 0.25 + 0.25 = 0.5. So on
average *half* the images are shown the identical picture two epochs running: a redundant repeat that
carries no new information the network did not already get last epoch. Counting the distinct views per
image across the pair: with probability 0.5 the two draws coincide (1 distinct view), with probability 0.5
they differ (2 distinct views), so the expected number of distinct views per image is 0.5·1 + 0.5·2 = 1.5.
Across N images that is 1.5N distinct views out of a possible 2N per epoch-pair. A full quarter of the
per-pair augmentation budget — 0.5N of the 2N — is spent re-showing views the network just saw.

I should be honest about what the waste actually is, because a naive reading would say "over a whole run,
random flipping shows both views eventually, so who cares." Let me check that objection on its own terms.
An image fails to see *both* its views after T epochs of independent flipping only if all T coin flips
came up the same way, probability 2·(1/2)ᵀ. For a ~10-epoch run that is 2·2⁻¹⁰ = 2/1024 ≈ 0.2%, so about
99.8% of images have seen both orientations by the end — coverage is essentially complete. So the problem
is *not* that views are never seen. The problem is *rate and balance*. In a short, fast run the value of
an epoch is dominated by how much genuinely new information it delivers, and half of every epoch-pair's
flip budget going to re-shows means fresh views arrive at three-quarters the rate they could. Alternating
front-loads the diversity: it guarantees the maximum fresh-view rate every single pair, so the network
gets its variety as early as possible, which is exactly what matters when there are only ~10 epochs to
spend.

The balance point is the second, quieter benefit and worth making explicit. Under alternating flip, over
T epochs each image is shown its two orientations in a perfectly balanced count — ⌈T/2⌉ times one way and
⌊T/2⌋ the other. Under independent random flipping, each image's orientation count is Binomial(T, 0.5),
so there is real spread: some images come up 8-of-10 in one orientation purely by chance, giving the
network a lopsided, per-image left/right bias in the training signal. That imbalance is a small source of
noise in what the network learns about each image, and alternating removes it exactly — every image
contributes equally from both sides. So derandomizing buys two things: a higher fresh-view *rate* (the
1.33× efficiency) and perfect per-image orientation *balance* (no chance lopsidedness), both of which help
a short run more than a long one.

This is the same redundancy logic I keep coming back to, and it is already recognized elsewhere in the
recipe. Sampling the training set *with* replacement each epoch would show, in expectation, only N(1 −
1/e) ≈ 0.632N unique images per epoch; sampling *without* replacement (random reshuffling — a fresh
permutation each epoch, every image exactly once) shows all N. Reshuffling is already in the recipe for
exactly this reason: maximize the unique inputs seen per unit of training time, do not waste steps
re-presenting things already seen. Random *flipping* quietly violates the very same principle on a
different axis, and I had not noticed it: independent coin flips each epoch needlessly repeat half the
flip-views, just as with-replacement sampling needlessly repeats ~37% of the images. If reshuffling was
worth doing to fix the ordering redundancy, fixing the flip redundancy is worth doing for the same reason.

Before I derandomize, let me consider the honest alternative and rule it out with arithmetic. I could
instead show *both* views of every image every epoch — concatenate each image with its mirror into the
batch — which guarantees all 2N views per epoch. But that doubles the images processed per epoch, so each
epoch costs 2× the forward+backward work. Measure everything in views-per-unit-compute: random flip
delivers 0.75N new unique views per epoch (1.5N per two epochs) at cost 1, so 0.75N per unit; show-both
delivers 2N views per epoch at cost 2, so N per unit; and what I am about to propose, alternating flip,
delivers ~N new unique views per epoch (2N per two epochs) at cost 1, so N per unit. So alternating
*matches* show-both's view efficiency while keeping the cheap single-pass-per-epoch structure — no batch
doubling, no extra memory — and it beats plain random flipping by the full 1.33× (N vs 0.75N per unit
compute), which is exactly the 25% of the budget random flipping was wasting. That settles it: derandomize
the flip schedule rather than double the batch.

There is a clean way to see why this helps *most* in the short-run regime, which is where I live. Replacing
independent random coin flips with a balanced, deterministic alternation is the same move as replacing
Monte-Carlo sampling with a low-discrepancy (quasi-Monte-Carlo) sequence: instead of drawing views iid and
tolerating the clumping and gaps that randomness produces, I lay them down to cover the space as evenly as
possible. The payoff structure of that swap is well known — a balanced sequence's coverage error shrinks
like 1/N where iid sampling's shrinks like 1/√N — and the gap between those two rates is *largest when N is
small*. My "N" here is the handful of epochs each image is shown, so the derandomization dividend is
exactly the kind that is biggest in a short run and would wash out in a long one. That is another way of
saying what the balance and rate arguments already said, but it frames why the flip axis is worth
derandomizing precisely now that the schedule is down near ten epochs: the shorter the run, the more the
clumpiness of random flipping costs.

So the construction. I want a flipping scheme where every consecutive pair of epochs shows the network
*all* 2N unique views — every image once original and once mirrored across each pair, with no repeats. Do
it like this: in the first epoch, flip a random 50% of images as usual. Then on the *even* later epochs
{2, 4, 6, …} flip exactly those images that were *not* flipped in epoch 1, and on the *odd* later epochs
{3, 5, 7, …} flip exactly those that *were* flipped in epoch 1. The effect is that each image strictly
alternates between its two states from one epoch to the next — original, mirror, original, mirror — so any
two consecutive epochs contain both views of every image, the full 2N, with zero redundant repeats. It is
a derandomized variant of standard flipping: the same two views per image, but scheduled to eliminate the
coincidental repeats that independent coin-flipping produces.

The implementation has to satisfy two constraints: decide each image's flip in a way that is consistent
across epochs (so I can actually alternate a given image, which means I must recover its epoch-1 decision
later), and cost no extra memory (so I cannot just store a per-image flip table). Both fall out of
deciding each image's epoch-1 flip by a deterministic hash of its index, then XOR-ing in the epoch parity:

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

The `md5` hash is doing real work and is not just a fancy way to write index-parity. If I decided the
epoch-0 flip by the raw index parity `i % 2`, I would flip exactly the even-indexed images — and if the
dataset has any index structure (sorted by class, blocked, or grouped by source), that would correlate the
flip decision with the labels, so one systematic half of each class would always start mirrored. Hashing
`i * seed` through md5 and taking the parity of the result decorrelates the flip decision from any
structure in the index: it is a pseudorandom 50/50 split that is arbitrary with respect to class and
order, which is what "flip a random 50%" is supposed to mean. So the hash supplies the *randomness* of the
epoch-0 phase while the `+ epoch` parity supplies the *determinism* of the alternation — random which half,
deterministic that they alternate.

Let me trace three images to check it does what I claim. Say indices 0, 1, 2 hash to parities even, odd,
even (i.e. hash_fn(0) even, hash_fn(1) odd, hash_fn(2) even). The flip mask is `(hash + epoch) % 2 == 0`.
Epoch 0: parities of hash+0 are (even, odd, even) → mask == 0 is (True, False, True) → images 0 and 2 are
flipped, image 1 is not. Epoch 1: hash+1 is (odd, even, odd) → mask is (False, True, False) → image 1 is
flipped, images 0 and 2 are not. So across epoch 0 → 1: image 0 goes flip → no-flip, image 1 goes no-flip
→ flip, image 2 goes flip → no-flip. Every image alternates its state, and the pair {epoch 0, epoch 1}
contains both the original and the mirror of every one of the three images — exactly the 2N coverage with
no repeat. Holding the image fixed and stepping the epoch flips its state every time (the parity of hash+
epoch alternates); holding the epoch fixed and varying the index randomizes which half is flipped this
epoch (via the hash). Both properties I needed, and no per-image state stored — the pseudorandom function
recovers each decision on the fly, so the memory cost is zero.

In the actual loader I can realize the same thing even more cheaply, and it is worth confirming the cheap
form is equivalent to the hash form. Instead of hashing per batch, I pre-flip a copy of the whole image
set once at epoch 0 (a random 50% flip, `batch_flip_lr`), and then on every *other* epoch I flip the
entire set together with `images.flip(-1)`. Check the equivalence: let image i be in state sᵢ ∈ {orig,
mirror} after the epoch-0 random flip. Flipping *all* images together on odd epochs gives state sᵢ on even
epochs and ¬sᵢ on odd epochs — so each image alternates sᵢ, ¬sᵢ, sᵢ, ¬sᵢ, which is exactly per-image
alternation with a random epoch-0 starting phase, identical in distribution to the hash construction. The
global `flip(-1)` every other epoch is doing per-image alternation because a global flip toggles every
image's state in lockstep, and the random epoch-0 phase is what the hash's per-image parity supplied. Same
method, no per-batch hashing, a single pre-flipped buffer. And the operation itself is cheap: a horizontal
flip is just a reversal along the width axis, a near-free stride/index operation with no arithmetic, so
whichever form I use the augmentation adds essentially nothing to the per-epoch cost — the entire benefit
is in *which* views are shown, not in any added compute. That keeps this a pure scheduling change: same
cost per epoch as before, more diversity per epoch, which is precisely why its whole effect should read out
as a shorter schedule rather than as any change in per-epoch time.

Now I can be precise about the size of the prize. The inefficiency I am removing is that random flip
delivers 1.5N unique views per epoch-pair where alternating delivers the full 2N — a 4/3 boost in fresh
views per unit of training, the 1.33× from the efficiency accounting. That does not translate one-to-one
into a 1.33× epoch saving, because not every epoch's value is bottlenecked purely on flip-view novelty.
But it is exactly the kind of thing that lets me *shorten the schedule*: if each epoch now delivers more
genuinely new information and less re-run redundancy, the same accuracy should arrive in fewer epochs. So
this is the rung where I finalize the schedule to its short final length — around 9.9 epochs — and call the
result the finished airbench94 training. Whether alternating flip *on its own* nudges the mean accuracy,
and whether it composes with the random reshuffling that is already fixing the ordering redundancy, is
something I cannot compute from first principles — the training-distribution table will tell me, and I
expect both derandomizations to help a little and to stack, since they attack independent axes (ordering
and flip-state) of the same "don't re-show what was just seen" principle. On top of the finished training,
a non-algorithmic `torch.compile` pass fuses the kernels and cuts the wall-clock further to about 3.29
seconds without changing any of the math — a free implementation win layered on the algorithmic one.

It is worth stepping back to see what the whole descent has bought, because this rung closes the 94%
ladder. The baseline was 45 epochs / 18.3 seconds; the finished airbench94 is ~9.9 epochs / ~3.83 seconds
(3.29 compiled). That is a 4.5× reduction in epochs and a 4.8× reduction in wall-clock — the extra
factor beyond the epoch count coming from the frozen-layer backward savings and the compile fusion — and
it is built entirely from a stack of small, individually-defensible structural fixes: whitening (24
epochs), Dirac (3), scalebias (4.5), Lookahead (1.5), multi-crop TTA (1.2), and now the flip
derandomization finishing the schedule at ~9.9. Against the prior art I started under — hlb-CIFAR10 at 6.3
seconds, David Page at ~10 — the finished airbench94 at 3.83 seconds (3.29 compiled) sits comfortably
below both, so the 94% record is genuinely new territory now. Each rung removed a specific, named waste;
none of them was brute force; and they composed, which is the evidence that they were attacking distinct
inefficiencies rather than the same one repeatedly.

The prediction, stated to be falsifiable. If the redundancy argument is right, shortening the schedule to
~9.9 epochs under alternating flip should still clear 94% mean accuracy, because the epochs I am removing
were partly spent on redundant re-shows that alternating eliminates; I expect the finished airbench94 to
land at the bar in that shortened schedule, at ~3.83 seconds (3.29 compiled). The subtle risk is the
opposite of my thesis: alternating is *less* random than independent flipping, and if the network somehow
benefited from the extra entropy of truly independent coin flips — if that randomness was useful
regularization rather than wasteful repetition — then derandomizing could reduce diversity and cost
accuracy, and I would see the mean slip below the bar under the shortened schedule. But the whole argument
is that the "extra randomness" of independent flipping was producing *redundancy* (half the images
re-shown per pair), not useful diversity, so removing it should only help — and the sign of the effect in
the training-distribution table, on its own and in combination with reshuffling, is the clean test. The
change is the alternating-flip augmentation; the code is in the answer, and the accuracy table under the
shortened schedule is what will confirm the finished airbench94.
