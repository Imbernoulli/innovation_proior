The fused two-token readout beat every alternative on the table: the single class-token classifier, the
single distillation-token classifier, and the rung-2 baseline that only had one token doing both jobs at
once. That settles the architecture question for now — a dedicated distillation token with a fused
readout is the design I am carrying forward — and it settles the loss question inherited from rung 2:
hard-label pseudo-labels from a fixed RegNetY-16GF teacher. Nothing about the network, the loss, or the
teacher is up for revision this rung. What is still sitting untouched from the very first rung is the
training schedule itself: 300 epochs, chosen at the outset as "enough epochs to let a heavily augmented,
weakly data-primed model converge," on general grounds, before there was any dual-token architecture or
distillation objective to converge at all. That number has never itself been tested against a longer
alternative under the recipe as it now stands, and I want to check whether it is actually a floor or
whether it is a budget I picked once and never revisited.

There is a specific reason to suspect 300 epochs might be an undertrained point for exactly this
configuration, distinct from the general "more training helps" intuition. Repeated augmentation, part of
the recipe since rung 1, draws three independently augmented views of the same source image into a batch
rather than three distinct source images; the practical effect is that any given nominal "epoch" only
walks through roughly a third of the dataset's distinct images before the schedule calls it complete, so
the optimizer sees comparatively less distinct raw content per epoch than a non-repeated-augmentation
schedule of the same nominal length would. On top of that, the architecture now being trained has two
classifiers reading two separate tokens off the same shared attention stack, each pulling the trunk toward
a partially different target — the true label through the class token, the teacher's frequently-different
hard decision through the distillation token. A shared trunk serving two related-but-not-identical
objectives has, at minimum, no fewer constraints to reconcile than a trunk serving one, and there is no
guarantee that whatever epoch budget was "enough" for the single-objective case is still enough once a
second, only partially overlapping objective is added on top of it. Both of these are reasons the model
in front of me now might have more room left to improve with additional optimization steps than the
rung-1 recipe did on its own — not a certainty, since I have not actually measured either effect in
isolation, but a directional argument worth taking seriously before treating 300 epochs as fixed.

There is a matching reason the extra epochs might be safe to spend rather than wasteful or actively
harmful. The usual risk of training longer is overfitting: once a model has seen a fixed, finite set of
training examples enough times, additional epochs mostly memorize rather than generalize, and validation
accuracy plateaus or degrades. But the recipe already in place — RandAugment, Mixup, CutMix, random
erasing, and repeated augmentation itself — means the network is essentially never shown the same exact
input twice; every pass over the data, however many epochs it takes, synthesizes a fresh augmented view
rather than repeating an identical one. A model trained under that much data diversification is a
different case from an unaugmented model being run for extra epochs over literally the same 1.28 million
images: the augmentation stack is specifically the mechanism this whole exploration built to substitute
for missing data, and if it does what it is meant to do, it should keep supplying comparatively novel
training signal well past the point where an unaugmented run would have started memorizing. That is a
reason to expect the marginal epoch to still buy something at 1000 that it might not buy for a model
without this augmentation stack — again, not something I can assert without checking, since it is possible
the network has already extracted everything useful from this augmentation distribution by epoch 300 and
further training just spins in place.

The tradeoff that keeps this from being a free lever is cost, not risk of overfitting: a 300-epoch run at
the Base size already takes on the order of two full days on a single 8-GPU node, so more than tripling the epoch
count triples that compute and wall-clock bill for whatever gain it buys. That is exactly the kind of
decision that should not be made on the strength of the directional argument above alone — the two
competing possibilities (real headroom under this recipe's augmentation, versus a plateau already reached
by 300 epochs) have to be told apart by actually running the longer schedule and reading off the number,
not by reasoning about which is more plausible in the abstract.

Concretely: keep every other choice frozen exactly as it stands after rung 3 — two-token architecture,
fused readout as the primary metric (with the individual-token readouts still worth checking, since a
schedule change could in principle shift the balance between them even if it doesn't shift which one wins
outright), hard-label loss against the same fixed RegNetY-16GF teacher, the full rung-1 augmentation and
regularization bundle unchanged — and extend only the schedule length, from 300 to 1000 epochs, with the
cosine decay's horizon rescaled so the learning rate still anneals to zero at the new endpoint rather than
being truncated partway through its planned decay. Run this at 224² for all three sizes, and separately
carry the resulting Base-size checkpoint through the same 384² fine-tune (bicubic-interpolated positional
embeddings, teacher retained during fine-tuning) already established as the resolution lever, to see
whether the two levers — more pretraining epochs, and the resolution bump — still compound the way they
did independently at 300 epochs, or whether one of them has already captured most of the available
headroom by the time the other is applied. Whether 1000 epochs is a meaningfully better use of the same
architecture and recipe, a wasteful multiplication of compute for a marginal return, or something in
between at different model sizes, is exactly what running it will tell me — and it is the last lever I
have left to pull before treating whatever this configuration reaches as the answer to the original
question of how far a convolution-free, ImageNet-1k-only Transformer can go.
