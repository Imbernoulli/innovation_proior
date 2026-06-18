The defense is the whole point, but it bolts onto a trained classifier, and with no defense that
classifier is the floor — so the thing to start from is just training the model on the private half of
the data with the ordinary objective and seeing how badly it leaks. The frozen loop hands me one
method, `compute_loss(logits, labels, epoch)`, and the default fill is the obvious one: standard
cross-entropy, `F.cross_entropy(logits, labels)`. No privacy machinery, no schedule tricks — minimize
`−log p(y)` against the one-hot target, exactly what every classifier does. That is the baseline the
three published regularizers all react to, so I want its number on the board before I touch anything.

Let me be precise about why this is the weakest possible setting, because the reason is structural, not
incidental. Cross-entropy on a one-hot target places its optimum at an *infinite* correct-vs-rest logit
gap: `p(y) → 1` only as `z_y − z_k → +∞` for every wrong class `k`. The per-logit gradient is `p − y`,
bounded in `[−1, 1]`, so each step is gentle, but the target it chases sits at infinity and never
vanishes until the loss is essentially zero. Nothing in the objective tells the optimizer to *stop*
being confident on a training example. So over 300 epochs with augmentation deliberately turned off,
the model is free to grind member losses down to ~0 and member confidence up to ~1. Members pile up at
the confidence ceiling; non-members, never optimized, sit lower and more scattered. Two separated
humps — and the fixed attack is exactly a threshold on that confidence. The training procedure is
manufacturing the precise signal the adversary reads.

Let me be exact about what the attacker reads, because the whole ladder is aimed at this one quantity
and I should not start vague about it. The frozen loop, after training, runs the same forward pass on
every member and every non-member example, takes the *max softmax probability* of each — the model's
confidence in its own top prediction — and reports the AUC of that scalar as a member/non-member
discriminator, computed by the Mann–Whitney U statistic so that 0.5 is a coin flip and 1.0 is perfect
separation. Max-confidence is a monotone stand-in for the per-sample loss: a member the model has
memorized has near-zero loss and near-1 confidence; a non-member the model never optimized has higher
loss and lower, more scattered confidence. So the AUC the harness reports is, operationally, "how well
does a threshold on confidence tell members from non-members." Anything I do in `compute_loss` only
matters insofar as it changes the *two confidence distributions* — the member one and the non-member
one — and specifically how separable they are. That framing is the thing I want fixed in my head before
I touch any defense: relocating a distribution is not the same as making two distributions overlap, and
the attack only cares about overlap.

The composite the leaderboard ranks on, `privacy_score = test_acc − max(mia_auc − 0.5, 0)`, is built to
punish exactly the cheap escape. The `max(·, 0)` means an attack AUC at or below 0.5 costs nothing —
the model is rewarded purely for its accuracy — while every point of AUC above 0.5 is subtracted
one-for-one from accuracy. So a defense is only worth its accuracy cost if it actually drags the attack
toward the coin-flip floor; a defense that halves the leakage but also halves accuracy is a wash, and a
defense that destroys accuracy to win privacy scores near zero. For the no-defense floor, both terms
work against me wherever the model overfits: the overfit *is* the high AUC, and the datasets where the
model overfits hardest are also, on these benchmarks, the ones where test accuracy is lowest. The two
penalties compound.

That tells me what the three benchmarks should do, and they should split on one axis: how much the
model overfits, because the membership signal *is* the overfit. On **vgg16bn-cifar100** I expect the
worst leakage by far. A hundred classes, a high-capacity VGG, no augmentation — the model can memorize
the member half almost perfectly while test accuracy stays low (CIFAR-100 is hard and the test set is
untouched by training). Large train-test gap means a wide confidence gap between members and
non-members, which means a high attack AUC. On **resnet20-cifar10** I expect strong but milder leakage:
ResNet-20 is small enough that it can't memorize CIFAR-10 perfectly, test accuracy is high (~0.8), so
the gap is real but narrower. On **mobilenetv2-fmnist** I expect the least leakage of the three —
FashionMNIST is easy, the model generalizes well, train and test confidence are both near the ceiling,
so there is little gap for a threshold to exploit even with plain training. The composite
`privacy_score = test_acc − max(mia_auc − 0.5, 0)` will therefore be dragged down hardest on
CIFAR-100, where both terms hurt at once — low accuracy *and* high attack AUC.

It is worth saying why the harness is rigged to *let* the model overfit, because that is a deliberate
design choice that shapes the whole task. Train-time augmentation is turned off, and the model trains
for a full 300 epochs with a step-decayed learning rate. Augmentation is the single most effective
implicit regularizer against memorization on these datasets — random crops and flips mean the model
essentially never sees the same input twice, so it cannot drive any individual member to zero loss. By
removing it, the loop guarantees there *is* a measurable membership signal to defend against; with
augmentation on, plain ERM already yields attack AUCs close to 0.5 on CIFAR-10 and there is nothing for
a defense to improve. So the setting is adversarial by construction: the loop manufactures the
worst-case overfit and asks the loss to claw the leakage back without the crutch of data augmentation,
without touching the optimizer, and without trading away the accuracy that the same overfit produces.
This is why the only lever I have is the per-minibatch scalar loss — the loop will not let me regularize
any other way.

I should also note what the loop deliberately *hands* me but ERM ignores. The `compute_loss` signature
includes `epoch`, the current 0-indexed training epoch, which a method could use to change its behavior
over the course of training — anneal a penalty, alternate phases, switch regimes after the learning-rate
decay. Plain cross-entropy has no use for it; it is the same objective every epoch. But the fact that
the contract exposes it tells me the harness anticipates losses that are *not* time-invariant, and I file
that away — the strongest defense on this ladder will be the one that actually uses `epoch`. The method
can also read `logits.size(1)` to learn the class count `K` and adapt per dataset, which again ERM does
not need (cross-entropy is identical at every `K`) but a per-dataset defense will.

There is nothing to derive here; ERM is the absence of a defense. What I am establishing is the
reference the ladder climbs away from: a model that fits its members so confidently that a single
threshold on confidence separates members from non-members, with the damage scaling with how badly each
benchmark overfits. Let me also reason about *why* the threshold attack is so hard to beat, because it sets the bar for
every rung above. The attack does not need to know the threshold in advance — the AUC integrates over
*all* thresholds, so it measures the intrinsic separability of the two confidence populations regardless
of where a real adversary would cut. That means I cannot defeat it by, say, lowering everyone's
confidence by a constant: shifting both distributions down by the same amount leaves their relative
ordering — and hence every threshold's true/false-positive rates — untouched, so the AUC does not move
at all. The only things that move the AUC are changes to the *shape and relative position* of the two
distributions: closing the gap between their means, or widening the member distribution so its tail
reaches into the non-member mass. A defense that merely rescales or uniformly translates confidence is
invisible to this metric. I suspect, even at the floor, that this is the trap the simplest defenses will
fall into — but ERM does not even attempt it, so it should sit at the leakage ceiling and let the
defenses prove whether they can do better than a uniform shift.

Putting the two terms of the composite together per benchmark sharpens the prediction. On CIFAR-100,
both the low test accuracy and the wide confidence gap pull the score down, so I expect the *smallest*
privacy_score of the three by a large margin — the benchmark where a real defense has the most to gain
and the most room to either help or, if it is too aggressive, destabilize. On CIFAR-10, accuracy is high
and the gap moderate, so the score should be middling but the attack clearly above the floor. On
FashionMNIST, accuracy is highest and the attack near the floor already, so the score should be highest
and there is the least for any defense to improve — the benchmark most likely to *cost* accuracy without
buying privacy. So even the floor's numbers should already foreshadow where the defenses earn their keep
(CIFAR-100, CIFAR-10) and where they risk being dead weight (FashionMNIST).

The diagnosis is already pointed at the next step — I have a *confidence* problem,
not a fitting problem. The model fits fine; it fits *too* confidently. The fix has to reach into the
output distribution and put a floor under member confidence so the member hump stops collapsing onto
the ceiling, and the first, simplest way to do that is the one the lineage reached for first: soften the
one-hot target itself. The distilled scaffold fill is in the answer.
