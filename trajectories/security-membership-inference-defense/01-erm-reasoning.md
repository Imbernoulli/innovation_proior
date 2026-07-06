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

Let me put numbers on how much room the loop gives that runaway, because the sheer count is the reason
memorization is inevitable rather than merely possible. The loop splits the full training set 50/50 by
index into members and non-members, so on CIFAR-10 and CIFAR-100 the member half is 25,000 images
(half of 50,000), and at batch size 128 one epoch is `⌈25000/128⌉ = 196` gradient steps. Over 300
epochs that is roughly 58,800 updates, all on the *same* 25,000 images with augmentation off — each
member is revisited about 300 times as literally the same pixels, never a jittered crop. On
FashionMNIST the member half is 30,000 (half of 60,000), `⌈30000/128⌉ = 235` steps per epoch, about
70,500 updates. That is an enormous number of passes over a fixed, un-augmented set; there is simply no
mechanism to stop the correct logit from creeping up on every one of those passes. And the learning-rate
schedule is arranged to *finish the job*: MultiStepLR decays ×0.1 at epochs 150 and 225, so training
runs in three phases — lr 0.1 for epochs 0–149, lr 0.01 for 150–224, lr 0.001 for 225–299. The first
phase does the bulk of the fitting; the two low-lr phases are exactly when a network stops exploring and
polishes the training examples it already nearly fits, driving their losses the last stretch toward zero.
The tail of the schedule is a memorization-sharpener, and ERM has nothing that fights it.

The gradient itself explains why this asymptotes at the ceiling rather than reversing. The per-logit
gradient `p − y` shrinks as `p → y`: once a member is nearly memorized, `‖p − y‖ → 0` and the update on
that sample becomes tiny, so ERM *naturally slows* on the examples it has already fit. But slowing is not
stopping and it is certainly not pushing back — the gradient only ever points toward higher confidence,
never away from it, so across ~58,800 gentle-but-monodirectional updates the member confidences asymptote
right at the ceiling and stay pinned there through the low-lr tail. This is the structural reason a real
defense cannot just make the pull *smaller*; it has to introduce an active force that pushes confidence
back *down*, because ERM's own dynamics will otherwise fill in any slack.

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

Let me make the rank-statistic fact concrete with a tiny hand trace, because it is the single property
that will decide which defenses can possibly work. The AUC of the confidence score equals
`P(conf of a random member > conf of a random non-member)`, i.e. the Mann–Whitney U divided by
`n_member · n_nonmember` — it counts, over all member/non-member pairs, the fraction where the member is
more confident. Take three members at confidence `{0.90, 0.95, 0.99}` and three non-members at
`{0.60, 0.70, 0.85}`. All nine pairs have member > non-member, so U = 9, AUC = 9/9 = 1.0: perfectly
separable. Now subtract a flat 0.30 from *everyone* — members `{0.60, 0.65, 0.69}`, non-members
`{0.30, 0.40, 0.55}` — and re-count: still all nine pairs have member > non-member, so AUC is *still*
1.0. A uniform downward shift moved the confidences a long way but did not flip a single pairwise
comparison, so the rank statistic is exactly unchanged. That is the trap, in arithmetic: any strictly
increasing map `f` applied to all confidences turns every `c_m > c_nm` into `f(c_m) > f(c_nm)`, leaving U
invariant. Only an operation that *reorders* member and non-member confidences — pulling some member
below some non-member — can move the AUC.

And this is exactly where `privacy_gap` and `mia_auc` come apart, which is why the harness reports both.
`privacy_gap` is the difference of means, `mean(member conf) − mean(non-member conf)`; in the example
above it is `0.9467 − 0.7167 = 0.23`, and after the flat shift it is `0.6467 − 0.4167 = 0.23`, unchanged,
because a difference of means is translation-invariant too. So both metrics ignore a pure shift — but they
diverge under *reshaping*. Widen the member cluster so one member drops to `0.55`, below a `0.60`
non-member: now one pair flips, U falls to 8, AUC to `8/9 ≈ 0.889`, while the *mean* member confidence may
barely move. `privacy_gap` tracks the first moment; `mia_auc` tracks full separability. A defense can
shrink the gap and leave the AUC flat (relocate the mean), or leave the gap roughly fixed and drop the AUC
(fatten the tails into overlap). The metric that ranks me is `mia_auc`, so overlap — reordering — is the
only thing that ultimately counts, and `privacy_gap` is a diagnostic that tells me whether I moved the
mean without necessarily moving separability.

There is one more reason this readout is clean, and it matters for trusting the numbers. Members and
non-members are the *same* underlying distribution, split only by index within one shuffled training set;
they are not a train-set-versus-a-different-test-set comparison. So there is no distribution shift to
confound the attack: any systematic confidence difference between the two groups can *only* be an artifact
of one group having been optimized and the other not. The AUC is therefore an honest thermometer of
memorization, and driving it toward 0.5 is precisely equivalent to erasing the train/non-train
distinction the loop created.

The composite the leaderboard ranks on, `privacy_score = test_acc − max(mia_auc − 0.5, 0)`, is built to
punish exactly the cheap escape. The `max(·, 0)` means an attack AUC at or below 0.5 costs nothing —
the model is rewarded purely for its accuracy — while every point of AUC above 0.5 is subtracted
one-for-one from accuracy. So a defense is only worth its accuracy cost if it actually drags the attack
toward the coin-flip floor; a defense that halves the leakage but also halves accuracy is a wash, and a
defense that destroys accuracy to win privacy scores near zero. For the no-defense floor, both terms
work against me wherever the model overfits: the overfit *is* the high AUC, and the datasets where the
model overfits hardest are also, on these benchmarks, the ones where test accuracy is lowest. The two
penalties compound.

Let me read the composite as an exchange rate, because it dictates whether any given defense is worth
buying. Wherever `mia_auc > 0.5`, the `max(·, 0)` is inactive and locally
`privacy_score = test_acc − (mia_auc − 0.5)`, so a small change gives
`Δprivacy_score = Δtest_acc − Δmia_auc`. In this regime a defense earns net score if and only if it drops
the attack AUC by *more* than the accuracy it costs — shave 0.05 off `mia_auc` while losing 0.03 accuracy
and I net `+0.02`; shave 0.02 while losing 0.03 and I net `−0.01` and would have been better off doing
nothing. That one-for-one trade is the whole economics of the ladder: privacy is only worth its accuracy.
But the moment a defense pushes `mia_auc` down to 0.5, the `max` clamps and the derivative with respect to
`mia_auc` goes to zero — further reduction in leakage buys *nothing*, while any accuracy it costs is pure
loss. So there is a sharp corner in the objective: under-defend and I leave leakage on the table;
over-defend a benchmark that is already near the coin-flip floor and every point of accuracy I sacrifice
is subtracted for no return. That corner is precisely why FashionMNIST is the dangerous benchmark for a
defense — if its attack is already near 0.5 under plain training, there is no AUC left to win there, and a
regularizer that costs accuracy on it is strictly negative. The floor's job is to tell me where each
benchmark sits relative to that corner before I spend any accuracy.

There is also a `K`-dependence in the confidence signal that will shape which benchmark leaks hardest,
and it is worth pinning down. The attack thresholds the max softmax probability, whose floor is `1/K`
(a uniform prediction). On the 10-class problems (CIFAR-10, FashionMNIST) an unconfident output bottoms
out around 0.1; on the 100-class problem (CIFAR-100) it bottoms out around 0.01. So CIFAR-100 has a far
wider dynamic range for confidence to travel — a memorized member can still pin near 1.0 while a genuinely
uncertain non-member can sit an order of magnitude lower than anything possible at `K = 10`. More room
between "certain" and "uncertain" means a member/non-member confidence gap can open wider before the
ceiling saturates it, which compounds the capacity argument: not only can VGG memorize 100-class members,
but the confidence scale itself gives that memorization more room to separate from the non-members. That
is a second, independent reason to expect CIFAR-100 to be the leakiest of the three.

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

I should be honest that there is a real choice even at rung one, and say why I take the plainer branch.
Two options are open: leave the default cross-entropy and measure the undefended floor, or skip straight
to one of the named regularizers and start climbing immediately. The pull toward the second is that the
floor "does nothing" and feels like a wasted run. But the exchange-rate argument I just made kills that
impulse: a defense can *lose* net privacy_score if its AUC drop does not outrun its accuracy cost, and I
cannot tell whether any given AUC reduction paid for itself unless I know what AUC and accuracy the model
reaches with no defense at all. Without the floor I would be comparing a defended number against nothing.
Worse, the three published regularizers form a lineage in which each one is a reaction to the specific
way the previous one falls short — so I need not just *a* reference but *this* reference, the exact
per-benchmark leakage that the first softening trick will be asked to reduce. And the floor is nearly free
to obtain: it is the default fill, one line, no new failure modes. So the plain branch dominates — I
establish the floor, read off which benchmarks sit far from the 0.5 corner (where defending pays) and
which sit near it (where defending is dead weight), and only then spend accuracy. Choosing to "do nothing"
here is itself the reasoned move.

It is also worth naming what max-softmax as the attack score is and is not. It is a cheap, black-box
proxy for the per-sample loss — no shadow models, no per-example loss calibration, just one forward pass
and a max. That makes it a *lower bound* on what a determined adversary could extract: a stronger attacker
could calibrate per-example thresholds or use the full loss, so beating this attack is necessary but not
by itself proof of privacy. But it is the exact number the harness ranks me on, so it is the number I
fight, and its cheapness cuts both ways — because it only reads the top probability, a defense that
reshapes the *whole* output distribution (not just the peak) has surface area to work on that this
particular attacker may not fully exploit.

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

One structural fact about the *shapes* of the two humps is worth extracting now, because it changes what
"overlap" would even require. The member confidences are all driven toward the same ceiling by the same
relentless objective, so the member hump is *narrow* — a tight spike near 1.0, low variance. The
non-member confidences are never optimized; they are whatever the model happens to output on unseen data,
scattered across correct-but-uncertain, wrong-but-confident, and everything between — so the non-member
hump is *wide*, high variance, and sits at a lower mean. That asymmetry — a narrow high spike against a
broad lower cloud — is doubly favorable to the attacker: the member spike is both far from the non-member
mean and thin enough that almost none of its mass reaches down into the non-member range. To make them
overlap I would therefore need to do one of two things (or both): drag the member mean *down* toward the
non-member cloud, or *widen* the member spike until its lower tail bleeds into that cloud. A pure
translation touches neither the ordering nor the shape, so it fails; shrinking only the mean gap without
widening leaves two humps of different width that a threshold can still cut. Keeping this narrow-spike vs
broad-cloud picture explicit is the thing that will let me judge each future defense by asking not "did it
lower confidence" but "did it move the mean or change the width" — the only two questions the AUC answers.

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
