The defense is the whole point, but it bolts onto a trained classifier, and with no defense that
classifier is the floor — so the thing to start from is just training the model on the private half of
the data with the ordinary objective and seeing how badly it leaks. The frozen loop hands me one
method, `compute_loss(logits, labels, epoch)`, and the default fill is the obvious one: standard
cross-entropy, `F.cross_entropy(logits, labels)`. Minimize `−log p(y)` against the one-hot target, no
privacy machinery, no schedule tricks. That is the baseline the three published regularizers all react
to, so I want its number on the board before I touch anything.

The reason this is the weakest possible setting is structural, not incidental. Cross-entropy on a
one-hot target places its optimum at an *infinite* correct-vs-rest logit gap: `p(y) → 1` only as
`z_y − z_k → +∞` for every wrong class `k`. The per-logit gradient is `p − y`, bounded in `[−1, 1]`, so
each step is gentle, but the target it chases sits at infinity and never vanishes until the loss is
essentially zero — nothing in the objective tells the optimizer to *stop* being confident on a training
example. And the gradient only ever points toward higher confidence, never away from it: as `p → y` the
update on that sample shrinks toward zero, so ERM naturally *slows* on the examples it has fit but never
pushes back. So over 300 epochs with augmentation off, the model grinds member losses down to ~0 and
member confidence up to ~1, while non-members, never optimized, sit lower and more scattered. Two
separated humps — and the fixed attack is exactly a threshold on that confidence. The training procedure
is manufacturing the precise signal the adversary reads. This is the structural reason a real defense
cannot just make the pull *smaller*; it has to introduce an active force that pushes confidence back
*down*, because ERM's own dynamics will otherwise fill in any slack.

The loop gives that runaway an enormous amount of room. It splits the full training set 50/50 by index,
so on the CIFAR benchmarks the member half is 25,000 images and one epoch is `⌈25000/128⌉ = 196`
gradient steps — roughly 58,800 updates over 300 epochs, each member revisited about 300 times as
literally the same pixels, never a jittered crop (FashionMNIST's member half is 30,000, about 70,500
updates). And MultiStepLR decays ×0.1 at epochs 150 and 225, so the two low-lr tails are exactly when a
network stops exploring and polishes the training examples it already nearly fits, driving their losses
the last stretch toward zero. The schedule tail is a memorization-sharpener, and ERM has nothing that
fights it.

Now be exact about what the attacker reads, because the whole ladder is aimed at this one quantity. After
training the loop takes the *max softmax probability* of every member and every non-member — the model's
confidence in its own top prediction — and reports the AUC of that scalar as a member/non-member
discriminator, computed by the Mann–Whitney U statistic so that 0.5 is a coin flip and 1.0 is perfect
separation. Max-confidence is a monotone stand-in for the per-sample loss: a memorized member has
near-zero loss and near-1 confidence; a non-member the model never optimized has higher loss and lower,
more scattered confidence. So the AUC the harness reports is, operationally, "how well does a threshold
on confidence tell members from non-members."

That the AUC is a pure *rank* statistic is the single property that will decide which defenses can
possibly work. It equals `P(conf of a random member > conf of a random non-member)` — the fraction of
member/non-member pairs where the member is more confident. So any strictly increasing map `f` applied
to all confidences turns every `c_m > c_nm` into `f(c_m) > f(c_nm)` and leaves U invariant: a uniform
downward shift, subtracting a constant from everyone, can move the confidences a long way without
flipping a single pairwise comparison, so the AUC is exactly unchanged. Only an operation that
*reorders* member and non-member confidences — pulling some member below some non-member — can move it.
Relocating a distribution is not the same as making two distributions overlap, and the attack only cares
about overlap.

This is exactly where `privacy_gap` and `mia_auc` come apart, which is why the harness reports both.
`privacy_gap` is the difference of means, `mean(member conf) − mean(non-member conf)`, so it too is
translation-invariant and ignores a pure shift. But the two diverge under *reshaping*: widen the member
cluster so one member drops below a non-member, one pair flips, and the AUC falls while the mean barely
moves. `privacy_gap` tracks the first moment; `mia_auc` tracks full separability. A defense can shrink
the gap and leave the AUC flat (relocate the mean), or leave the gap roughly fixed and drop the AUC
(fatten the tails into overlap). The metric that ranks me is `mia_auc`, so overlap is the only thing
that ultimately counts, and `privacy_gap` is a diagnostic telling me whether I moved the mean without
moving separability.

One reason this readout is clean and worth trusting: members and non-members are the *same* underlying
distribution, split only by index within one shuffled training set — not a train-set-versus-different-
test-set comparison. So there is no distribution shift to confound the attack; any systematic confidence
difference between the two groups can *only* be an artifact of one group having been optimized and the
other not. The AUC is an honest thermometer of memorization, and driving it toward 0.5 is precisely
equivalent to erasing the train/non-train distinction the loop created.

The composite the leaderboard ranks on, `privacy_score = test_acc − max(mia_auc − 0.5, 0)`, is built to
punish the cheap escape. Wherever `mia_auc > 0.5` the `max` is inactive and locally
`Δprivacy_score = Δtest_acc − Δmia_auc` — a one-for-one trade, so a defense earns net score only if it
drops the attack AUC by *more* than the accuracy it costs. But the moment a defense pushes `mia_auc`
down to 0.5, the `max` clamps and the derivative goes to zero: further reduction in leakage buys
nothing, while any accuracy it costs is pure loss. That sharp corner is why a benchmark already sitting
near 0.5 under plain training is dangerous ground for a defense — there is no AUC left to win there and a
regularizer that costs accuracy is strictly negative. The floor's job is to tell me where each benchmark
sits relative to that corner before I spend any accuracy.

There is also a `K`-dependence in the confidence signal worth pinning down. The max softmax floor is
`1/K`, so an unconfident output bottoms out around 0.1 on the 10-class problems (CIFAR-10, FashionMNIST)
but around 0.01 on the 100-class one (CIFAR-100). CIFAR-100 has a far wider dynamic range for confidence
to travel: a memorized member can still pin near 1.0 while a genuinely uncertain non-member sits an order
of magnitude lower, so the member/non-member gap can open wider before the ceiling saturates it. That
compounds the capacity argument — a second, independent reason to expect CIFAR-100 to leak hardest.

One structural fact about the *shapes* of the two humps is worth extracting now, because it changes what
overlap would even require. The member confidences are all driven toward the same ceiling by the same
relentless objective, so the member hump is *narrow* — a tight spike near 1.0. The non-member confidences
are never optimized; they are whatever the model happens to output on unseen data, scattered across
correct-but-uncertain, wrong-but-confident, and everything between — so the non-member hump is *wide* and
sits at a lower mean. A narrow high spike against a broad lower cloud is doubly favorable to the
attacker. To make them overlap I would have to drag the member mean *down* toward the non-member cloud,
or *widen* the member spike until its lower tail bleeds into that cloud — a pure translation touches
neither and fails. That gives me the only two questions worth asking of every future defense: did it move
the mean, or change the width. Those are the only two things the AUC answers.

The three benchmarks should split on that overfit axis, because the membership signal *is* the overfit.
On **vgg16bn-cifar100** I expect the worst leakage by far: a hundred classes, a high-capacity VGG, no
augmentation, so the model memorizes the member half almost perfectly while the test set stays hard — a
wide confidence gap and a high AUC, on *top* of low test accuracy, so both terms of the composite hurt at
once and the score should be smallest by a large margin. On **resnet20-cifar10** I expect strong but
milder leakage: ResNet-20 is too small to memorize CIFAR-10 perfectly, test accuracy is high (~0.8), so
the gap is real but narrower. On **mobilenetv2-fmnist** the least: FashionMNIST is easy, the model
generalizes, train and test confidence both sit near the ceiling, leaving little gap to threshold even
under plain training — the benchmark most likely to *cost* accuracy without buying privacy. That the
harness turns train-time augmentation off is deliberate: augmentation is the single most effective
implicit regularizer against memorization on these datasets, and removing it guarantees there *is* a
measurable membership signal to defend against. The setting is adversarial by construction, and the only
lever it leaves me is the per-minibatch scalar loss.

Two things the loop hands me that ERM ignores are worth filing away. The signature includes `epoch`, so a
method could change its behavior over training — anneal a penalty, alternate phases, switch regimes after
the learning-rate decay — and it can read `logits.size(1)` to learn `K` and adapt per dataset. Plain
cross-entropy needs neither: it is the same objective every epoch and identical at every `K`. But the
contract exposing them tells me the harness anticipates losses that are not time-invariant or `K`-blind,
and the strongest defense on this ladder will probably be the one that actually uses them.

There is a real choice even at rung one — leave the default and measure the undefended floor, or skip
straight to a named regularizer — and the exchange-rate argument settles it: I cannot tell whether any
AUC reduction paid for itself unless I know what AUC and accuracy the model reaches with no defense at
all. And the three regularizers form a lineage in which each reacts to the specific way the previous one
fell short, so I need not just *a* reference but *this* one — the exact per-benchmark leakage the first
softening trick will be asked to reduce. The floor is one line, no new failure modes; choosing to do
nothing here is the reasoned move. One caveat on the number I fight: max-softmax is a cheap black-box
proxy for the per-sample loss, a *lower bound* on what a determined adversary could extract, but it is
the exact number the harness ranks me on — and its cheapness cuts both ways, since a defense that
reshapes the *whole* output distribution has surface area the peak-only attacker may not exploit.

So there is nothing to derive; ERM is the absence of a defense, and what I establish is the reference the
ladder climbs away from — a model that fits its members so confidently that a single threshold on
confidence separates them, with the damage scaling with how badly each benchmark overfits. The diagnosis
already points at the next step: this is a *confidence* problem, not a fitting problem. The model fits
fine; it fits *too* confidently. The fix has to reach into the output distribution and put a floor under
member confidence, and the simplest way the lineage reached for first is to soften the one-hot target
itself. The distilled scaffold fill is in the answer.
