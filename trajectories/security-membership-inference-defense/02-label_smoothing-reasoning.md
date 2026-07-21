The no-defense run told me exactly what is wrong, and it told me in numbers. On
**resnet20-cifar10** the attack reached 0.7275 AUC at 0.7972 test accuracy — well above the 0.5
coin-flip floor, so confidence really does separate members from non-members, but the network is small
enough that it can't memorize CIFAR-10 perfectly and the leak is moderate. On **vgg16bn-cifar100** the
leak is brutal: 0.8677 AUC at only 0.5045 accuracy, privacy_gap 0.1479 — the high-capacity VGG memorizes
the member half while the test set stays hard, so the confidence gap is wide and the composite
privacy_score collapses to 0.1368, the worst of the three. On **mobilenetv2-fmnist** the leak is mild
(0.5575 AUC at 0.926 accuracy): FashionMNIST is easy, the model generalizes, members and non-members
both sit near the confidence ceiling. So the diagnosis is confirmed and sharp — leakage scales with
overfit, concentrated exactly where the model can drive member confidence to ~1 while leaving
non-members behind. It is a *confidence* problem, and the first thing to try is the simplest intervention
that puts a floor under that confidence: stop asking the model to be perfectly certain on its members.

The composite penalizes the attack's margin above the coin flip, `mia_auc − 0.5`: that is 0.2275 on
resnet20-cifar10, 0.3677 on vgg16bn-cifar100, 0.0575 on mobilenetv2-fmnist. CIFAR-100's margin is
`0.3677 / 0.0575 ≈ 6.4×` FashionMNIST's — the overfit axis I predicted at the floor, now quantified. That
locates where a defense can matter: FashionMNIST already sits within 0.0575 of the 0.5 corner where AUC
reduction stops paying, so there is almost nothing to win there and real accuracy to lose, while
CIFAR-10 and CIFAR-100 have 0.2275 and 0.3677 of margin on the table respectively — that is where a
defense earns its keep, if it can actually move separability rather than just relocate confidence.

The privacy_gap column sharpens the picture. FashionMNIST's gap is a tiny 0.0211 with AUC 0.5575,
consistent with the near-floor AUC; VGG's is 0.1479 with AUC 0.8677, and ResNet's 0.0924 with AUC 0.7275
— a larger mean gap tracks a larger AUC, but not linearly: ResNet has 62% of VGG's gap yet a much closer
AUC, which tells me the two humps' *widths* matter too, not just the distance between their means. That
is the mean-versus-shape distinction I fixed at the floor, and it is the lens I hold up to smoothing:
does it change the gap, the widths, or only slide everything.

The ERM flaw, established at the floor, is that the one-hot target lives at infinity, so the correct
logit runs off unbounded and member confidence pins at the ceiling. If that is the disease, the cure
suggests itself: don't put the target at infinity. Bleed a little mass `ε` off the one-hot onto a fixed
distribution `u`,

  `q'(k) = (1 − ε) δ_{k,y} + ε u(k)`,

and on these class-balanced vision datasets the natural `u` is uniform, so
`q'(k) = (1 − ε)δ_{k,y} + ε/K`. This kills the runaway: every entry of `q'` is at least `ε/K > 0`, so if
`z_y` tried to run off to `+∞`, then `p(k) → 0` for `k ≠ y` and the cross-entropy
`−Σ_k q'(k) log p(k)` would blow up on those wrong-class terms, because `q'(k) = ε/K` is positive while
`log p(k) → −∞`. An infinite logit gap is now infinitely *expensive*, not free; the target sits at a
finite, `ε`-controlled configuration of the logits — a bounded member confidence, exactly the floor I
wanted against the attack.

Cross-entropy is linear in the target, so
`H(q', p) = (1 − ε)·H(q, p) + ε·H(u, p)` — ordinary hard-label cross-entropy downweighted by `(1 − ε)`,
plus an `ε`-weighted term `H(u, p)`. And `H(u, p) = D_KL(u‖p) + H(u)` with `H(u)` constant, so up to a
constant that second term is a penalty on how far `p` has drifted from uniform: "stay a bit humble." For
the harness this collapses to a single call — PyTorch's `F.cross_entropy` takes a `label_smoothing`
argument that constructs `q'` and computes `H(q', p)` internally, so I hand back
`F.cross_entropy(logits, labels, label_smoothing=ε)`, one number per minibatch, no soft target to
materialize.

Cross-entropy against the fixed target `q'` is minimized at `p = q'`, so the best the model can do on a
member is output the smoothed target itself, confidence `p(y) = (1 − ε) + ε/K`. At `ε = 0.1` that ceiling
is `0.91` for `K = 10` and `0.901` for `K = 100` — capped at about 0.91 essentially regardless of class
count, down from the ~1.0 that ERM's near-zero member losses imply. A real, computed drop of ~0.09 in the
member ceiling. (As `ε → 0` the ceiling returns to 1.0, i.e. ERM, so smoothing is a continuous
deformation off the floor; as `ε → (K−1)/K` it collapses to `1/K`, destroying signal and accuracy alike.)
But `ε = 0.1` sits very near the ERM end of that continuum — the ceiling moved only ~0.09 — my first
quantitative warning that the intervention is mild.

And that ~0.09 drop is unlikely to buy ~0.09 worth of AUC. The cap applies to every member the same way,
and because the smoothing term is in the training loss it reshapes the model's outputs on non-members too
— the network that learns to sit at 0.91 on members also sits lower on the non-members it generalizes to.
So both humps slide down by comparable amounts, and by the rank-statistic argument from the floor a
roughly common downward shift leaves nearly every pairwise comparison unflipped, so the AUC barely moves
even as the *gap* shrinks. This is the wall this rung is built to expose: a mean-shifting regularizer
relocates the member distribution rigidly, and a threshold attacker reads separability, not absolute
level — if the shift overshoots, members and non-members can stay about as separable as before, just
relocated, while the accuracy cost shows up directly in the `test_acc` term.

There is a sharper structural mismatch hiding in the `ε/K` term, and it cuts against exactly the
benchmark that needs help most. The uniform component puts mass `ε/K` on each wrong class, and the
forward-KL penalty `H(u, p)` weights every class by that same `1/K`. So on CIFAR-10 each wrong class
carries target mass `0.1/10 = 0.01`, but on CIFAR-100 only `0.1/100 = 0.001` — a tenth the grip per
class. The total off-target mass is `ε = 0.1` either way, but smeared across 99 classes versus 9, so the
pressure on any *individual* competing logit is an order of magnitude weaker on the 100-class problem.
That is perverse: CIFAR-100 is where ERM leaked hardest (margin 0.3677, the 6.4× case), yet where
smoothing's per-class humbling is thinnest. A concrete reason to doubt it can pull the 0.8677 wall down
at all.

The accuracy cost should be small, and the exchange rate makes that decisive. Test accuracy is an argmax
property — it depends only on which logit is largest — and smoothing does not change which class the model
prefers, only how confident it is allowed to be, so to first order it is near-neutral on accuracy and as a
regularizer can even help slightly. The one way it costs accuracy is the equidistant-cluster geometry:
forcing every wrong class equally far from the activation can, on a genuinely hard sample near a two-class
boundary, nudge the argmax across it — a small effect of uncertain sign. So my accuracy prediction is
"within a point or so of ERM, either direction," and combined with an AUC I expect to barely move,
`Δprivacy_score = Δtest_acc − Δmia_auc` should land near zero on every benchmark.

The geometry is worth seeing, because it is where the "will it beat the attack" question lives. The logit
`z_k = x^T w_k`, and `‖x − w_k‖² = x^Tx − 2z_k + w_k^Tw_k`; the `x^Tx` term cancels in the softmax and
`w_k^Tw_k` is roughly constant across templates, so what varies with `k` is `−2z_k` — the softmax is a
soft nearest-template classifier in penultimate space. With hard targets only the gap `z_y − z_k` matters
and wants to be huge, so activations sprawl at large magnitude, over-confidence visible as scale. With
smoothing the wrong-class target is the *same* `ε/K` for every wrong class, so the loss wants all
wrong-class logits equal — `x` equidistant from every incorrect template — at a finite gap to the correct
one: tight, equally-separated clusters at a ceiling strictly below 1. But equidistant is still
*sample-blind*: it applies the same pressure to a memorized member and an uncertain one, which is exactly
why I expect it to relocate the member hump without changing its separability.

There is a reason to spend smoothing *first* even expecting it to underwhelm, and it is safety. Smoothing
trains toward a strictly-positive target, so its gradient is ordinary cross-entropy — always a descent
direction, never a term that can push the loss *up*, no regime where the objective fights itself. That
matters because the CIFAR-100 setting, at test_acc 0.5045 on a 100-class problem it is only just managing
to fit, is the one an aggressive defense is most likely to destabilize — and the other target-humbling
option on the shelf, subtracting an output-distribution penalty, *can*, if strong or badly timed,
overpower the fitting term and drive the objective the wrong way. So the disciplined order is to measure
how much a perfectly stable, collapse-proof confidence cap buys, and only then decide whether a riskier
penalty's adaptivity is worth its collapse risk on the fragile case. Running the riskier method first
would confound "the adaptivity helped" with "I got lucky on stability."

The knob is `ε`, the fraction of mass moved to uniform, trading fitting against humility: too small and
the floor is too high to dent the attack, too large and accuracy craters. The vision default is
`ε = 0.1`, and I have no task-specific reason to move off it — the datasets are class-balanced so uniform
`u` is right, and 0.1 lifts accuracy as a regularizer without flattening the output. I do consider
raising `ε` on CIFAR-100 to fight the 10× thinner grip, and reject it: restoring per-class mass `0.01` at
`K = 100` would need `ε = 1.0`, the fully-uniform target that destroys signal and accuracy, and even
reaching partway means lowering the true-class ceiling on a benchmark already at 0.5045 — spending
accuracy I cannot afford. More fundamentally, a bigger *uniform* push does not fix a *uniformity*
problem; the right response to the dilution is an *adaptive* push, a different method entirely. So I hold
`ε = 0.1` fixed across all three benchmarks, and since smoothing has no reason to vary by epoch,
`compute_loss` ignores the `epoch` argument the loop hands me.

So the honest expectations against the ERM numbers. Smoothing puts a hard floor under member confidence,
so I expect the member ceiling to fall — plausibly the privacy_gap with it, though the regularizer moves
the non-member outputs too and I cannot be sure of the net. But I expect the *attack AUC to move little*:
the shift is roughly uniform across members and non-members and does not change their separability, so on
the CIFAR-100 wall (0.8677) I would be surprised to see smoothing pull it out of the high-0.8s, and its
small accuracy cost could leave the composite flat or slightly worse than ERM on every benchmark rather
than better. If that holds — confidence floored but separability untouched — it is the precise evidence
that I need a regularizer that changes the *shape* of the member distribution, not just its location:
something adaptive that pushes hardest on the over-confident predictions. That is the next rung. The
distilled scaffold fill is in the answer.
