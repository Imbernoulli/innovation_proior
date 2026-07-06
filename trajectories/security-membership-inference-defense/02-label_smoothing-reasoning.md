The no-defense run told me exactly what is wrong, and it told me in numbers. On
**resnet20-cifar10** the attack reached 0.7275 AUC at 0.7972 test accuracy — well above the 0.5
coin-flip floor, so confidence really does separate members from non-members, but the network is small
enough that it can't memorize CIFAR-10 perfectly and the leak is moderate. On
**vgg16bn-cifar100** the leak is brutal: 0.8677 AUC at only 0.5045 accuracy, privacy_gap 0.1479 — the
high-capacity VGG memorizes the member half while the test set stays hard, so the confidence gap is
wide and the composite privacy_score collapses to 0.1368, the worst of the three. On
**mobilenetv2-fmnist** the leak is mild (0.5575 AUC at 0.926 accuracy): FashionMNIST is easy, the model
generalizes, members and non-members both sit near the confidence ceiling, and there is little gap to
threshold. So the diagnosis is confirmed and sharp — leakage scales with overfit, and the damage is
concentrated exactly where the model can drive member confidence to ~1 while leaving non-members
behind. It is a *confidence* problem. Cross-entropy's optimum sits at an infinite logit gap and never
stops rewarding confidence, so members pile up at the ceiling. The first thing to try is the simplest
intervention that puts a floor under that confidence: stop asking the model to be perfectly certain on
its members in the first place.

Before I reach for the cure let me read the floor's numbers as a mechanism, because the arithmetic tells
me exactly how much room there is and where. The quantity the composite actually penalizes is the margin
of the attack above the coin flip, `mia_auc − 0.5`: on resnet20-cifar10 that is `0.7275 − 0.5 = 0.2275`,
on vgg16bn-cifar100 `0.8677 − 0.5 = 0.3677`, on mobilenetv2-fmnist `0.5575 − 0.5 = 0.0575`. CIFAR-100's
leakage margin is `0.3677 / 0.0575 ≈ 6.4×` FashionMNIST's — the overfit axis I predicted at the floor,
now quantified. And the composite is exactly these margins subtracted from accuracy: `0.7972 − 0.2275 =
0.5697`, `0.5045 − 0.3677 = 0.1368`, `0.9260 − 0.0575 = 0.8685`, matching the three reported
privacy_scores to the digit. So the scoreboard is not hiding anything: CIFAR-100 scores worst because it
is punished on *both* terms at once (lowest accuracy 0.5045 *and* largest AUC margin 0.3677), while
FashionMNIST scores best because it is barely leaking (0.0575 margin) on top of high accuracy. That
locates where a defense can matter. FashionMNIST already sits within 0.0575 of the 0.5 corner where AUC
reduction stops paying, so there is almost nothing to win there and real accuracy to lose. CIFAR-10 and
CIFAR-100 have 0.2275 and 0.3677 of AUC margin on the table respectively — that is where a defense earns
its keep, if it can actually move separability rather than just relocate confidence.

The privacy_gap column sharpens the picture in a way I should not skip, because gap and AUC are different
readouts and their relationship is diagnostic. FashionMNIST's gap is a tiny 0.0211 with AUC 0.5575 — the
member and non-member confidence means are almost coincident, consistent with the near-floor AUC. VGG's
gap is 0.1479 with AUC 0.8677, and ResNet's gap is 0.0924 with AUC 0.7275 — so a larger mean gap tracks a
larger AUC, as expected, but not linearly: ResNet has 62% of VGG's gap yet a much closer AUC, which tells
me the two humps' *widths* also matter, not just the distance between their means. That is the same
mean-versus-shape distinction I fixed in my head at the floor, and it is the lens I have to hold up to
smoothing: I need to know whether smoothing changes the gap, the widths, or only slides everything.

Let me start from what actually bugs me about the ERM objective, because the fix should come out of it.
I have a softmax head, `p(k) = exp(z_k)/Σ_i exp(z_i)`, trained with cross-entropy against a one-hot
target `q(k) = δ_{k,y}`, so the loss on an example is `−log p(y)`. To make `p(y) → 1` I need `z_y` to
dominate every other logit by an *unbounded* margin: `p(y) → 1` only as `z_y − z_k → +∞` for all
`k ≠ y`. The one-hot target is a maximum I can never reach at finite logits; it just keeps pulling the
correct logit further above the rest, forever. The gradient `∂ℓ/∂z_k = p(k) − q(k)` is bounded in
`[−1, 1]`, so each step is gentle, but the target it chases sits at infinity. That is precisely the
0.8677-AUC behavior on CIFAR-100: nothing stops `z_y` from running off, so member confidence pins at
the ceiling and the gap to the never-optimized non-members blows open.

If the problem is "the target lives at infinity, so confidence runs away," the cure suggests itself:
don't put the target at infinity. Give every class a little floor of target probability so the correct
logit has no incentive to escape to `+∞`. Take the one-hot and bleed a little mass `ε` onto a fixed
distribution `u` over labels:

  `q'(k) = (1 − ε) δ_{k,y} + ε u(k)`.

The natural `u`, absent any prior knowledge and on these class-balanced vision datasets, is uniform,
`u(k) = 1/K`, so `q'(k) = (1 − ε)δ_{k,y} + ε/K`. Now check that this kills the runaway. Every entry of
`q'` is at least `ε/K > 0`. If `z_y` tried to run off to `+∞`, then `p(y) → 1` and `p(k) → 0` for
`k ≠ y`, and the cross-entropy `−Σ_k q'(k) log p(k)` would blow up on those wrong-class terms, because
`q'(k) = ε/K` is positive while `log p(k) → −∞`. So an infinite logit gap is now infinitely
*expensive*, not free. The target no longer sits at infinity; it sits at a finite, `ε`-controlled
configuration of the logits — which means a finite, bounded member confidence, which is exactly the
floor I wanted against the attack.

Let me rewrite the loss to see its structure, because that tells me how to implement it and what to
expect. Cross-entropy is linear in the target, so

  `H(q', p) = −Σ_k q'(k) log p(k) = (1 − ε)·H(q, p) + ε·H(u, p)`,

ordinary hard-label cross-entropy downweighted by `(1 − ε)`, plus an `ε`-weighted term `H(u, p)` that
pulls the prediction toward the prior `u`. And `H(u, p) = D_KL(u‖p) + H(u)`; `H(u)` is constant, so
that second term is, up to a constant, a penalty on how far `p` has drifted from uniform. It is a
regularizer that says "stay a bit humble, don't get too far from the prior." For the harness this is
even simpler: PyTorch's `F.cross_entropy` takes a `label_smoothing` argument that constructs exactly
this `q'` and computes `H(q', p)` internally, so the entire defense is the single call
`F.cross_entropy(logits, labels, label_smoothing=ε)` — I do not need to materialize the soft target or
hand-write the two-term split; the loop calls my `compute_loss` once per minibatch and I hand back that
one number. (This is the same criterion as a hand-rolled `(1−ε)·nll + ε·mean_k(−log p_k)` module, but
the built-in flag is the literal edit the scaffold wants, and it is numerically identical.)

Let me compute the confidence ceiling this actually imposes, because that number is what determines
whether it can dent the leakage margins I just read off. Cross-entropy against the fixed target `q'` is
minimized exactly when `p = q'`, so the *best* the model can do on a member is to output the smoothed
target itself — its confidence on the true class is then `p(y) = (1 − ε) + ε/K`. With `ε = 0.1` and
`K = 10` that ceiling is `0.9 + 0.1/10 = 0.91`; with `K = 100` it is `0.9 + 0.1/100 = 0.901`. So smoothing
caps member confidence at about 0.91 essentially regardless of the class count — down from the ~1.0 that
ERM's near-zero member losses imply. That is a real, computed drop of roughly 0.09 in the member ceiling.
Two limiting checks confirm the knob behaves: as `ε → 0` the target returns to one-hot and the ceiling
returns to 1.0, i.e. ERM exactly, so smoothing is a continuous deformation away from the floor; as
`ε → (K−1)/K` the target becomes uniform and the ceiling collapses to `1/K`, destroying both the signal
*and* the accuracy.

Let me verify the finite-gap claim on the smallest possible case, because "the target no longer lives at
infinity" should show up as an actual finite number. Take `K = 2` and `ε = 0.1`, so the smoothed target is
`[0.95, 0.05]`. The loss is minimized at `p = [0.95, 0.05]`, and for a two-logit softmax
`p(y) = σ(z_y − z_k)`, so the correct-vs-wrong logit gap the optimizer settles at is
`z_y − z_k = logit(0.95) = ln(0.95/0.05) = ln 19 ≈ 2.94`. So smoothing pins the winning margin at about
2.94 nats — a finite, `ε`-set value — where ERM's one-hot target drives that same gap to `+∞`. That is the
mechanism made numeric: a bounded logit gap, hence a bounded confidence, hence a member spike that stops
at 0.95 instead of 1.0. It is exactly the floor under confidence I wanted; the open question the numbers
must answer is whether a *bounded* spike is any less separable from the non-member cloud than an unbounded
one, and the rank-statistic argument says: not unless the non-members fail to slide down with it. At `ε = 0.1` I sit very near the ERM end of that continuum — the ceiling has moved
only ~0.09 — which is my first quantitative warning that the intervention is mild.

Here is why that ~0.09 drop is unlikely to translate into a ~0.09-worth reduction in the attack. The
ceiling `0.91` is applied to *every* member the same way, and because the smoothing term is in the
training loss it reshapes the model's outputs on non-members too — the network that learns to sit at 0.91
on members will also, at inference, sit lower on the non-members it generalizes to. So both humps slide
down by comparable amounts. Recall the rank-statistic arithmetic from the floor: subtracting a roughly
common amount from both member and non-member confidences leaves almost every pairwise comparison
unflipped, so the AUC barely moves. A cap that lowers the member mean from ~1.0 to ~0.91 is, to the
attacker, close to the flat-shift example that left AUC at 1.0 — the *gap* may shrink (and I expect
privacy_gap to fall below ERM's 0.0924/0.1479/0.0211) but the *separability* need not. That is the wall
this rung is built to expose.

There is a sharper structural mismatch hiding in the `ε/K` term, and it cuts against exactly the
benchmark that needs help most. The uniform component of the smoothed target puts mass `ε/K` on each
wrong class, and the forward-KL penalty `H(u, p)` weights every class by that same `1/K`. So on CIFAR-10
each wrong class carries target mass `0.1/10 = 0.01`, but on CIFAR-100 each wrong class carries only
`0.1/100 = 0.001` — a tenth as much grip per class. The total off-target mass is `ε = 0.1` either way,
but it is smeared across 99 classes on CIFAR-100 versus 9 on CIFAR-10, so the pressure the smoothing term
places on any *individual* competing logit is an order of magnitude weaker on the 100-class problem. That
is perverse: CIFAR-100 is where ERM leaked hardest (AUC margin 0.3677, the 6.4× case), yet it is exactly
where smoothing's per-class humbling is thinnest. The confidence *ceiling* is still ~0.9 there, but the
force keeping any single rival logit alive is diluted 10×, so I should expect smoothing to do even less
against the CIFAR-100 attack than against CIFAR-10's — the opposite of what a well-matched defense would
do. This is a concrete reason to doubt smoothing can pull the 0.8677 wall down at all.

I should also reason about the accuracy cost directly, because the exchange rate makes it decisive. Test
accuracy is an argmax property — it depends only on *which* logit is largest, not on how large — and
smoothing does not change which class the model prefers on an easy example; it only caps how confident it
is allowed to be. So to first order smoothing should be near-neutral on accuracy, and as a regularizer it
can even help slightly by discouraging degenerate large-magnitude solutions. The one way it *costs*
accuracy is through the equidistant-cluster geometry: forcing every wrong class equally far from the
activation can, on a genuinely hard sample sitting near a two-class boundary, nudge the argmax across that
boundary. That effect is small and its sign is not guaranteed. So my accuracy prediction is "within a
point or so of ERM, either direction," and combined with an AUC that I expect to barely move, the
composite `Δprivacy_score = Δtest_acc − Δmia_auc` should land close to zero on every benchmark — smoothing
buys little because it pays little and moves little. On the benchmark where it might have mattered most,
CIFAR-100, the `1/K` dilution argues it does the least. If that holds, the message for the next rung is
unambiguous: uniform target-humbling is the wrong tool, and I need pressure that scales with how spiked
each *individual* output is, not a flat `ε/K` on every class.

Now I want to understand what this does to the network geometrically, because that is where the "will it
actually beat the attack" question lives, and I am suspicious it will not beat it by much. The logit for
class `k` is `z_k = x^T w_k`, where `x` is the penultimate activation and `w_k` is the class template.
Look at a squared distance: `‖x − w_k‖² = x^T x − 2 x^T w_k + w_k^T w_k`. The `x^T x` term is the same
for every class (cancels in the softmax), and `w_k^T w_k` is roughly constant across templates, so what
varies with `k` is `−2 z_k`. The softmax is a soft nearest-template classifier in penultimate space.
With hard targets, only the gap `z_y − z_k` matters and it wants to be huge; the wrong-class logits are
otherwise free, so activations sprawl at large magnitude — the over-confidence, now visible as scale.
With smoothing, the wrong-class target is the *same* value `ε/K` for *every* wrong class, so the loss
wants all wrong-class logits equal — `x` equidistant from every incorrect template — and a *particular
finite* gap to the correct one. Smoothing drives each activation close to its own template and equally
far from all others, at bounded magnitude. Tight, equally-separated clusters at a confidence ceiling
that is now strictly below 1.

Here is the catch, and it is exactly what I have to confront given the attack I am facing. Smoothing
applies the *same* pressure to *every* sample: it pulls every output toward uniform by roughly the same
amount, member and non-member alike (the regularizer is in the training loss, but the bounded-magnitude
geometry it imposes shows up at inference on both populations). So it *translates* the member-confidence
hump down to a lower mean — good, the member confidences no longer pin at the ceiling — but it
translates the whole distribution more or less rigidly, and the spread barely changes. A threshold
attacker does not care about the absolute confidence level; it cares about *separability*. Rigidly
shifting the member distribution down, while the non-member distribution is largely where it was, can
even *hurt*: if the shift overshoots, members and non-members can end up about as separable as before,
just relocated, and the accuracy cost of the smoothing (members are now less confident on genuinely
correct answers too) shows up directly in the `test_acc` term of the composite. This is the wall I
expect to hit: a mean-shifting regularizer leaves the two distributions roughly as distinguishable as
ERM left them, only moved — so `mia_auc` may not drop much, and it may not drop at all net of the
accuracy I pay.

There is a subtler reason smoothing is the *right first* rung to try even if I expect it to underwhelm,
and it is about safety. Smoothing's gradient is benign everywhere: it is just cross-entropy against a
strictly-positive target, so it can never push the loss in a destabilizing direction — there is no
gradient ascent, no sign flip, no regime where the objective fights itself. That matters because the
CIFAR-100 setting is the one most likely to misbehave under a more aggressive defense, and I want to
know how much privacy a *perfectly stable* regularizer can buy before I reach for anything that
deliberately destabilizes training. Smoothing is the gentle, collapse-proof probe of "how much does
bounding confidence, with no variance lever, actually help here?"

This is also the reason I reach for smoothing *before* the other target-humbling option on the shelf.
The lineage offers two ways to bound confidence: edit the target (label smoothing) or subtract a penalty
on the output distribution (a confidence/entropy penalty). They differ in a way the CIFAR-100 number
makes concrete. Smoothing trains toward a strictly-positive target `q'`, so its gradient is ordinary
cross-entropy — always a descent direction, never a term that can push the loss *up*. A penalty that
subtracts something from the loss can, if it is strong enough or badly timed, overpower the fitting term
and drive the objective the wrong way. Now look at where CIFAR-100 already sits: test_acc 0.5045, barely
above half, on a 100-class problem the model is only just managing to fit at all. That is precisely the
benchmark where an objective that can fight its own fitting term is most likely to tip into failure. So
the disciplined order is to spend the *safe* regularizer first and measure how much a pure
confidence-cap buys with zero destabilization risk, and only then — knowing that baseline — decide
whether the extra adaptivity of a penalty is worth its collapse risk on the 0.5045 case. Running the
riskier method first would confound "the adaptivity helped" with "I got lucky on stability." Smoothing
first is the clean experiment.

One more design point, the value of `ε`. It is the single knob, the fraction of target mass moved to
uniform; it trades data-fitting against humility. Too small and the floor under confidence is too high
to dent the attack; too large and accuracy craters as the correct class loses its mass. The
well-established default for vision classification is `ε = 0.1`, and I have no task-specific reason to
move off it — the datasets are class-balanced, so uniform `u` is the right prior, and 0.1 is the value
that lifts accuracy as a regularizer on ImageNet-scale problems without flattening the output. I keep
`ε = 0.1` fixed across all three benchmarks; the loop already exposes `epoch` but smoothing has no
reason to vary by epoch, so `compute_loss` ignores it.

I do consider the one adaptation the `1/K` dilution seems to invite — raise `ε` on CIFAR-100 to
counteract the 10× thinner per-class grip — and reject it for concrete reasons. First, the compensating
`ε` would have to be large: to restore per-class mass `0.01` at `K = 100` I would need `ε = 1.0`, which is
the fully-uniform target that destroys the signal and the accuracy outright, and even reaching CIFAR-10's
grip only partway means pushing `ε` far enough to noticeably lower the true-class ceiling on a benchmark
already sitting at 0.5045 accuracy — spending accuracy I can ill afford on the fragile case. Second, and
more fundamentally, a bigger *uniform* push does not fix a *uniformity* problem: raising `ε` still applies
the same pressure to a memorized member and an uncertain one, so it would relocate the member mean further
without changing separability — more of the move I already expect to fail. The right response to the
dilution is not a bigger flat push but an *adaptive* one, which is a different method entirely. So I hold
`ε = 0.1` fixed and let this rung report the honest ceiling of what uniform target-humbling can do.

So the falsifiable expectations against the ERM numbers. I expect smoothing to *reduce member
confidence* — the privacy_gap should come down from ERM's levels — because it puts a hard floor under
how confident the model can be. But I expect the *attack AUC to move little*, because the shift is
roughly uniform across members and non-members and does not change their separability; on the
CIFAR-100 case where ERM hit 0.8677, I would be surprised to see smoothing pull it below the high-0.8s,
and the small accuracy cost of smoothing could leave the composite privacy_score *flat or slightly
worse* than ERM's 0.1368 rather than better. On resnet20-cifar10 I expect roughly ERM-level accuracy
(~0.79, maybe a touch lower) with an AUC that does not improve, so privacy_score near or just below
ERM's 0.5697. On mobilenetv2-fmnist, where leakage is already mild, smoothing has little to fix and a
small accuracy cost, so I expect privacy_score at or slightly below ERM's 0.8685. If those hold —
confidence floored but separability untouched — they are the precise evidence that I need a regularizer
that changes the *shape* of the member distribution, not just its location: something adaptive that
pushes hardest on the over-confident predictions. That is the next rung. The distilled scaffold fill is
in the answer.
