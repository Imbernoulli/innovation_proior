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

One more design point, the value of `ε`. It is the single knob, the fraction of target mass moved to
uniform; it trades data-fitting against humility. Too small and the floor under confidence is too high
to dent the attack; too large and accuracy craters as the correct class loses its mass. The
well-established default for vision classification is `ε = 0.1`, and I have no task-specific reason to
move off it — the datasets are class-balanced, so uniform `u` is the right prior, and 0.1 is the value
that lifts accuracy as a regularizer on ImageNet-scale problems without flattening the output. I keep
`ε = 0.1` fixed across all three benchmarks; the loop already exposes `epoch` but smoothing has no
reason to vary by epoch, so `compute_loss` ignores it.

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
