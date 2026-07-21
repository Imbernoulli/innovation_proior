Retain finetuning told me exactly what it would: it held utility and barely forgot. On resnet20-cifar10
`retain_acc` came in at 0.8758 with `forget_mia_auc` 0.4512; on vgg16bn-cifar100, 0.5345 and 0.4765; on the
hidden mobilenetv2-fmnist, 0.9373 and 0.4817. Read those MIA numbers against the only value that matters —
0.5, the AUC of a model that genuinely never saw `D_f`. They sit just *below* 0.5, the fingerprint I
predicted: the membership signal is essentially un-erased. The attack can barely separate forget-train from
forget-test confidences, but that is because the original model was already near that operating point on this
statistic, *not* because retain finetuning did any forgetting — it never touched class 0, so it could not
have moved the gap. And `forget_acc` on all three came in at exactly 0.0, confirming the other half of the
floor: the argmax on class-0 images migrated off by pure logit suppression, banking `(1 - forget_acc) ≈ 1`
for free without erasing anything the attacker can read.

Turn the decomposition into a constraint. On resnet20 the passive rule scored `(0.8758 + 1 + 0.5488)/3 =
0.8082`, exactly the reported number, so I trust it. The two forget-axis terms already contribute
`1 + 0.5488 = 1.5488` out of a possible 2.0 — the `(1 - forget_acc)` term is maxed and cannot improve, so
the *only* headroom left anywhere in the score on this benchmark is `2.0 − 1.5488 = 0.4512`, precisely the
membership slack. The same arithmetic on the others: vgg16bn leaves `0.4765` of forget-axis headroom, fmnist
`0.4817`. So the entire remaining score, on every benchmark, lives in closing the membership gap while
utility is already parked near its ceiling — which is exactly why the next move must be an *active* forgetting
pressure aimed at that gap.

That same decomposition warns me, numerically, of the trap the obvious active move walks into. Any active
term can at best drive the two forget terms to 2.0 (perfect argmax-forgetting *and* MIA driven to zero), and
it can only do so by moving weights, and moving weights through the shared trunk costs `retain_acc`. So the
active move beats passive only if the forget-axis *gain* exceeds the retain *loss*. On resnet20 the maximum
available forget gain is `0.4512`, so it wins only if `retain_acc` stays above `0.8758 − 0.4512 = 0.4246`.
On vgg16bn the gain is `0.4765` but passive retain was only `0.5345`, so the break-even floor is
`0.5345 − 0.4765 = 0.0580` — a shockingly low bar, meaning almost any retain that is not a near-total wipe
still wins on vgg. On fmnist break-even is `0.9373 − 0.4817 = 0.4556`. These three numbers — `0.4246`,
`0.0580`, `0.4556` — are the falsifiable thresholds I am predicting against: if the ascent I am about to add
crashes `retain_acc` below them, the score sinks *below the passive floor* despite perfect forgetting, and
the crash is the lesson.

Passive erosion failed because the shared trunk it reinforces is the very thing that recognizes class 0 —
the same coupling that will make the active fix dangerous. The textbook active move is gradient *ascent* on
the forget loss: where descent makes the model more right, I negate the gradient on `D_f` and make it more
wrong, climbing the cross-entropy to drive the model off the class-0 answer it memorized. That directly
attacks what retain finetuning left untouched, and unlike logit suppression it moves the *members*
specifically, because the ascent gradient is largest on exactly the high-confidence memorized training
images, so it should finally push on the train/test confidence gap the MIA reads. But I cannot ascend on
`D_f` alone: `D_f` and `D_r` are not processed by disjoint weights, and pushing the model wrong on class 0
sends gradients back through the shared trunk and perturbs the retained classes. So the active term needs a
defensive descent on the retain set. That is the NegGrad+ shape: a single combined loss
`L = retain_loss − β · forget_loss`, `β` trading forgetting ascent against retain descent, here fixed at
`forget_weight = 0.5` (a moderate balance — a large `β` would let the ascent dominate the shared trunk from
epoch one and crash retain immediately, a tiny `β` would move the memorized confidences too little over 20
epochs to matter), one backward/step on the summed loss.

Now the danger in this term, which is what will set the rung's number. Cross-entropy to the true label is
*unbounded above*: it is `−log p` in the true-class probability, and as `p` approaches zero `−log p` runs to
infinity, so the ascent term has no fixed point — it keeps demanding more probability mass off class 0,
forever. A confidently memorized class-0 image starts near `p ≈ 1`, so `forget_loss ≈ 0` and the ascent
gradient is small; drive it to `p = 0.1` and the loss is `≈ 2.3`, to `0.01` it is `≈ 4.6`, to `0.001` it is
`≈ 6.9`. Worse, `∂(−log p)/∂p = −1/p` *blows up* as `p → 0`, so the ascent gradient gets *stronger* the more
the model has already forgotten — positive feedback with no brake. The retain descent term *is* bounded
(cross-entropy floors at 0), so the two are asymmetric: the bounded retain term saturates near its minimum
and stops producing large gradient while the unbounded forget-ascent term keeps producing more. The ascent
therefore wins the late dynamics — `β = 0.5` slows this but cannot change the asymptotics; there is no `β`
that turns an unbounded ascent into a bounded one, it only rescales how fast the weights run off. And they
will run off: the endpoint of an unbounded ascent is diverging weights, a torn representation, and
*retained* accuracy collapsing as collateral, because the features the ascent corrupts are the shared
features the retain classes depend on.

This failure is the opposite of the previous one. Retain finetuning failed softly — high utility, weak
forgetting. NegGrad will fail hard: it will absolutely forget — `forget_acc → 0` robustly and the MIA AUC
may even drop *below* 0.5 — but pay by wrecking `retain_acc` well below the 0.8758 / 0.5345 / 0.9373 ceiling
and below the `0.4246 / 0.0580 / 0.4556` break-even floors, so `unlearn_score` comes in below the passive
baseline despite the perfect forgetting. vgg16bn packs 99 retained classes into the densest shared trunk, so
it has the most structure for the ascent to corrupt and should crash hardest in absolute terms — yet its
break-even was only `0.0580`, so paradoxically it is the one benchmark where even a catastrophic crash could
clear its tiny bar; the resnet20 and fmnist thresholds near `0.43` and `0.46` are far more demanding and
should fall decisively. The one certainty on all three: a shared-trunk architecture cannot absorb an
unbounded ascent without the retained classes paying for it.

There is a subtler problem specific to the *privacy* axis, and it is the gap the next rung must close.
Driving `forget_acc` to exactly zero — making the model *confidently wrong* on class 0 — is not what
forgetting should look like. A model that genuinely never trained on class 0, shown a class-0 image, does
not confidently shout some other class; it sits at generalization-level uncertainty, spreading probability
as its ignorance warrants, with a train/test confidence gap of zero, i.e. MIA AUC = 0.5. A model taught a
sharp anti-fact — always predict something-other-than-0, with confidence — has not forgotten; it has learned
a new inverted competence on exactly those inputs, and its MIA AUC can swing *below* 0.5. The score design is
perverse here: an AUC below 0.5 makes `(1 − forget_mia_auc)` exceed 0.5, so *confident wrongness is rewarded
more than genuine forgetting* — even though it is strictly worse privacy, because an attacker who notices the
model is weirdly confident-wrong on precisely these inputs has learned they were specially scrubbed. So
"maximal forgetting" via unbounded ascent overshoots into a regime the score flatters and the threat model
punishes. NegGrad has no notion of *how much* to forget — hard-label cross-entropy ascent has no
"stop at generalization-level uncertainty" fixed point. The right target is to forget only as much as a
model that never saw `D_f` would, landing the MIA at 0.5, and NegGrad cannot express it.

I situate this in the harness: one retain minibatch and one forget minibatch per step, both labeled, both on
device, the fixed Adam. So `unlearn_step` forwards on `retain_x` and `forget_x`, takes cross-entropy on each,
forms `retain_loss − forget_weight · forget_loss`, and does a single `zero_grad / backward / step`. No
separate forget optimizer, no per-set schedule, no checkpointing — the only hyperparameter I own is
`forget_weight = 0.5`. I report `retain_loss` and `forget_loss` alongside `loss` so the dynamics are visible:
`forget_loss` should climb without bound across the 20 epochs (the tell the ascent never settles) and
`retain_loss` should creep up as the shared trunk degrades.

One mechanism refines what "the weights run off" will look like numerically. The harness pins me to
`Adam(lr=0.001)`, and Adam divides each coordinate's update by a running RMS of its own gradient, so the
*magnitude* of every step is capped near `lr` regardless of how large the raw forget gradient grows. The
divergence will therefore not present as a numeric explosion; it presents as the forget-ascent coordinate
taking a *sustained, full-size* step in the same corrupting direction epoch after epoch, because its
normalized gradient keeps pointing "make class 0 more wrong" and never shrinks the way a saturated retain
coordinate's does. Over 20 epochs of consistently-signed max-size steps on the shared trunk, the retained
representation is walked steadily off its minimum — a slow-motion collapse rather than a blow-up, but a
collapse all the same. This is why the small fixed learning rate does not save NegGrad: Adam converts the
unbounded gradient into a bounded-magnitude but relentlessly-directed drift.

So the sharpened expectations: `forget_acc` at or to zero on all three — NegGrad does forget, hard — but
`retain_acc` crashed below the previous rung's ceiling and below the break-even floors, so `unlearn_score`
lands below retain finetuning's 0.8082 / 0.6860 / 0.8185, and at least one benchmark's MIA dips below 0.5,
the confidently-wrong overshoot. If that is what I see — total forgetting bought at the cost of a wrecked
model — then the two failures together write the next problem: the forget pressure needs a *place to stop*.
An objective with a fixed point it can reach, rather than an ascent demanding ever more, is the only kind
that could push the forget set to generalization-level uncertainty and settle there at MIA = 0.5 instead of
running the weights off and overshooting into the conspicuous confidently-wrong regime. What that bounded
objective is I leave open; this rung establishes that the unbounded one is disqualified. The full scaffold
module is in the answer.
