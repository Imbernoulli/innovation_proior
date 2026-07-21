I start where the scaffold starts, because the cheapest conceivable unlearning rule is also the right
thing to measure first: keep finetuning the trained model on the retained data and never show it the
forget set again. Before I add any active forgetting machinery, I want to know how far passive erosion
alone gets me, and exactly where it falls short — that failure is what every later rung will have to
out-do, so I need its number on the board before I spend design budget on something cleverer.

The setup forced on me is narrow. The harness has already pretrained `f(·; w)` on the full dataset for
80 epochs of SGD with cosine annealing (`lr=0.1`, momentum `0.9`, weight decay `5e-4`); class 0 is the
forget set; and now I get 20 epochs of an unlearning loop that hands me, on every step, one retain
minibatch `(retain_x, retain_y)` and one forget minibatch `(forget_x, forget_y)`, both already on the
device, plus a *fixed* `Adam(lr=0.001)` optimizer. I do not pick the optimizer, the learning rate, the
schedule, or the number of steps — I author only the body of `unlearn_step`, one parameter update
computed from those two minibatches. That narrowness decides which methods are even expressible before I
weigh which is best. Of the four background directions, three die on this edit surface, not on their
merits: retrain-from-scratch is the reference I am trying to avoid paying for, not something I can run;
SISA needs a sharded, per-shard-ensembled training architecture, and I am handed one monolithic trained
model; Fisher/NTK scrubbing wants a Hessian or Jacobian-outer-product solve, and nothing here gives me
second-order information or lets me swap the fixed first-order Adam for a closed form; amnesiac unlearning
needs a logged per-step training history that pretraining already threw away. What is left and expressible
is exactly the family that reads the two live minibatches and takes one Adam step — a retain-descent term,
optionally combined with some term on the forget minibatch. The scaffold picks the most conservative member
of that family, retain descent with the forget batch ignored (`forget_weight = 0.0` an unused placeholder),
and that is precisely the member I want as my floor.

There is a real temptation to skip this rung and open with an active forget term, since the forget minibatch
sits unused in the signature. A budget argument says measure the corner first. The unlearning loop is 20
epochs against pretraining's 80, at a nominal step size of `Adam(lr=0.001)` against pretraining's peak
`SGD(lr=0.1)` — two orders of magnitude smaller before Adam's normalization. Whatever this loop does, it
does with a small fraction of the optimization pressure that built the model. That cuts both ways: a passive
rule cannot do much damage, so it reports close to the best utility any method here can hold; and any active
term I add later spends from that same shallow well, so I had better know precisely how much forgetting is
left to buy and how much retain accuracy I risk to buy it. Measuring the passive corner calibrates both axes
at once. The forget class is also not a large slice — one of ten on CIFAR-10 and FashionMNIST, one of a
hundred on CIFAR-100 — so the retained set the passive rule reinforces is the overwhelming majority of the
data, another reason to expect it holds utility and forgets little.

The reason to expect preserved utility is direct. The retain cross-entropy is exactly the original training
objective restricted to `D_r`, run from weights that already minimize it, with a small Adam step. No force
in this loss pushes the model off the function it already computes on the retained classes — I am only
reinforcing what it already does, descending a loss whose minimum I am near. So `retain_acc` stays high, and
this matters beyond the rung: whatever passive finetuning preserves is roughly the *ceiling* on `retain_acc`
for the whole harness, because every active-forgetting term I add later can only ever *cost* retain
accuracy — it introduces a gradient that fights the retained representation — never add it. So this rung's
`retain_acc` is the utility budget the rest of the ladder spends against.

The reason to expect it to *fail at forgetting* is just as direct, but I have to be careful what "fail"
means, because two metrics measure two different things and one will mislead me. No term pushes against
`D_f`; the forget images never enter the forward pass, so they contribute nothing to loss or gradient. The
only erosion mechanism is indirect, catastrophic forgetting — the hope that reinforcing the other classes
drifts the shared representation until class-0 competence decays. But the shared lower-layer features that
recognize class 0 (edges, textures, shapes) are load-bearing for *every* retained class, so the retain
objective has every incentive to keep them sharp: passive finetuning actively *protects* the very features
that recognize class 0. Twenty small-step epochs are nowhere near enough drift to erase a class memorized
over 80.

And yet `forget_acc` — accuracy on held-out class-0 images — should fall, possibly to zero, and this is
*not* real forgetting. Retain cross-entropy never supplies class 0 as a positive target, so across 20 epochs
the class-0 logit is never reinforced while the retained-class logits are repeatedly pushed up, and softmax
normalization drags the unused class-0 logit down on essentially every input. The argmax on a class-0 image
migrates to whichever retained class is nearest in feature space, so `forget_acc` collapses toward zero
purely by *logit suppression of an unreinforced head row* — with no change to the features that encode the
class-0 concept. That is why `forget_acc` is the cheap, misleading axis: a model can score a perfect zero
while still carrying a fully extractable memory of the forget set underneath the suppressed row.

The axis not fooled is `forget_mia_auc`. The attack reads max-softmax confidence on forget-*train* images
(members, seen in the 80-epoch run) versus forget-*test* images (non-members, same class, never seen), and
reports the Mann-Whitney AUC — the probability a random member is scored more confidently than a random
non-member. A model that genuinely never trained on `D_f` has no systematic reason to be more confident on
train than test, so its AUC sits at ≈ 0.5. A model that memorized the specific forget-train images is
systematically more confident on them, pushing AUC above 0.5. The key point for this rung: logit suppression
shifts confidence *equally* on members and non-members — it pushes the whole class-0 head down on train and
test alike — so it barely moves the *gap*, which is the only thing the AUC reads. Passive finetuning leaves
the membership signal essentially intact; I expect `forget_mia_auc` stubbornly in the high-0.4-to-0.5 range,
marking a class still memorized underneath a suppressed row.

Now the score decomposition shows exactly what binds. The primary metric is
`unlearn_score = (retain_acc + (1 - forget_acc) + (1 - forget_mia_auc)) / 3`. If `forget_acc` collapses to
0 by suppression, the middle term saturates at ≈ 1 — a full point of "forgetting" credit banked for free
without any erasure. The residual memorization is charged entirely in `(1 - forget_mia_auc)`: with the AUC
stuck in the high 0.4s that term contributes a little over one half, and it cannot improve without a
mechanism that closes the train/test confidence gap — which this rule has none. So the score is
near-ceiling `retain_acc`, a saturated `(1 - forget_acc)` won cheaply, and a *capped* `(1 - forget_mia_auc)`.
The binding constraint is the MIA term, and the only headroom in the whole score is the gap between this
rung's `forget_mia_auc` and 0.5. That gap — the un-erased membership signal — is the exact quantity every
later rung has to attack, and its size is what this rung puts on the board alongside the retain ceiling.

I can anchor that against the one behaviour I can reason about exactly: the retrain-from-scratch reference I
am approximating. Such a model never saw class 0. It scores whatever the architecture-and-data ceiling
`R*` is on retained data; predicts some retained class on class-0 images so `forget_acc ≈ 0`; and, having
never trained on any class-0 image, is equally (un)confident on forget-train and forget-test, so its
`forget_mia_auc = 0.5` exactly. Its score is `(R* + 1 + 0.5) / 3`. Passive finetuning matches it term by
term on `retain_acc` and `(1 - forget_acc)` and loses only on the MIA term, by exactly the amount its AUC
exceeds 0.5. So the entire deficit of this rung relative to the gold standard is `(forget_mia_auc − 0.5)/3` —
the residual memorization made quantitative, a single number the ladder has to close.

I could soften the passivity now by nudging `forget_weight` above zero, but I deliberately will not: the
value of this rung is that it isolates one variable — what doing *nothing active* achieves — so the next
rung's active term can be attributed cleanly, and so the retain drop it costs can be read against a
*measured* ceiling rather than a remembered one. That is the tension the ladder opens with: real forgetting
demands an active term, and any active term strong enough to move the membership gap must move weights that
run through the shared trunk the retained classes depend on — spending against exactly the `retain_acc` this
rung establishes as worth protecting.

The three benchmarks will not report the same floor, which sharpens what comes next. resnet20-cifar10 is
easy — ten well-separated classes, a small residual net — so `R*` is high and the passive rule holds it
comfortably. vgg16bn-cifar100 packs 99 classes into a denser shared trunk, so absolute retain accuracy is
lower and, more importantly for later rungs, there is more cross-class structure for an active forget term
to damage — the fragile case to watch. mobilenetv2-fmnist is hidden, but its depthwise-separable trunk
factorizes spatial and channel mixing into separate lighter operators, so its feature sharing is
structurally different from the two dense convnets; FashionMNIST is an easy ten-way problem, so I expect a
high floor but flag that its unusual trunk could respond differently to pressure routed through it. For this
passive rung the upshot is uniform: near-ceiling retain, cheap argmax-forgetting, residual MIA — with the
differences standing as advance warning of where the active rungs will be tested.

What I watch on all three: `retain_acc` high and near the harness ceiling; `forget_acc` low, read
skeptically as logit suppression not erasure; `forget_mia_auc` stubbornly above 0.5, marking memorization
still to be undone; and `unlearn_score` middling — good utility, weak *real* forgetting, capped by the
membership term. The full scaffold module is in the answer.
