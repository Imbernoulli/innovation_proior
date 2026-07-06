I start where the scaffold starts, because the cheapest conceivable unlearning rule is also the right
thing to measure first: keep finetuning the trained model on the retained data and never show it the
forget set again. Before I add any active forgetting machinery, I want to know how far passive erosion
alone gets me, and exactly where it falls short — that failure is what every later rung will have to
out-do, so I need its number on the board before I spend any design budget on something cleverer.

The setup forced on me is specific, and whatever rule I write has to live entirely inside it. The
harness has already pretrained `f(·; w)` on the full dataset for 80 epochs of SGD with cosine annealing
(`lr=0.1`, momentum `0.9`, weight decay `5e-4`); class 0 is the forget set; and now I get 20 epochs of
an unlearning loop that hands me, on every step, one retain minibatch `(retain_x, retain_y)` and one
forget minibatch `(forget_x, forget_y)`, both already on the device, plus a *fixed* `Adam(lr=0.001)`
optimizer. I do not get to pick the optimizer, the learning rate, the schedule, or the number of steps —
the only thing I author is the body of `unlearn_step`, one parameter update computed from those two
minibatches. So the design space here is genuinely just "what update do I compute from these two
minibatches," and I should be honest that this narrowness is doing a lot of work: it decides which
methods are even expressible before I weigh which is best.

Let me actually walk the space, because the background hands me four prior-art directions and I want to
see which survive the edit surface rather than assume. Retraining from scratch on `D_r` is the reference
behaviour I am trying to approximate cheaply — it is explicitly *not* a method I can run here, it is the
gold standard I am trying to avoid paying for, so it is out by definition. SISA shards the training set
and retrains only the shard holding the deleted point — but sharding and per-shard ensembling is a
*training-time* architecture decision; I am handed a single already-trained monolithic model and a
per-step update hook, with no shards to retrain and no ensemble to edit, so SISA is structurally
unavailable at this interface. Fisher/NTK scrubbing wants a closed-form weight update sized by the
Fisher information or an NTK linearization — that needs a Hessian or a Jacobian-outer-product estimate
over the data, and nothing in `unlearn_step` gives me second-order information or lets me swap the fixed
first-order Adam for a closed-form solve; I could in principle estimate a diagonal Fisher from squared
gradients, but that is a large detour to take before I even know the passive floor. Amnesiac unlearning
subtracts the exact parameter updates that touched `D_f` during the original run — but that presupposes
a *logged training history*, every minibatch update recorded, and I have no such log: pretraining
already happened and threw its per-step updates away. So three of the four background methods are ruled
out not because they are bad ideas but because the frozen substrate does not expose the state they need.
What is left, and expressible, is exactly the family of rules that read the two live minibatches and
take one Adam step: a retain-descent term, optionally combined with *some* term computed on the forget
minibatch. The scaffold picks the most conservative member of that family — retain descent alone, forget
batch ignored — and that is precisely the member I want as my floor. The `forget_weight = 0.0` field in
the scaffold is a placeholder advertising that the forget batch *could* be weighted in; the default
leaves it at zero.

There is a real temptation to skip this rung and open directly with an active forgetting term, since I
can already see the forget minibatch sitting unused in the signature. I should resist that, and the
reason is a budget argument I can make on paper. The unlearning loop is 20 epochs against the pretraining
run's 80 — a quarter of the epochs — and the nominal step size is `Adam(lr=0.001)` against pretraining's
peak `SGD(lr=0.1)`, two orders of magnitude smaller in nominal terms even before accounting for Adam's
normalization. Whatever this loop does, it does with a small fraction of the optimization pressure that
built the model. That cuts both ways and it is exactly why the floor matters: with so little budget, a
passive rule cannot do much *damage*, so it will report close to the best utility any method in this
harness can hold; and symmetrically, that same small budget means any *active* term I add later is
spending from a shallow well, so I had better know precisely how much forgetting is actually left to buy
and how much retain accuracy I am risking to buy it. Measuring the passive corner first calibrates both
axes at once. And the forget class is not a negligible slice — it is one of ten classes on the CIFAR-10
and FashionMNIST benchmarks (roughly ten percent of the data) and one of a hundred on CIFAR-100 (about
one percent) — so the retained set the passive rule reinforces is the overwhelming majority of the data
on every benchmark, which is another reason to expect it to hold utility well and forget little.

The reason to expect this to preserve utility is direct. The retain cross-entropy is exactly the
original training objective restricted to `D_r`, run from weights that already minimize it well, with a
small Adam step size. There is no force in this loss that pushes the model off the function it already
computes on the retained classes — I am only reinforcing what it already does, descending a loss whose
minimum I am already near. So `retain_acc` should stay high; the model keeps being good at the classes
it is allowed to remember. This matters beyond this rung: whatever passive finetuning preserves is
roughly the *ceiling* on `retain_acc` for the whole harness, because every active-forgetting term I add
in later rungs will only ever *cost* retain accuracy — it introduces a gradient that fights the retained
representation — never add it. So this rung's `retain_acc` is not just a data point; it is the utility
budget the rest of the ladder is spending against.

The reason to expect this to *fail at forgetting* is just as direct, and it is the crux, but I have to
be careful about what "fail at forgetting" even means here, because two different metrics measure two
different things and one of them will mislead me. There is no term anywhere in this objective that
pushes *against* `D_f`; the forget images never enter the forward pass, so they contribute nothing to
the loss and nothing to the gradient. The only mechanism that could erode class-0 knowledge is indirect,
catastrophic forgetting: the hope that reinforcing the other classes long enough drifts the shared
representation until class-0 competence decays on its own. But the shared lower-layer features that
recognize class 0 — the edges, textures, shapes — are load-bearing for *every* retained class too, so
the retain objective has every incentive to keep them sharp, which means passive finetuning actively
*protects* the very features that still recognize class 0. Twenty small-step epochs are nowhere near
enough drift to erase a class the model spent 80 epochs memorizing.

And yet I should expect `forget_acc` — accuracy on held-out class-0 images — to fall, possibly all the
way toward zero, and I need to be clear this is *not* real forgetting, because that distinction is the
whole point of the metric design. Cross-entropy on the retain classes is a softmax objective: it never
supplies class 0 as a positive target, so across the 20 epochs the class-0 output logit is never
reinforced while the retained-class logits are repeatedly pushed up, and softmax normalization drags the
unused class-0 logit down relative to them on essentially every input. The predicted argmax on a
held-out class-0 image therefore migrates off class 0 to whichever retained class is nearest in feature
space — so `forget_acc` can collapse toward zero purely by *logit suppression of an unreinforced head
row*, with no change whatsoever to the features that encode the class-0 concept. That is why `forget_acc`
is the cheap, misleading axis: a model can score a perfect zero on it while still carrying a full,
extractable memory of the forget set underneath the suppressed output row.

The axis that is not fooled is `forget_mia_auc`, and it is worth being precise about the mechanics
because it is what actually caps the score. The membership-inference attack takes the forget-*train*
images (members — the specific images seen during the original 80-epoch run) and the forget-*test*
images (non-members — same class, never seen), reads the model's max-softmax confidence on each, and
reports the Mann-Whitney AUC of the two confidence distributions. That AUC is exactly the probability
that a random member is scored more confidently than a random non-member. A model that genuinely never
trained on `D_f` has no systematic reason to be more confident on the train images than the test images,
so its AUC sits at ≈ 0.5 — the never-saw-it operating point. A model that *memorized* the specific
forget-train images is systematically more confident on them than on unseen class-0 images, pushing the
AUC above 0.5. Now the key observation for this rung: logit suppression shifts confidence *equally* on
members and non-members — it pushes the whole class-0 head down on train and test images alike — so it
barely moves the *gap* between the two, which is the only thing the AUC reads. Passive finetuning
therefore leaves the membership signal essentially intact; I expect `forget_mia_auc` to sit stubbornly
in the high-0.4-to-0.5 range, marking a class that is still memorized underneath a suppressed output row.

Now I can decompose the score on paper and see exactly what binds it. The primary metric is
`unlearn_score = (retain_acc + (1 - forget_acc) + (1 - forget_mia_auc)) / 3`. If `forget_acc` collapses
toward 0 by logit suppression, the middle term saturates at ≈ 1 — a full point of "forgetting" credit
that this rung banks essentially for free, without any real erasure. The forget axis' remaining term,
`(1 - forget_mia_auc)`, is where the residual memorization is actually charged: with the AUC stuck near,
say, the high 0.4s, that term contributes only a little over one half, and it cannot be improved without
a mechanism that closes the train/test confidence gap — which this rule has none. So the score is a sum
of a near-ceiling `retain_acc`, a saturated `(1 - forget_acc)` won cheaply, and a *capped*
`(1 - forget_mia_auc)`. The binding constraint is plainly the MIA term: utility is near the harness
ceiling and the argmax-forgetting credit is already maxed, so the only headroom left in the whole score
is the gap between this rung's `forget_mia_auc` and 0.5. That gap — the un-erased membership signal — is
the exact quantity every later rung has to attack, and its size is the second thing this rung puts on the
board alongside the retain ceiling.

Let me sanity-check that decomposition against the one behaviour I can reason about exactly — the
reference I am approximating, a model retrained from scratch on `D_r`. Such a model never saw class 0
at all. On the retained test set it scores whatever the architecture-and-data ceiling is — call it
`R*`. On class-0 test images it predicts some retained class, so its `forget_acc` is ≈ 0 and
`(1 - forget_acc)` ≈ 1. And critically, having never trained on any class-0 image, it is equally
(un)confident on forget-train and forget-test, so its `forget_mia_auc` = 0.5 exactly and
`(1 - forget_mia_auc)` = 0.5. The reference score is therefore `(R* + 1 + 0.5) / 3`. Now line up
passive finetuning against that reference term by term: it matches on `retain_acc` (both ≈ `R*`), it
matches on `(1 - forget_acc)` (both ≈ 1, though mine gets there by logit suppression and the reference
by genuine ignorance), and it *loses only on the MIA term*, `(1 - forget_mia_auc)` < 0.5 by exactly the
amount its AUC exceeds 0.5. So the entire deficit of this rung relative to the gold standard is
`(forget_mia_auc − 0.5) / 3`, a single number, and it is the residual memorization made quantitative.
This is a clean falsifiable prediction: passive finetuning should land essentially *on* the retrain
reference in retain accuracy and argmax-forgetting, and fall short of it only, and by exactly, the
membership gap divided by three.

It is also worth tracing one `unlearn_step` concretely to confirm the rule does what I claim and nothing
more. A retain minibatch arrives as `(retain_x, retain_y)` with `retain_x` shaped `128×3×32×32` and
`retain_y` a length-128 vector of labels drawn only from the retained classes; I forward it to logits
`128×C` (C = 10, 100, or 10 across the benchmarks), take `F.cross_entropy` to a scalar, `zero_grad`,
`backward`, `step`. The forget minibatch is passed in but never read, so no gradient ever flows from a
class-0 image — which is the whole definition of "passive." The returned dict carries `"loss"` as the
contract requires. There is nothing in this trace that could raise `retain_acc` above `R*` or move the
train/test confidence gap on class 0; the update is exactly a small reinforcement step on the retained
classes, which is precisely why the prediction above holds.

I could soften the passivity right now by nudging `forget_weight` above zero and folding in even a tiny
forget term, but I deliberately will not, and the reason is both disciplinary and diagnostic. The whole
value of this rung is that it isolates a single variable — "what does doing *nothing active* achieve" —
so that when the next rung adds an active term I can attribute the entire change in every metric to that
term and to nothing else. If I mixed in a partial forget weight here, I would blur the floor and lose
the clean attribution. There is also a forward-looking reason to bank this passive number cleanly and by
itself: whatever active forgetting term the next rung introduces, it can only push against the forget set
by moving weights, and every weight it moves sits in a trunk shared with the retained classes, so its
cost will show up as a *drop* in `retain_acc` from wherever passive finetuning leaves it. To read that
drop honestly I need the undropped level measured, not estimated — the difference between "retain fell to
X" and "retain fell by Y from a known ceiling" is the difference between a guess and an attribution. So
pinning this passive corner down first, on both the utility axis and the membership axis, is what will let
me price every later active step against a real baseline rather than a remembered one.

So this rung is deliberately the floor for the *forgetting* axis while being strong on the *utility*
axis, and it hands the ladder two calibrated numbers: the achievable `retain_acc` ceiling that later
active methods will erode from, and the size of the membership signal — the distance from
`forget_mia_auc` to 0.5 — that still has to be killed. What this rung conspicuously does *not* supply is
any *active* forgetting pressure — a term that genuinely moves the train/test confidence gap rather than
merely suppressing an unreinforced output row — and closing that gap is the problem I am leaving for the
next step to take up. I go in with one worry already visible from the geometry of the score: any pressure
strong enough to move the membership gap must do so by moving weights, and those weights run through the
shared trunk the retained classes depend on, so whatever I add later will be spending against exactly the
`retain_acc` this rung is establishing as the thing worth protecting. That is the tension the ladder
opens with — real forgetting demands an active term, and an active term threatens the shared trunk — and I
need this passive floor measured, on both axes, before I can even size the trade.

The three benchmarks will not report the same floor, and thinking through why sharpens what I am
measuring. On resnet20-cifar10 the task is easy — ten well-separated classes, a small residual network —
so `R*` should be high and the passive rule should hold it comfortably; this is the benchmark where I
expect the cleanest "high retain, residual MIA" floor. On vgg16bn-cifar100 the retained head carries 99
classes packed into the same trunk, so the *absolute* retain accuracy is inherently lower (a hundred-way
problem is harder than a ten-way one) and, more importantly for later rungs, the shared trunk is denser
with cross-class structure — many classes leaning on the same features — which means this is the
benchmark where an active forget term will have the most collateral to damage. The passive rule does no
damage, so its floor here just reports the honest hundred-class ceiling, but I am already noting vgg as
the fragile case for what comes next. The hidden mobilenetv2-fmnist benchmark I cannot see, but its
architecture is depthwise-separable — the trunk factorizes spatial and channel mixing into separate,
lighter operators — which makes its feature sharing structurally different from the two dense convnets,
and FashionMNIST is an easy ten-way problem, so I expect a high passive floor there too but I flag that
its unusual trunk could respond differently to any pressure I later route through it. For this passive
rung the practical upshot is uniform across all three: near-ceiling retain, cheap argmax-forgetting,
residual MIA — but the differences matter as advance warning of where the active rungs will be tested.

What I watch on all three benchmarks: `retain_acc` high and roughly at the harness ceiling;
`forget_acc` low, and I will read it skeptically as logit suppression rather than erasure;
`forget_mia_auc` stubbornly above 0.5, marking the memorization still to be undone; and `unlearn_score`
middling — good utility, weak *real* forgetting, capped by the membership term. The full scaffold module
is in the answer.
