Retain finetuning told me exactly what it would: it held utility and barely forgot. On
resnet20-cifar10 `retain_acc` came in at 0.8758 with `forget_mia_auc` at 0.4512; on vgg16bn-cifar100,
`retain_acc` 0.5345 and `forget_mia_auc` 0.4765; on the hidden mobilenetv2-fmnist, `retain_acc` 0.9373
and `forget_mia_auc` 0.4817. Read those MIA numbers against the only value that matters for forgetting —
0.5, the AUC of a model that genuinely never saw `D_f`. They sit just *below* 0.5, which is the
fingerprint of the thing I predicted: the membership signal is essentially un-erased. The attack can
barely separate forget-train from forget-test confidences, but that is because the original model was
already near that operating point on this confidence statistic, *not* because retain finetuning did any
forgetting — it never touched class 0, so it could not have moved that gap on purpose. And the
`forget_acc` on all three came in at exactly 0.0, which confirms the other half of my floor prediction:
the argmax on class-0 images did migrate off class 0 by pure logit suppression, banking the
`(1 - forget_acc) ≈ 1` credit for free without erasing anything the attacker can read.

Let me turn the score decomposition into an actual constraint, because it dictates the next move
quantitatively rather than by intuition. The primary metric is
`unlearn_score = (retain_acc + (1 - forget_acc) + (1 - forget_mia_auc)) / 3`. On resnet20 the passive
rule scored `(0.8758 + 1 + 0.5488)/3 = 2.4246/3 = 0.8082`, exactly the reported number, so I trust the
decomposition. The two forget-axis terms already contribute `1 + 0.5488 = 1.5488` out of a possible
`2.0` — the `(1 - forget_acc)` term is maxed and cannot improve, so the *only* headroom left anywhere in
the score on this benchmark is `2.0 − 1.5488 = 0.4512`, precisely the `(forget_mia_auc − 0)` slack in the
membership term. Repeating the arithmetic on the other two: vgg16bn banks `1 + (1 − 0.4765) = 1.5235`,
leaving `0.4765` of forget-axis headroom; fmnist banks `1 + (1 − 0.4817) = 1.5183`, leaving `0.4817`. So
the diagnosis is not just "forgetting is weak" — it is that the entire remaining score, on every
benchmark, lives in closing the membership gap, and utility is already parked near its ceiling. That is
the clean reason the next move must be an *active* forgetting pressure aimed straight at that gap.

But that same decomposition also warns me, before I write a line, of the trap the obvious active move
walks into — and I can make the warning numerical. Any active term I add can, at best, drive the two
forget terms to their maximum `2.0` (perfect argmax-forgetting *and* MIA driven to zero). It can only do
so by moving weights, and moving weights through the shared trunk costs `retain_acc`. So the active move
beats the passive baseline only if the forget-axis *gain* exceeds the retain *loss*. On resnet20 the
maximum available forget gain is `0.4512`, so NegGrad beats passive only if `retain_acc` stays above
`0.8758 − 0.4512 = 0.4246`. On vgg16bn the available gain is `0.4765` but the passive retain was only
`0.5345`, so the break-even retain floor is `0.5345 − 0.4765 = 0.0580` — a shockingly low bar, meaning on
vgg *almost any* retain that is not a near-total wipe would still win. On fmnist the break-even is
`0.9373 − 0.4817 = 0.4556`. These three numbers — `0.4246`, `0.0580`, `0.4556` — are the falsifiable
thresholds I am really predicting against: if the unbounded ascent I am about to add crashes `retain_acc`
below them, the score sinks *below the passive floor* despite perfect forgetting, and the crash is the
whole lesson.

It is worth dwelling on *why* passive erosion failed so cleanly, because the reason is the same shared
representation that will make the active fix dangerous. The retain finetuning step only ever descended the
cross-entropy on the retained classes, starting from weights already at a good minimum of that loss. The
gradient it computed had no component that asked the model to change anything about class 0 — the forget
images never entered the forward pass, so they contributed nothing to the loss and nothing to the
gradient. The only way class-0 competence could have decayed is *indirectly*, if reinforcing the retained
classes happened to overwrite the features class 0 depends on. But those features are not class-0-specific:
the early convolutional layers compute edges, textures, shapes that *every* class uses, and retain
finetuning has every incentive to keep them sharp because the retained classes need them too. So the
shared trunk, which is what makes unlearning hard, is exactly what made passive forgetting impossible:
the retain objective actively *protects* the very features that still recognize class 0. That is why the
MIA AUC barely moved off the original model's operating point — nothing pushed it.

The textbook active move is gradient *ascent* on the forget loss. Where descent on a minibatch makes the
model more right, I negate the gradient on `D_f` and make it more wrong: climb the cross-entropy on the
forget examples, drive the model off the class-0 answer it memorized. That directly attacks the thing
retain finetuning left untouched — and, crucially, unlike logit suppression it moves the *members*
specifically, because the ascent gradient is largest on exactly the high-confidence memorized training
images, so it should finally push on the train/test confidence gap the MIA reads. And I cannot do ascent
on `D_f` alone, for the same reason the prior rung's failure was structural: `D_f` and `D_r` are not
processed by disjoint weights. The lower convolutional layers compute features used by every class;
pushing the model to be wrong on class 0 sends gradients back through that shared trunk and perturbs the
retained classes too. So the active forget term has to be paired with a defensive descent term on the
retain set that holds the line. That is the NegGrad+ shape: descend the retain cross-entropy while
ascending the forget cross-entropy, a single combined loss `L = retain_loss − β · forget_loss`, with `β`
trading the strength of the forgetting ascent against the retain descent. In this harness `β` is
`forget_weight = 0.5`, and the whole thing is one backward / step on the summed loss — not two separate
updates.

Before I accept that shape I should check the two obvious alternatives to it, because "combined loss with
β = 0.5" is a choice and I want to have rejected the competitors on paper. The first alternative is to
make `β` large — say weight the forget ascent several times the retain descent — to force fast, decisive
forgetting. Walk it a couple of steps: a large `β` makes the ascent gradient dominate the summed gradient
on the shared trunk from the very first epoch, so the retain descent cannot even locally hold, and the
retained accuracy starts falling immediately; worse, because the ascent term is unbounded (below) there is
no `β` at which it stops demanding more, so a large `β` just reaches the divergence faster. That is
strictly worse than a moderate `β`, so it is out. The second alternative is a tiny `β` — a whisper of
ascent — hoping to nudge the membership gap without disturbing utility. But the budget arithmetic kills
it: I have only 20 small-step epochs, and a tiny ascent coefficient means a tiny forget gradient, so over
the whole run it would move the memorized confidences barely at all and the MIA would stay near the
passive floor — I would spend the rung and land back where retain finetuning already is. So neither
extreme is right; `β = 0.5`, the standard NegGrad+ balance, is the moderate setting that lets the ascent
actually bite while giving the retain term a fighting chance — and the point of measuring it is precisely
to see *whether* that fighting chance is enough, which the boundedness analysis below says it is not.

Now I have to be honest about the danger in this term, and the danger is exactly what will set this
rung's number. Cross-entropy to the true label is *unbounded above*: it is `−log p` in the true-class
probability `p`, and as the model's predicted probability on the true class 0 approaches zero, `−log p`
runs to infinity, so the ascent term has no fixed point to settle at — it keeps demanding the model push
more probability mass off class 0, forever. Let me put one number on it to feel the asymmetry: a
confidently memorized class-0 image starts near `p ≈ 1`, so `forget_loss = −log p ≈ 0` and the ascent
gradient is small; drive it to `p = 0.1` and the loss is `−log 0.1 ≈ 2.3`; to `p = 0.01`, `≈ 4.6`; to
`p = 0.001`, `≈ 6.9`. The loss keeps growing and, more to the point, `∂(−log p)/∂p = −1/p` *blows up* as
`p → 0`, so the ascent gradient gets *stronger* the more the model has already forgotten — a positive
feedback with no brake. The retain descent term *is* bounded (cross-entropy is minimized at the correct
answer with a finite floor of 0), so as training proceeds the two terms are not symmetric: the bounded
retain term saturates near its minimum and stops producing large gradients, while the unbounded
forget-ascent term keeps producing ever-larger gradient. The ascent term therefore *wins the
late-training dynamics*. With `β = 0.5` the forget term is half-weighted, which slows this but does not
change the asymptotics: there is no β that turns an unbounded ascent into a bounded one; β only rescales
how fast the weights run off. And they will run off — the natural endpoint of an unbounded ascent
objective is the weights diverging, the representation getting torn apart, and the *retained* accuracy
collapsing as collateral, because the very features the ascent term is corrupting are the shared features
the retain classes depend on.

Let me predict the shape of that collapse concretely, because it is what distinguishes this rung's
failure from the previous one, and tie it back to the break-even numbers. Retain finetuning failed
*softly* — high utility, weak forgetting. NegGrad will fail *hard* in the opposite direction: it will
absolutely forget — `forget_acc` will go to zero (it already was zero, but now for the right reason, and
robustly) and the MIA AUC may even drop *below* 0.5 as the model becomes confidently, abnormally wrong on
class 0 — but it will pay for that by wrecking `retain_acc`. So I expect the diagnostic signature to be a
*crashed* `retain_acc`, dramatically below the 0.8758 / 0.5345 / 0.9373 the previous rung established as
the achievable ceiling, and below the `0.4246 / 0.0580 / 0.4556` break-even thresholds I computed, so
that `unlearn_score` comes in below the passive baseline on every benchmark despite the perfect
forgetting. There is a subtlety in *where* it crashes worst. vgg16bn packs 99 retained classes into a
dense shared trunk, so it has the most cross-class structure for the ascent to corrupt and should crash
the hardest in absolute terms — plausibly to a near-total head wipe. But note its break-even was only
`0.0580`, so paradoxically vgg is the one benchmark where even a catastrophic crash could in principle
still clear the (tiny) bar; the resnet20 and fmnist thresholds, near `0.43` and `0.46`, are far more
demanding, so those two should fall below break-even decisively. The one thing I am confident of on all
three is that a shared-trunk architecture cannot absorb an unbounded ascent without the retained classes
paying for it.

There is a second, subtler problem with this term that matters for the *privacy* axis specifically, and
it is the conceptual gap the next rung will have to close. Driving `forget_acc` to exactly zero — making
the model *confidently wrong* on class 0 — is not what forgetting should look like. Picture a model that
genuinely never trained on class 0: shown a class-0 image, it does not confidently shout some other
specific class; it sits at generalization-level uncertainty, spreading probability roughly the way its
ignorance warrants, with a train/test confidence gap of zero, i.e. MIA AUC = 0.5. A model that has been
*taught a sharp anti-fact* about class 0 — always predict something-other-than-0, with confidence — has
not forgotten; it has learned a new, sharp, *inverted* competence on exactly those inputs, and its MIA
AUC can swing *below* 0.5. Here the score design is perverse and worth naming: an AUC below 0.5 makes the
`(1 − forget_mia_auc)` term exceed 0.5, so *confident wrongness is rewarded in the score more than
genuine forgetting is* — even though it is strictly worse privacy, because a membership attacker who
notices the model is weirdly, confidently wrong on precisely these inputs has learned they were specially
scrubbed. So "maximal forgetting" via unbounded ascent overshoots into the confidently-wrong regime,
which the score flatters and the threat model punishes. NegGrad has no notion of *how much* to forget —
the unbounded term simply forgets as hard as the optimizer allows before it diverges. The right target is
to forget *only as much as a model that never saw `D_f` would* — land the MIA at 0.5, not past it — and
NegGrad cannot express that target, because hard-label cross-entropy ascent has no "stop at
generalization-level uncertainty" fixed point.

Let me verify the "ascent wins the late dynamics" claim on a single shared weight rather than assert it,
because it is the load-bearing step of the whole prediction. Take one trunk parameter `θ` that both a
retained class and class 0 use. The combined gradient it feels is
`∂L/∂θ = ∂retain_loss/∂θ − 0.5·∂forget_loss/∂θ`. Early on, the model is right on both sets, so
`retain_loss` is near its floor and `∂retain_loss/∂θ ≈ 0`, while the forget images are still confidently
class 0, so `forget_loss = −log p` with `p ≈ 1` is also near zero and its gradient is small too — the
step is gentle. Now run it forward a few epochs. The retain term, being a descent toward a minimum it
starts near, stays small: `∂retain_loss/∂θ` cannot grow without the retained accuracy first getting
*worse*, and the term's job is to prevent exactly that, so it self-limits. The forget term does the
opposite: every step that lowers `p` on the memorized images raises `−log p` and, because
`∂(−log p)/∂p = −1/p`, *amplifies* its own gradient — at `p = 0.01` the per-sample forget gradient is
already ~100× the scale it had at `p ≈ 1`. So the combined gradient on `θ` becomes progressively
dominated by the `−0.5·∂forget_loss/∂θ` term, and the sign of the update on the shared weight is set by
"make class 0 wrong" rather than "keep the retained class right." That is the collapse in miniature: the
retained classes lose their vote on the shared weights precisely because their loss is bounded and
saturates, while the forget term's is unbounded and accelerates. The `0.5` coefficient scales the forget
gradient by a half but cannot change which term diverges, which is the analytic confirmation that no
choice of `β` rescues an unbounded ascent — only replacing the unbounded objective can.

Let me situate this strictly in the harness, because the edit surface constrains the implementation to be
much simpler than the general NegGrad+ story. I get one retain minibatch and one forget minibatch per
step, both labeled, both on device, and the fixed Adam. So `unlearn_step` is: forward the model on
`retain_x` and `forget_x`, take cross-entropy on each against their true labels, form the combined loss
`retain_loss − forget_weight · forget_loss`, then a single `zero_grad / backward / step`. There is no
separate forget optimizer, no per-set learning-rate schedule, no checkpointing or rewinding available
here — the rule is exactly the one-line combined objective, and the only hyperparameter I own is
`forget_weight`, fixed at 0.5 to follow the standard NegGrad+ balance. I report `retain_loss` and
`forget_loss` alongside `loss` so the dynamics are visible: I expect `forget_loss` to climb without bound
across the 20 epochs (the tell that the ascent term never settles) and `retain_loss` to creep up as the
shared trunk degrades — the numerical signature of the collapse I am predicting. A quick shape check that
the combined loss is well-formed: both cross-entropies are scalars, so `L = retain_loss − 0.5·forget_loss`
is a scalar, `backward` populates one gradient per parameter, and the single Adam step moves all weights
along the combined direction — retain-descent minus forget-descent, i.e. retain-descent *plus* forget-
ascent, exactly as intended.

One more mechanism, specific to the fixed optimizer, refines what "the weights run off" will actually
look like in the numbers. The harness pins me to `Adam(lr=0.001)`, and Adam divides each coordinate's
update by a running root-mean-square of its own gradient, so the *magnitude* of every step is capped near
`lr` regardless of how large the raw forget gradient grows. That means the divergence will not present as
a numeric explosion in the weights; it presents as the forget-ascent coordinate taking a *sustained,
full-size* step in the same corrupting direction epoch after epoch, because its normalized gradient keeps
pointing "make class 0 more wrong" and never shrinks the way a saturated retain coordinate's does. Over
20 epochs of consistently-signed max-size steps on the shared trunk, the retained representation is
walked steadily off its minimum — a slow-motion collapse rather than a blow-up, but a collapse all the
same. This is why the small fixed learning rate does not save NegGrad: Adam converts the unbounded
gradient into a bounded-magnitude but relentlessly-directed drift, and relentless drift on shared weights
is exactly what erases the retained classes.

So the falsifiable expectations against the previous rung's numbers, sharpened: `forget_acc` should be
(or stay at) zero on all three benchmarks — NegGrad *does* forget, hard. But `retain_acc` should crash
well below the previous rung's 0.8758 / 0.5345 / 0.9373 — that is the unbounded-ascent collapse — and
below the `0.4246 / 0.0580 / 0.4556` break-even floors, so that `unlearn_score` comes in *below* retain
finetuning's 0.8082 / 0.6860 / 0.8185 despite the better forgetting, because the score punishes the lost
utility more than it rewards forgetting that was already near its ceiling. I also expect at least one
benchmark's MIA to dip below 0.5 — the confidently-wrong overshoot. If that is what I see — total
forgetting bought at the cost of a wrecked model — then this rung's two failures together write the
problem the next step inherits: the forget pressure has to have a *place to stop*. An objective with a
fixed point it can reach, rather than an ascent that demands ever more, is the only kind that could push
the forget set to generalization-level uncertainty and *settle* there at MIA = 0.5 instead of running the
weights off, corrupting the retained classes through the shared trunk, and overshooting into the
conspicuous confidently-wrong regime. What such a bounded objective actually is, I leave open here; what
this rung establishes is that the unbounded one is disqualified. The full scaffold module is in the answer.
