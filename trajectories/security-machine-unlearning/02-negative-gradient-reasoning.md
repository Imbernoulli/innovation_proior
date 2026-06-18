Retain finetuning told me exactly what it would: it held utility and barely forgot. On
resnet20-cifar10 `retain_acc` came in at 0.8758 with `forget_mia_auc` at 0.4512; on vgg16bn-cifar100,
`retain_acc` 0.5345 and `forget_mia_auc` 0.4765; on the hidden mobilenetv2-fmnist, `retain_acc` 0.9373
and `forget_mia_auc` 0.4817. Read those MIA numbers against the only value that matters for forgetting —
0.5, the AUC of a model that genuinely never saw `D_f`. They sit just *below* 0.5, which is the
fingerprint of the thing I predicted: the membership signal is essentially un-erased. The attack can
barely separate forget-train from forget-test confidences, but that is because the original model was
already near that operating point on this confidence statistic, *not* because retain finetuning did any
forgetting — it never touched class 0, so it could not have moved that gap on purpose. The `unlearn_score`
landed at 0.8082 / 0.6860 / 0.8185, and the binding constraint is plainly the forgetting axis: utility is
near the harness ceiling, and the only headroom left is in the `(1 - forget_mia_auc)` and `(1 - forget_acc)`
terms. So the diagnosis is clean and it dictates the next move: I need an *active* forgetting pressure —
a term in the loss that actually pushes against `D_f` — because catastrophic forgetting alone, as the
numbers confirm, does not erode a class the model spent 80 epochs memorizing.

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
retain finetuning left untouched. And I cannot do ascent on `D_f` alone, for the same reason the prior
rung's failure was structural: `D_f` and `D_r` are not processed by disjoint weights. The lower
convolutional layers compute features used by every class; pushing the model to be wrong on class 0
sends gradients back through that shared trunk and perturbs the retained classes too. So the active
forget term has to be paired with a defensive descent term on the retain set that holds the line. That
is the NegGrad+ shape: descend the retain cross-entropy while ascending the forget cross-entropy, a
single combined loss `L = retain_loss - β · forget_loss`, with `β` trading the strength of the forgetting
ascent against the retain descent. In this harness `β` is `forget_weight = 0.5`, and the whole thing is
one backward / step on the summed loss — not two separate updates.

Now I have to be honest about the danger in this term, and the danger is exactly what will set this
rung's number. Cross-entropy to the true label is *unbounded above*. There is no maximum to "be wrong":
as the model's predicted probability on the true class 0 approaches zero, `-log p` runs to infinity, so
the ascent term has no fixed point to settle at — it keeps demanding the model push more probability mass
off class 0, forever. The retain descent term *is* bounded (cross-entropy is minimized at the correct
answer with a finite floor), so as training proceeds the two terms are not symmetric: the bounded retain
term saturates near its minimum and stops producing large gradients, while the unbounded forget-ascent
term keeps producing gradient as long as the model is not infinitely wrong on class 0. The ascent term
therefore *wins the late-training dynamics*. With `β = 0.5` the forget term is half-weighted, which slows
this but does not change the asymptotics: there is no β that turns an unbounded ascent into a bounded one;
β only rescales how fast the weights run off. And they will run off — the natural endpoint of an
unbounded ascent objective is the weights diverging, the representation getting torn apart, and the
*retained* accuracy collapsing as collateral, because the very features the ascent term is corrupting are
the shared features the retain classes depend on.

Let me predict the shape of that collapse concretely, because it is what distinguishes this rung's
failure from the previous one. Retain finetuning failed *softly* — high utility, weak forgetting.
NegGrad will fail *hard* in the opposite direction: it will absolutely forget — `forget_acc` will go to
zero and the MIA AUC may even drop *below* the prior rung as the model becomes confidently, abnormally
wrong on class 0 — but it will pay for that by wrecking `retain_acc`. The unbounded ascent, leaking
through the shared trunk and unchecked by a saturated retain term, drags the retained-class accuracy down
with it. So I expect the diagnostic signature to be a *crashed* `retain_acc`, dramatically below the
0.8758 / 0.5345 / 0.9373 the previous rung established as the achievable ceiling — and because the score
averages utility and forgetting, a crashed retain term will sink `unlearn_score` below the passive
baseline even though forgetting itself succeeds. The deeper architectures with the most shared structure
to corrupt (vgg16bn on cifar100, with 100 classes packed into a shared trunk) should crash the hardest;
the smaller resnet20 on cifar10 should survive a little better but still fall.

There is a second, subtler problem with this term that matters for the *privacy* axis specifically, and
it is the conceptual gap the next rung will have to close. Driving `forget_acc` to exactly zero — making
the model *confidently wrong* on class 0 — is not what forgetting should look like. Picture a model that
genuinely never trained on class 0: shown a class-0 image, it does not confidently shout some other
specific class; it sits at generalization-level uncertainty, spreading probability roughly the way its
ignorance warrants. A model that has been *taught a sharp anti-fact* about class 0 — always predict
something-other-than-0, with confidence — has not forgotten; it has learned a new, sharp, *inverted*
competence on exactly those inputs. And that inverted competence is itself a fingerprint: a membership
attacker who notices the model is weirdly, confidently wrong on precisely these inputs has learned they
were specially scrubbed. So "maximal forgetting" via unbounded ascent overshoots into the
confidently-wrong regime, which is privacy-conspicuous. NegGrad has no notion of *how much* to forget —
the unbounded term simply forgets as hard as the optimizer allows before it diverges. The right target is
to forget *only as much as a model that never saw `D_f` would*, no more; NegGrad cannot express that
target, because hard-label cross-entropy ascent has no "stop at generalization-level uncertainty" fixed
point. That is the precise weakness that motivates moving from hard-label ascent to a *reference-anchored,
bounded* forgetting signal at the next rung.

Let me situate this strictly in the harness, because the edit surface constrains the implementation to be
much simpler than the general NegGrad+ story. I get one retain minibatch and one forget minibatch per
step, both labeled, both on device, and the fixed Adam. So `unlearn_step` is: forward the model on
`retain_x` and `forget_x`, take cross-entropy on each against their true labels, form the combined loss
`retain_loss - forget_weight · forget_loss`, then a single `zero_grad / backward / step`. There is no
separate forget optimizer, no per-set learning-rate schedule, no checkpointing or rewinding available
here — the rule is exactly the one-line combined objective, and the only hyperparameter I own is
`forget_weight`, fixed at 0.5 to follow the standard NegGrad+ balance. I report `retain_loss` and
`forget_loss` alongside `loss` so the dynamics are visible: I expect `forget_loss` to climb without bound
across the 20 epochs (the tell that the ascent term never settles) and `retain_loss` to creep up as the
shared trunk degrades — the numerical signature of the collapse I am predicting.

So the falsifiable expectations against the previous rung's numbers, sharpened: `forget_acc` should drop
to (or very near) zero on all three benchmarks — NegGrad *does* forget, hard. But `retain_acc` should
crash well below the previous rung's 0.8758 / 0.5345 / 0.9373 — that is the unbounded-ascent collapse,
and it is the whole point of the diagnosis. Because of the crash, `unlearn_score` should come in *below*
retain finetuning's 0.8082 / 0.6860 / 0.8185 despite the better forgetting — the score punishes the lost
utility more than it rewards the forgetting, since retain finetuning was already near the MIA floor. If
that is what I see — total forgetting bought at the cost of a wrecked model — then the lesson for the next
rung is written: I need a forgetting pressure that is *bounded* and *anchored to a reference behavior*, so
it pushes the forget set toward generalization-level uncertainty and *stops there*, instead of an
unbounded ascent that runs the weights off and corrupts the retained classes through the shared trunk. The
full scaffold module is in the answer.
