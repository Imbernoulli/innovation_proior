NegGrad confirmed the failure I predicted, and it confirmed it in exactly the diagnostic shape that
points the way out. It forgot, hard — `forget_acc` went to 0.0 on all three benchmarks — but it paid by
wrecking the model. On vgg16bn-cifar100 `retain_acc` collapsed to 0.0101, a near-total destruction of the
100-class retained head; on resnet20-cifar10 it fell to 0.1726; on the hidden mobilenetv2-fmnist to
0.1111. Set those against the previous rung's utility ceiling — 0.5345 / 0.8758 / 0.9373 — and the loss
is catastrophic, and worst on the deepest, most-shared-trunk architecture (vgg16bn), precisely as the
unbounded-ascent-through-shared-features argument predicted. The `unlearn_score` came in at 0.5489 /
0.6822 / 0.6910, *below* retain finetuning's 0.6860 / 0.8082 / 0.8185 on every benchmark — the score
punished the crashed utility more than it rewarded the perfect forgetting. And look at the MIA numbers:
on vgg16bn the AUC actually *dropped* to 0.3635, well below 0.5. That is not better privacy; that is the
confidently-wrong fingerprint I warned about — the model is now so systematically inverted on class 0
that the attacker could read the scrubbing off the abnormality. So the lesson is sharp and double: the
forgetting signal must be (1) *bounded*, so it cannot run the weights off, and (2) anchored to a
*reference behavior* — generalization-level uncertainty — so it stops at "looks like it never studied
this" instead of overshooting into "confidently wrong about this."

Let me ask the question that reframes the whole problem. I have a trained model, and I want a *different*
model that behaves a *prescribed* way on `D_f` and a *prescribed* way on `D_r`. "Make a model behave like
a prescribed target distribution on given inputs" is exactly what distillation does: a student is trained
to match a teacher's softened output distribution rather than hard labels, and it copies *whatever
distribution the teacher emits*. The piece I underused with NegGrad is that the teacher does not have to
be *good* — it is just a source of target behavior. The student will faithfully imitate a wise teacher,
and it will just as faithfully imitate an idiot. That is the lever. The disaster with NegGrad was that
hard-label ascent has no target distribution at all — it just says "move away from the true label," with
no destination, hence unbounded. Distillation replaces "move away from X" with "move toward Y," and a
*bounded* target Y is exactly what kills the divergence: KL toward a fixed finite distribution has a
genuine minimum, so the optimizer settles instead of running off.

So what should Y be? I want the unlearned model to behave like the *full, competent* model on `D_r` and
like a *model that never learned anything* on `D_f`. That asks for two teachers. On retain inputs, distill
from the competent teacher — a frozen copy of the original trained model — so the student keeps every bit
of its retain knowledge, *toward a target* rather than just defending the argmax the way NegGrad's retain
term did. On forget inputs, distill from an *incompetent* teacher: a same-architecture network with fresh
random weights, never trained. Its output on a class-0 image is not a semantic wrong label and it is not
an inverted anti-fact; it is an untrained, largely uninformative distribution — close to the
generalization-level uncertainty I argued is the correct forgetting target. Distilling the student toward
*that* on `D_f` pushes it toward untrained randomness, not toward confident wrongness. I get the
privacy-safe regime not as a tuning afterthought but as a *direct consequence* of choosing the forget
teacher to be random rather than malicious — which is exactly the property NegGrad could not express.

Name the pieces. The competent teacher `T_s` = a frozen deep copy of the model as it enters unlearning
(it has seen all of `D`). The incompetent teacher `T_d` = a frozen, randomly-reinitialized copy of the
*same architecture* (it has seen nothing). And the student = the model I actually update. Crucial: the
student starts *as the competent model* — it is the very `model` the harness hands me, already at the
pretrained weights — so one of my two goals, retain utility, is satisfied for free at step zero, and I
only have to work the forget region without corrupting it. Compare that to NegGrad, where the retain term
was forever fighting a rising tide of ascent damage; here the retain distillation only has to *hold* a
student that already matches the competent teacher.

The objective routes each sample to its teacher by an unlearning label `l_u`: 1 if the sample is in `D_f`,
0 if in `D_r`. Per sample, pull the student toward `T_s` when `l_u = 0` and toward `T_d` when `l_u = 1`.
Using KL as the distillation discrepancy, that is `(1 - l_u)·KL(T_s ‖ S) + l_u·KL(T_d ‖ S)`. Because
`l_u` is a hard 0/1 selector, exactly one term is alive per sample, so I can fold the selection into a
single per-sample target distribution, `target = l_u·t_d + (1 - l_u)·t_s` with the two teacher softmaxes,
and the whole batched objective collapses to one `KL(target ‖ student)`. That is not an approximation —
for `l_u = 0` the target is `t_s`, for `l_u = 1` it is `t_d`, identical to the two-term form — it is just
the efficient vectorization: build the mixed target once, call KL once.

There is a subtlety the random teacher handles that I should make explicit, because it is *why* I must
feed retain and forget *together* rather than forget alone. The random teacher's noise, if it were the
only signal, would bleed through the shared trunk and corrode nearby retained behavior — the same
shared-feature coupling that destroyed NegGrad. But here the corrosion is checked structurally: a class-0
aircraft shares wings and fuselage with the other aircraft in `D_r`, and on all those retain samples the
*competent* teacher is simultaneously holding the student to the right "aircraft-ish" distribution. The
forget signal erases the *specific* class-0 competence; the retain distillation, live in the same step,
preserves the *generic* features. That is why the harness feeds a balanced mixture of retain and forget
samples each step — without the live retain pressure, the random teacher's noise would spread.

Now the harness specifics, because the edit surface is narrower than the general method and I must land
exactly its implementation. I do not get a separate unlearning DataLoader carrying `l_u` per sample, and
I do not get to run a JS-divergence diagnostic against the random teacher — those live outside `unlearn_step`.
What I get is one retain minibatch and one forget minibatch per call. So I build the mixed batch by
*concatenation*: `x = cat([retain_x, forget_x])`, and the selector `is_forget = cat([zeros(|retain|),
ones(|forget|)])`. The teachers are captured lazily on the first step from the model itself — a deep copy
frozen in `eval()` for the competent teacher, and a second deep copy whose weights I re-randomize for the
incompetent teacher. The re-randomization must match the harness's own `initialize_weights`: Kaiming-normal
`fan_out` on Conv2d, constant 1/0 on BatchNorm weight/bias, Kaiming-normal `fan_in` on Linear with zero
bias — so the incompetent teacher is the same kind of fresh-init network the harness would have built,
not some arbitrary noise. Both teachers run under `no_grad`; the fixed Adam owns only the student
(`model`) parameters. The temperature `T` I keep at 1: the usual reason to raise `T` is to soften a sharp
*competent* teacher so its dark knowledge shows, but here the forget target is an untrained distribution
that needs no softening, and the competent teacher's ordinary probabilities are exactly the retain
behavior I want to copy, sharpness and all — so `T = 1` is the natural setting and there is no `T^2`
gradient-scaling term to worry about since the objective is pure distillation with no hard-label term
mixed in. The loss is `F.kl_div(log_softmax(student/T), target)` — student log-probabilities as the input,
the mixed teacher *probabilities* as the target, the standard direction; I use `batchmean` reduction so
the KL is the true per-sample average (sum over classes, mean over samples), the mathematically correct
normalization that the fixed `lr=0.001` Adam is calibrated against.

So the delta from NegGrad is precise. Where NegGrad ascended an unbounded hard-label loss on `D_f` and
descended a hard-label loss on `D_r` in one step — and ran the weights off — I now distill the student
toward a *bounded* per-sample teacher mixture: the competent original on retain samples, an untrained
random network on forget samples, with the two fed together so the retain distillation holds the shared
trunk while the random target erases the specific class. The forgetting can no longer diverge, because KL
toward a fixed finite target has a minimum; and it lands at generalization-level uncertainty by
construction, not at confident wrongness, because the target *is* an uninformative distribution rather
than an inverted label.

Falsifiable expectations against NegGrad's numbers. `retain_acc` should *recover dramatically* — back up
toward the retain-finetune ceiling (0.5345 / 0.8758 / 0.9373), not collapsed to 0.0101 / 0.1726 / 0.1111 —
because the student starts at the competent weights and the competent teacher actively holds the retained
classes, so I expect `retain_acc` far above NegGrad's on all three. `forget_acc` should still be low (the
random teacher gives the forget set no reason to predict class 0), but the privacy story should be
*better* than NegGrad's overshoot: the forget-set MIA AUC should sit near 0.5 — the never-saw-it operating
point — rather than NegGrad's conspicuous 0.3635 dip below it. The net effect on `unlearn_score` should be
to climb back *above* the retain-finetune baseline on the visible benchmarks, since I now recover most of
the utility NegGrad threw away while keeping the forgetting. The one place I am genuinely unsure is
stability: re-randomizing a teacher and distilling toward near-uniform noise is a *weak, noisy target*, and
on an architecture where that noise couples strongly into the retain features it could fail to settle — if
any benchmark shows a collapsed `retain_acc` and a spiked `forget_acc`, that is the random-teacher noise
winning, and it would say the *target* itself is the weak link, which is the precise opening for the next
rung: replace the noisy random teacher with a single clean reference and push *away* from it rather than
toward noise. The full scaffold module is in the answer.
