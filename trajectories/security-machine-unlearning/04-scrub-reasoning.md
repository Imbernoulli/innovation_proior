Bad Teacher did most of what I hoped, and it failed in the one place I flagged as the risk — which is
exactly the opening I need. On the two visible benchmarks it recovered the utility NegGrad threw away:
resnet20-cifar10 `retain_acc` came back to 0.8902 (above even retain finetuning's 0.8758), vgg16bn-cifar100
to 0.5246 (right at the ceiling), with `forget_acc` 0.0 and MIA AUC 0.4809 / 0.4561 — near the 0.5
never-saw-it operating point, not NegGrad's conspicuous 0.3635 dip. Let me confirm the scores read the way
I think: resnet20 is `(0.8902 + 1 + (1 − 0.4809))/3 = 2.4093/3 = 0.8031`, and vgg16bn is
`(0.5246 + 1 + (1 − 0.4561))/3 = 2.0685/3 = 0.6895` — both exactly as reported, and both back above the
passive baseline (0.8082 / 0.6860). So the bounded, reference-anchored distillation worked precisely as
argued: the boundedness fixed the retain crash, and the random-teacher target landed the MIA near 0.5
instead of overshooting. Two of the three predictions from last rung came in clean.

But the hidden mobilenetv2-fmnist benchmark *collapsed* — `retain_acc` 0.0, `forget_acc` 1.0, MIA AUC
0.6974, `unlearn_score` 0.1009. And this is worth reading carefully, because the *shape* of the collapse
names its cause. `forget_acc` did not go to zero, it went to *one* — the model predicts class 0 on the
forget images — and simultaneously `retain_acc` went to zero, so the model is predicting class 0 on
*everything*. The MIA rose to 0.6974, *above* the passive baseline's 0.4817, meaning memorization on the
forget set is worse than where I started. Put together: the student did not forget class 0 and did not
keep the retained classes; it degenerated into a near-constant class-0 predictor. That is precisely the
failure mode I named — the random teacher is a *weak, noisy target*, and on this architecture the noise
coupled strongly enough into the shared features that the balance tipped: the retain distillation could
not hold, the shared trunk was dragged off its competent function, and the whole model degenerated. Score
check: `(0.0 + (1 − 1.0) + (1 − 0.6974))/3 = 0.3026/3 = 0.1009`, confirming the reading. So the diagnosis
is sharp and it localizes the weak link to the *forget-side target*: distilling toward an untrained random
network injects noise that, on a fragile enough trunk, corrodes the retain features faster than the
competent teacher can repair them. I need a forget-side signal that is just as bounded but *clean* — that
imports no randomness into the shared trunk at all.

Let me consider the cheapest repair first — keep the dual-teacher structure but denoise the random teacher —
so I can reject it deliberately. I could average several random re-inits to smooth the target, or distill
toward a softened/tempered random output. But averaging random networks pulls the target back toward
uniform, which is the hand-built guess I already argued is not the honest ignorant behavior; and softening
a random teacher does not remove the fact that its *particular* random weights impose a *particular*
structured-but-wrong distribution that has no relationship to the retained-class geometry, so its gradient
through the shared trunk is still an arbitrary direction with no reason to be retain-safe. The problem is
not the *amount* of noise, it is that a second, unrelated network is the reference at all. So I want to
drop the second teacher entirely.

Here is the reframe that fixes it. NegGrad's failure was *no target* (unbounded ascent away from the true
label). Bad Teacher's fix was *a bounded target* but a *noisy* one (a random teacher). What if I keep the
boundedness but drop the second teacher, and instead use the one clean reference I already have — the
competent teacher — for *both* directions? On `D_r` I already pull the student *toward* the competent
teacher (that part worked, cleanly, on all three benchmarks — even the one that collapsed, the retain KL
itself was never the problem). On `D_f`, instead of pulling toward a noisy random network, I push the
student *away from the competent teacher*. "Be anything but what the knowledgeable model would say here"
is a sharper, cleaner forget signal than "be like this particular random network": it uses a single,
well-defined reference for both directions and imports no unrelated network's noise into the shared
features. The competent teacher is the anchor; retain examples are positives I attract toward it; forget
examples are negatives I repel from it. That contrastive structure against a single reference is what I
want.

So define the forget distance the way distillation measures behavior — KL on softened outputs, which
captures the model's full learned *function* (the inter-class similarity structure in the small
probabilities), not just the argmax. For a sample `x`, `d(x) = KL(teacher ‖ student)` on temperature-`T`
softmaxes. Small `d` means the student obeys the teacher; large `d` means it disobeys. The forgetting goal
then writes itself: *maximize* `d` on `D_f` — push the student's forget distribution away from the
teacher's. I have to be careful this is not just NegGrad in disguise, because "maximize a divergence"
sounds unbounded again. The difference is real: NegGrad maximized `−log p` toward a *label*, which has no
upper bound because the student can always be made a little more certain of the wrong class. Here I
maximize KL away from a *fixed finite teacher distribution*; the student's softmax lives on the probability
simplex, a bounded set, so `KL(teacher ‖ student)` over that simplex is large-but-finite everywhere except
the measure-zero corners, and crucially the *gradient* of this KL is a well-scaled quantity coming from a
frozen finite reference, not the `−1/p` blow-up. It is the controlled kind of forget pressure I needed,
and it carries no random network's noise into the trunk the way Bad Teacher's collapse did.

A shape and sign check before I build on it, because the KL direction has to be consistent between the
max and min phases or the two would fight. I compute `d(x) = KL(teacher ‖ student)` — the teacher's
temperature-`T` softmax as the reference `p_t`, the student's log-softmax as the argument — in a single
helper used by both phases. On the retain set I *descend* `alpha·d` (pull the student toward the teacher);
on the forget set I *ascend* `d`, i.e. descend `−d`. The two phases therefore reference the *same*
distance to the *same* frozen teacher, differing only in sign and in which minibatch they read, so there
is exactly one notion of "teacher-like" in the method and forgetting is literally its negation on `D_f`.
Both minibatches are `128×C` at the logits, the softmaxes `128×C`, and each KL reduces with `batchmean`
to one scalar per phase — so each phase is one well-formed scalar loss feeding one `backward`, and the
`T^2` factor multiplies that scalar. Nothing here can diverge the way NegGrad did: `d` is a divergence
between two points on the bounded probability simplex against a fixed reference, so its ascent has no
`−log p → ∞` escape hatch, only a finite push away from a fixed distribution.

But I cannot maximize `d` on `D_f` and stop, for the reason that has bitten every rung: `D_f` and `D_r`
share the trunk, so pushing the student away from the teacher on forget examples leaks through the shared
features and degrades retain behavior — the same shared-feature coupling that crashed NegGrad and that
drove Bad Teacher's mobilenetv2 collapse. So I pair it with a counter-pressure: while pushing away on
`D_f`, simultaneously pull *toward* the teacher on `D_r` — minimize `d` on the retain set — and, because
the KL-to-teacher only keeps the student *teacher-like* (inheriting any small teacher errors), add the
ordinary cross-entropy to the *true* retain labels as an independent "and actually be correct" anchor. The
retain objective is `gamma·CE(student, y) + alpha·KL(teacher ‖ student)`, the forget objective is the
negated KL `−KL(teacher ‖ student)`. Let me size the two retain coefficients rather than take them on
faith: I want the true-label CE to *carry* the retain objective and the teacher-KL to be only a light
"don't drift" regularizer, because the CE is the ground truth and the KL merely keeps the student near a
teacher that is itself only approximately right. So `gamma ≈ 0.99` and `alpha ≈ 0.01` — a 99-to-1 split,
retain accuracy driven almost entirely by the true labels with a whisper of teacher-anchoring. That
asymmetry also means the retain objective barely fights the teacher, so most of the teacher's role in this
method is on the *forget* side as the thing to flee.

Now the optimization dynamics, which is where this rung earns its strength over a naive combined loss. If
I threw all three terms into one loss and stepped, on every minibatch I would be asking the same shared
parameters to move *toward* the teacher on retain points and *away* on forget points at once — the two
demands interfere and the resultant thrashes. This is the classic min-max signature, the same oscillation
GANs have, and the standard fix is not lockstep but *alternation*. So I separate the steps: a max-step on
the forget set (push away) and a min-step on the retain set (pull back + be correct), run as distinct
optimizer steps. And the number of max-steps matters, subtly: each max-epoch raises forget error but
nudges retain off through the trunk; each min-epoch restores retain but, pulling the student back toward
the teacher, can partially *re-teach* the forget set — because the min-step's teacher-KL, applied on
retain data, still moves shared weights back toward the competent function that recognized class 0.
Alternate forever and the two fight to a stalemate with neither goal reached. The resolution: do the
max-step only for the first `msteps` epochs to *inject* the forgetting, then run only min-steps for the
rest to *restore* the retain performance the max-steps disturbed — without re-teaching the forget set,
because the min-steps only touch `D_r` and reach the forget set only through the much weaker trunk
coupling. A couple of max-epochs forgets; the trailing min-only epochs clean up retain while forget error
stays elevated. Why `msteps = 2` and not more: each extra max-epoch does more shared-trunk damage for the
min tail to repair, and I have only 20 epochs total, so I want the smallest injection that reliably moves
the forget behavior — two epochs of max, eighteen of restore — leaving a long restorative tail, which is
exactly the budget allocation that should protect the fragile benchmark that collapsed last time.

Let me pin the temperature and its gradient factor, because it is the one place a sign/scale error would
quietly break the method. I soften with `T = 4`: high enough that the teacher's inter-class structure (the
dark knowledge) is exposed rather than a near-one-hot argmax, low enough that it does not flatten to noise.
The subtlety is that soft-target distillation gradients scale as `1/T^2` — softening by `T` shrinks the
logit differences the loss sees, and the backprop through two `1/T` softmax temperatures brings a `1/T^2`
factor into the gradient magnitude. To keep the distillation term's effective weight independent of `T`,
the standard fix is to multiply the KD loss by `T^2`, restoring the gradient scale so that `alpha` means
the same thing at `T = 4` as it would at `T = 1`. So `d(x) = T^2·KL(softmax(teacher/T) ‖ softmax(student/T))`
via `F.kl_div(log_softmax(student/T), softmax(teacher/T), reduction='batchmean')·T^2`. Without the `T^2`
the forget-push and the retain-anchor would both be silently down-weighted 16× at `T = 4`, and the method
would barely move — so this factor is load-bearing, not cosmetic.

Let me pin the rest of the implementation to this harness, because the edit surface is what I actually
fill and it shapes the schedule. The `unlearn_step` is called per (retain, forget) minibatch with `step`
and `epoch` counters and the fixed Adam. The teacher is the single clean reference — captured lazily on
the first step as a frozen deep copy of `model` in `eval()`, no second network, so none of Bad Teacher's
randomness enters. The schedule reads the `epoch` counter directly: while `epoch < msteps`, do the
max-step (forward the student and the teacher on `forget_x`, compute the temperature-`T` KL, backward
`−forget_kl`, step) *before* the min-step; for every epoch including the early ones, do the min-step
(forward on `retain_x`, `gamma`·CE to the true retain labels plus `alpha`·KL-to-teacher, backward, step).
Two separate `zero_grad/backward/step` calls in the max phase, one in the min-only phase. Defaults
`msteps = 2`, `kd_T = 4`, `alpha = 0.01`, `gamma = 0.99` are the settings I adopt. The fixed
`Adam(lr=0.001)` is exactly the small step the min-max instability wants — a large step would let the
max/min oscillation blow up, so the harness's conservative rate is fortunate here, and it is the same
reason the two-epoch injection is safe: small steps mean two epochs of max inject a controlled amount of
forgetting rather than a runaway one.

One implementation detail I want to get right is that the min-step runs *every* epoch, including the two
early epochs that also carry a max-step, not only in the trailing phase. This matters mechanically: during
the injection phase each epoch does max-then-min, so the retain anchor is defending the shared trunk *in
the same epoch* the forget push is disturbing it — the restore is not deferred entirely to the tail, it is
interleaved from the start and merely *continues alone* after epoch `msteps`. That is what keeps the two
injection epochs from doing unchecked damage: the forget push moves shared weights, then immediately in
the same epoch the retain CE-plus-KL pulls the retained-class behavior back, and only the *forget-specific*
displacement that the retain data does not constrain is left standing. The `msteps` cutoff then stops
adding new forget displacement while eighteen further min-only epochs keep consolidating the retain side.

Let me trace the intended signature in the returned diagnostics so I know what "working" looks like
epoch by epoch. In the first two epochs `forget_kl` should be *driven up* — the max-step is maximizing
`KL(teacher ‖ student)` on the forget set, so the student's forget distribution moves away from the
teacher's confident class-0 output, and `forget_kl` climbing is the tell that the forgetting is being
injected. After epoch 2 there is no max-step, so `forget_kl` is no longer pushed and should *plateau or
drift only slightly* — the min-only tail touches the forget set only through weak trunk coupling, so it
must not re-collapse the forget error back to zero, which is the whole reason for capping the max phase.
Meanwhile `retain_ce` should fall and stay low across all 20 epochs, and `retain_kl` should stay small
because the student began at the teacher and the 99-to-1 CE-weighting keeps it near-correct. If instead I
saw `forget_kl` decay back toward zero in the tail, that would mean the min-only phase is re-teaching the
forget set through the trunk faster than I assumed, and I would have to revisit the schedule — shorten the
restorative tail or lower `alpha` further — but with `alpha = 0.01` the teacher-KL's re-teaching pull is
already deliberately weak, so I expect the forget error to hold.

So the delta from Bad Teacher is precise and targeted at its collapse. Where Bad Teacher pulled the forget
set *toward a noisy random teacher* — a weak target that, on mobilenetv2-fmnist, corroded the shared trunk
faster than the competent teacher could repair it — I now push the forget set *away from the one clean
competent teacher*, importing no randomness, and I separate the push and pull into alternating min/max
phases with the max phase capped at `msteps` so the forgetting is injected early and the retain set is
restored late. Single reference, controlled finite forget pressure, scheduled to avoid the stalemate.

Falsifiable expectations against Bad Teacher's numbers. On the two visible benchmarks where Bad Teacher
already worked, I expect SCRUB to *match or slightly beat* it — resnet20-cifar10 around or above the 0.8031
`unlearn_score` and vgg16bn-cifar100 around or above 0.6895 — because the cleaner forget signal and the
restorative min-only tail should hold `retain_acc` at the ceiling while keeping forget error elevated; I
also expect the forget MIA AUC to come in near or slightly below Bad Teacher's (0.4809 / 0.4561), staying
close to the 0.5 never-saw-it point rather than overshooting. The decisive test is the hidden
mobilenetv2-fmnist benchmark *that Bad Teacher collapsed on* (0.1009): if removing the random teacher and
scheduling the min/max is the right fix, SCRUB should *not* collapse there — `retain_acc` should recover to
the high-0.8 range and `unlearn_score` should land near the visible-benchmark level (~0.8), because the
single clean reference cannot inject the trunk-corroding noise that sank Bad Teacher. If SCRUB still
collapses on mobilenetv2, then the problem was never the random teacher but the depthwise-separable
architecture's fragility under forget pressure, and the diagnosis would have to turn architecture-aware —
a different kind of question than the objective-design one every rung so far, this one included, has been
asking. But I expect the clean-reference min-max to be exactly the stabilizer that benchmark needed — making this
the strongest rung on the ladder, the first method that forgets sharply, preserves utility, *and* stays
stable across all three architectures. The full scaffold module is in the answer.
