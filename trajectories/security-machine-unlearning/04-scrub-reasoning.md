Bad Teacher did most of what I hoped, and it failed in the one place I flagged as the risk — which is
exactly the opening I need. On the two visible benchmarks it recovered the utility NegGrad threw away:
resnet20-cifar10 `retain_acc` came back to 0.8902 (above even retain finetuning's 0.8758), vgg16bn-cifar100
to 0.5246 (right at the ceiling), with `forget_acc` 0.0 and MIA AUC 0.4809 / 0.4561 — near the 0.5
never-saw-it operating point, not NegGrad's conspicuous 0.3635 dip. So the bounded, reference-anchored
distillation worked: `unlearn_score` climbed to 0.8031 / 0.6895, back above the passive baseline. But the
hidden mobilenetv2-fmnist benchmark *collapsed* — `retain_acc` 0.0, `forget_acc` 1.0, MIA AUC 0.6974,
`unlearn_score` 0.1009 — the model not only lost all retained accuracy but ended up predicting class 0 on
*everything*. That is precisely the failure mode I named: the random teacher is a *weak, noisy target*,
and on an architecture where that noise couples strongly into the shared features it did not settle; the
forget signal won, the retain distillation could not hold, and the whole model degenerated. So the
diagnosis is sharp: the *target* on the forget side is the weak link. Distilling toward an untrained random
network injects noise that, in the worst case, corrodes the retain features faster than the competent
teacher can repair them. I need a forget-side signal that is just as bounded but *clean* — that does not
import randomness into the shared trunk at all.

Here is the reframe that fixes it. NegGrad's failure was *no target* (unbounded ascent away from the true
label). Bad Teacher's fix was *a bounded target* but a *noisy* one (a random teacher). What if I keep the
boundedness but drop the second teacher entirely, and instead use the one clean reference I already have —
the competent teacher — for *both* directions? On `D_r` I already pull the student *toward* the competent
teacher (that part worked). On `D_f`, instead of pulling toward a noisy random network, I push the student
*away from the competent teacher*. "Be anything but what the knowledgeable model would say here" is a
sharper, cleaner forget signal than "be like this particular random network": it uses a single,
well-defined reference for both directions and imports no randomness into the shared features. The
competent teacher is the anchor; retain examples are positives I attract toward it; forget examples are
negatives I repel from it. That contrastive structure is what I want.

So define the forget distance the way distillation measures behavior — KL on softened outputs, which
captures the model's full learned *function* (the inter-class similarity structure in the small
probabilities), not just the argmax. For a sample `x`, `d(x) = KL(teacher ‖ student)` on temperature-`T`
softmaxes. Small `d` means the student obeys the teacher; large `d` means it disobeys. The forgetting goal
then writes itself: *maximize* `d` on `D_f` — push the student's forget distribution away from the
teacher's. Crucially, this push is *teacher-referenced and finite*, not the unbounded hard-label ascent
that blew up NegGrad: it is a divergence between two softened distributions against a frozen finite
teacher, so the forget pressure is the controlled kind I needed, and it carries no random network's noise
into the trunk the way Bad Teacher's collapse did.

But I cannot maximize `d` on `D_f` and stop, for the reason that has bitten every rung: `D_f` and `D_r`
share the trunk, so pushing the student away from the teacher on forget examples leaks through the shared
features and degrades retain behavior — this is the same shared-feature coupling that crashed NegGrad and
that drove Bad Teacher's mobilenetv2 collapse. So I pair it with a counter-pressure: while pushing away on
`D_f`, simultaneously pull *toward* the teacher on `D_r` — minimize `d` on the retain set — and, because
the KL-to-teacher only keeps the student *teacher-like* (inheriting any small teacher errors), add the
ordinary cross-entropy to the *true* retain labels as an independent "and actually be correct" anchor. The
retain objective is `gamma·CE(student, y) + alpha·KL(teacher ‖ student)`, the forget objective is the
negated KL `-KL(teacher ‖ student)`. The cross-entropy carries the weight (`gamma ≈ 0.99`), the
teacher-KL is a light don't-drift regularizer (`alpha ≈ 0.01`).

Now the optimization dynamics, which is where this rung earns its strength over a naive combined loss. If
I threw all three terms into one loss and stepped, on every minibatch I would be asking the same shared
parameters to move *toward* the teacher on retain points and *away* on forget points at once — the two
demands interfere and the resultant thrashes. This is the classic min-max signature, the same oscillation
GANs have, and the standard fix is not lockstep but *alternation*. So I separate the steps: a max-step on
the forget set (push away) and a min-step on the retain set (pull back + be correct), run as distinct
optimizer steps. And the number of max-steps matters, subtly: each max-epoch raises forget error but
nudges retain off through the trunk; each min-epoch restores retain but, pulling the student back toward
the teacher, can partially *re-teach* the forget set. Alternate forever and the two fight to a stalemate
with neither goal reached. The resolution: do the max-step only for the first `msteps` epochs to *inject*
the forgetting, then run only min-steps for the rest to *restore* the retain performance the max-steps
disturbed — without re-teaching the forget set, because the min-steps only touch `D_r` and reach the
forget set only through the much weaker trunk coupling. A couple of max-epochs forgets; the trailing
min-only epochs clean up retain while forget error stays elevated.

Let me pin the implementation to this harness, because the edit surface is what I actually fill and it
shapes the schedule. The `unlearn_step` is called per (retain, forget) minibatch with `step` and `epoch`
counters and the fixed Adam. The teacher is the single clean reference — captured lazily on the first
step as a frozen deep copy of `model` in `eval()`, no second network, so none of Bad Teacher's
randomness enters. The schedule reads the `epoch` counter directly: while `epoch < msteps`, do the
max-step (forward the student and the teacher on `forget_x`, compute the temperature-`T` KL, backward
`-forget_kl`, step) *before* the min-step; for every epoch including the early ones, do the min-step
(forward on `retain_x`, CE to the true retain labels plus `alpha`·KL-to-teacher, backward, step). Two
separate `zero_grad/backward/step` calls in the max phase, one in the min-only phase. The temperature-`T`
KL carries the `T^2` factor — `d(x) = T^2·KL(softmax(teacher/T) ‖ softmax(student/T))` via
`F.kl_div(log_softmax(student/T), softmax(teacher/T), reduction='batchmean')·T^2` — because the
soft-target gradient scales as `1/T^2`, and the `T^2` restores its magnitude so the distillation term
keeps a stable weight; `T = 4` sits where softening exposes the inter-class structure without flattening
to noise, the value from the authors' VGG settings. Defaults `msteps = 2`, `kd_T = 4`, `alpha = 0.01`,
`gamma = 0.99` follow that same notebook. The fixed `Adam(lr=0.001)` is exactly the small step the
min-max instability wants — a large step would let the max/min oscillation blow up, so the harness's
conservative rate is fortunate here.

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
architecture's fragility under any forget pressure, and the next move would have to be architecture-aware.
But I expect the clean-reference min-max to be exactly the stabilizer that benchmark needed — making this
the strongest rung on the ladder, the first method that forgets sharply, preserves utility, *and* stays
stable across all three architectures. The full scaffold module is in the answer.
