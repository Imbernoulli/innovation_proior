Bad Teacher did most of what I hoped, and it failed in the one place I flagged as the risk — which is the
opening I need. On the two visible benchmarks it recovered the utility NegGrad threw away: resnet20-cifar10
`retain_acc` back to 0.8902 (above even retain finetuning's 0.8758), vgg16bn-cifar100 to 0.5246 (right at the
ceiling), with `forget_acc` 0.0 and MIA AUC 0.4809 / 0.4561 — near the 0.5 never-saw-it point, not NegGrad's
conspicuous dips. The scores read `(0.8902 + 1 + 0.5191)/3 = 0.8031` and `(0.5246 + 1 + 0.5439)/3 = 0.6895`,
both back above the passive baseline. So the bounded, reference-anchored distillation worked as argued: the
boundedness fixed the retain crash, and the random-teacher target landed the MIA near 0.5. Two of three
predictions came in clean.

But the hidden mobilenetv2-fmnist benchmark *collapsed* — `retain_acc` 0.0, `forget_acc` 1.0, MIA AUC 0.6974,
`unlearn_score` 0.1009 — and the *shape* of the collapse names its cause. `forget_acc` did not go to zero, it
went to *one*: the model predicts class 0 on the forget images, and simultaneously `retain_acc` went to zero,
so it predicts class 0 on *everything*. The MIA rose to 0.6974, above the passive 0.4817 — memorization is
worse than where I started. The student did not forget class 0 and did not keep the retained classes; it
degenerated into a near-constant class-0 predictor. That is precisely the failure I named: the random teacher
is a weak, noisy target, and on this architecture the noise coupled strongly enough into the shared features
that the balance tipped — the retain distillation could not hold, the trunk was dragged off its competent
function, and the whole model degenerated. So the weak link localizes to the *forget-side target*: distilling
toward an untrained random network injects noise that, on a fragile trunk, corrodes retain features faster
than the competent teacher can repair them. I need a forget-side signal just as bounded but *clean* — one
that imports no randomness into the shared trunk.

The cheapest repair would keep the dual-teacher structure and denoise the random teacher, so I reject it
first. I could average several random re-inits to smooth the target, or temper its output — but averaging
pulls the target toward uniform, the hand-built guess I already argued is not honest ignorance; and softening
does not remove the fact that the teacher's *particular* random weights impose a *particular* structured-but-
wrong distribution with no relationship to the retained-class geometry, so its gradient through the shared
trunk is still an arbitrary, retain-unsafe direction. The problem is not the *amount* of noise; it is that a
second, unrelated network is the reference at all. Drop the second teacher entirely.

Here is the reframe. NegGrad's failure was *no target*; Bad Teacher's fix was *a bounded but noisy* target.
Keep the boundedness, drop the second teacher, and use the one clean reference I already have — the competent
teacher — for *both* directions. On `D_r` I already pull the student *toward* the competent teacher (that
part worked cleanly on all three, even the one that collapsed — the retain KL was never the problem). On
`D_f`, instead of pulling toward a noisy random network, push the student *away from the competent teacher*.
"Be anything but what the knowledgeable model would say here" is a sharper, cleaner forget signal than "be
like this particular random network": one well-defined reference for both directions, importing no unrelated
network's noise. The teacher is the anchor; retain examples are positives attracted toward it, forget
examples are negatives repelled from it.

Define the forget distance as distillation measures behavior — KL on softened outputs, capturing the full
learned function (the inter-class similarity in the small probabilities), not just the argmax. For a sample
`x`, `d(x) = KL(teacher ‖ student)` on temperature-`T` softmaxes: small `d` means the student obeys the
teacher, large `d` means it disobeys, so forgetting is *maximize* `d` on `D_f`. This is not NegGrad in
disguise even though "maximize a divergence" sounds unbounded again: NegGrad maximized `−log p` toward a
*label*, unbounded because the student can always be a little more certain of the wrong class; here I maximize
KL away from a *fixed finite teacher distribution*, and the student's softmax lives on the bounded probability
simplex, so `KL(teacher ‖ student)` is large-but-finite everywhere except the measure-zero corners, and its
gradient comes from a frozen finite reference, not the `−1/p` blow-up. Controlled forget pressure, no random
network's noise in the trunk.

But I cannot maximize `d` on `D_f` and stop, for the reason that has bitten every rung: `D_f` and `D_r` share
the trunk, so pushing away on forget examples leaks through the shared features and degrades retain behavior.
So I pair it with a counter-pressure — while pushing away on `D_f`, pull *toward* the teacher on `D_r`
(minimize `d` there) — and, because KL-to-teacher only keeps the student *teacher-like* (inheriting the
teacher's small errors), add ordinary cross-entropy to the *true* retain labels as an independent "and
actually be correct" anchor. The retain objective is `gamma·CE(student, y) + alpha·KL(teacher ‖ student)`,
the forget objective the negated KL. I want the true-label CE to carry retain and the teacher-KL to be only a
light "don't drift" regularizer — the CE is ground truth, the KL merely keeps the student near a teacher that
is itself only approximately right — so `gamma ≈ 0.99`, `alpha ≈ 0.01`, a 99-to-1 split. Both max and min
phases must reference the *same* `d = KL(teacher ‖ student)` to the *same* frozen teacher, differing only in
sign and minibatch, or the two would fight; there is then exactly one notion of "teacher-like" and forgetting
is literally its negation on `D_f`.

The optimization dynamics are where this earns its strength over a naive combined loss. Threw into one loss,
every minibatch would ask the same shared parameters to move *toward* the teacher on retain points and *away*
on forget points at once — the classic min-max thrash, the oscillation GANs have, whose standard fix is not
lockstep but *alternation*. So I separate the steps: a max-step on the forget set (push away) and a min-step
on the retain set (pull back + be correct), as distinct optimizer steps. The number of max-steps matters
subtly: each max-epoch raises forget error but nudges retain off through the trunk; each min-epoch restores
retain but, pulling the student back toward the teacher, can partially *re-teach* the forget set, because the
min-step's teacher-KL on retain data still moves shared weights back toward the competent function that
recognized class 0. Alternate forever and the two fight to a stalemate. The resolution: run the max-step only
for the first `msteps` epochs to *inject* forgetting, then min-steps only to *restore* retain — the min-only
tail touches the forget set only through the weaker trunk coupling, so it must not re-collapse the forget
error. `msteps = 2` because each extra max-epoch does more trunk damage for the tail to repair and I have
only 20 epochs; two epochs of injection with eighteen of restore is the smallest injection that reliably
moves the forget behavior while leaving a long restorative tail — exactly the allocation to protect the
fragile benchmark that collapsed last time.

The temperature carries the one place a scale error would quietly break the method. I soften with `T = 4`:
high enough to expose the teacher's inter-class structure, low enough not to flatten to noise. Soft-target
distillation gradients scale as `1/T^2` (softening shrinks the logit differences the loss sees, and
backprop through two `1/T` softmaxes brings a `1/T^2` factor), so to keep the term's effective weight
independent of `T` the standard fix multiplies the KD loss by `T^2` — restoring the scale so `alpha` means
the same at `T = 4` as at `T = 1`. Without it, both the forget-push and retain-anchor would be silently
down-weighted 16× and the method would barely move; the factor is load-bearing.

Pin the rest to the harness. `unlearn_step` is called per (retain, forget) minibatch with `step`/`epoch`
counters and the fixed Adam. The teacher is captured lazily on the first step as a frozen `eval()` deep copy
of `model` — no second network, so none of Bad Teacher's randomness enters. The schedule reads `epoch`: while
`epoch < msteps`, do the max-step (forward student and teacher on `forget_x`, temperature-`T` KL, backward
`−forget_kl`, step) before the min-step; every epoch does the min-step (`gamma`·CE to true retain labels plus
`alpha`·KL-to-teacher). Two `zero_grad/backward/step` calls in the injection phase, one after. The fixed
`Adam(lr=0.001)` is exactly the small step the min-max instability wants — a large step would let the
oscillation blow up, and it is the same reason the two-epoch injection is safe. The min-step running *every*
epoch, including the two that also carry a max-step, matters: during injection each epoch does max-then-min,
so the retain anchor defends the trunk *in the same epoch* the forget push disturbs it, and only the
*forget-specific* displacement the retain data does not constrain is left standing; the `msteps` cutoff then
stops adding new displacement while eighteen min-only epochs consolidate retain.

I know what "working" looks like in the diagnostics: `forget_kl` driven up in the first two epochs (the
forgetting being injected), then plateauing or drifting only slightly after the cutoff (the min-only tail must
not re-collapse forget error to zero, the whole reason for capping the max phase), while `retain_ce` falls
and stays low and `retain_kl` stays small. If `forget_kl` instead decayed toward zero in the tail, the
min-only phase would be re-teaching the forget set faster than assumed, and I would shorten the tail or lower
`alpha` — but with `alpha = 0.01` the re-teaching pull is already deliberately weak, so I expect forget error
to hold.

So the delta from Bad Teacher is targeted at its collapse: where it pulled the forget set *toward a noisy
random teacher* that corroded the trunk on mobilenetv2-fmnist, I push the forget set *away from the one clean
competent teacher*, importing no randomness, with the push and pull separated into alternating min/max phases
and the max phase capped at `msteps`. Falsifiable against Bad Teacher's numbers: on the two visible
benchmarks I expect parity-or-better — resnet20 around/above 0.8031, vgg16bn around/above 0.6895 — with
`retain_acc` held at the ceiling and the forget MIA near or slightly below 0.4809 / 0.4561, staying close to
0.5. The decisive test is the hidden mobilenetv2-fmnist that Bad Teacher collapsed on (0.1009): if removing
the random teacher and scheduling the min/max is the right fix, SCRUB should *not* collapse there,
`retain_acc` recovering to the high-0.8 range, because the single clean reference cannot inject the
trunk-corroding noise. If it still collapses on mobilenetv2, the problem was never the random teacher but the
depthwise-separable architecture's fragility under forget pressure, and the diagnosis would have to turn
architecture-aware — a different question than the objective-design one every rung has been asking. The full
scaffold module is in the answer.
