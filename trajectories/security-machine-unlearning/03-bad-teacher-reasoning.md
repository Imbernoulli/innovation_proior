NegGrad confirmed the failure I predicted, in exactly the diagnostic shape that points the way out. It
forgot hard — `forget_acc` went to 0.0 on all three — but paid by wrecking the model. On vgg16bn-cifar100
`retain_acc` collapsed to 0.0101, a near-total destruction of the 100-class head; on resnet20-cifar10 to
0.1726; on the hidden mobilenetv2-fmnist to 0.1111. Against the previous rung's ceiling of
0.8758 / 0.5345 / 0.9373 the loss is catastrophic, and worst on the deepest, most-shared-trunk architecture
(vgg16bn), precisely as the unbounded-ascent-through-shared-features argument said. The break-even thresholds
I computed last rung called it: I needed `retain_acc` above `0.4246` / `0.0580` / `0.4556` to beat passive,
and the measured `0.1726` / `0.0101` / `0.1111` fell below all three — the vgg case below even its
absurdly-low `0.0580` bar, the near-total head wipe. So `unlearn_score` had to land below passive, and did:
`0.6822 / 0.5489 / 0.6910` against retain finetuning's `0.8082 / 0.6860 / 0.8185`. The score punished the
crashed utility more than it rewarded the perfect forgetting, exactly the trade the break-even numbers priced.

And the MIA numbers confirm the *second* prediction — the overshoot — on all three: resnet20 AUC `0.1261`,
vgg16bn `0.3635`, fmnist `0.0381`, every one below 0.5. That is not better privacy; it is the
confidently-wrong fingerprint I warned about — the model is now so systematically inverted on class 0 that an
attacker could read the scrubbing off the abnormality. The score even flatters it: fmnist's
`(1 − 0.0381) = 0.9619` is a huge forgetting bonus, yet the benchmark still scored only `0.6910` because the
crashed `retain_acc = 0.1111` dragged it down — a clean demonstration that the score is retain-bound once
forgetting is saturated, and that chasing the MIA past 0.5 buys nothing while costing everything. So the
lesson is sharp and double: the forgetting signal must be (1) *bounded*, so it cannot run the weights off,
and (2) anchored to a *reference behavior* — generalization-level uncertainty, MIA = 0.5 — so it stops at
"looks like it never studied this" instead of overshooting into "confidently wrong about this."

The cheapest fix would rescue the existing mechanism by bounding it, so I should reject that deliberately.
The failure was that `−log p` is unbounded, so the obvious patch is to cap it — clamp the forget loss, or
stop ascending once `p` on class 0 drops below a threshold. But a hard clamp introduces a free parameter I
have no way to set (how wrong is wrong enough?) and, worse, still has no notion of *where* to land: it stops
the push at whatever confidently-inverted point the threshold catches, still in the below-0.5 overshoot
regime. A second patch, pushing the forget logits toward the *uniform* distribution, is better — uniform is
bounded with a real minimum and is near "no information" — but it hard-codes that ignorance is exactly
uniform, which an untrained network's output is not (a fresh net has structured, non-uniform biases from its
random weights). Both patches grope toward the same realization: I want not to bound the ascent by hand but
to *replace the objective with one that has the right target built in*.

Here is the reframe. I have a trained model and want a *different* model that behaves a *prescribed* way on
`D_f` and on `D_r`. "Make a model behave like a prescribed target distribution on given inputs" is exactly
distillation: a student trained to match a teacher's softened output rather than hard labels copies whatever
distribution the teacher emits — and the teacher does not have to be *good*. That is the lever NegGrad
underused. Hard-label ascent has no target distribution at all — "move away from the true label," no
destination, hence unbounded. Distillation replaces "move away from X" with "move toward Y," and a bounded
target Y kills the divergence: KL toward a fixed finite distribution has a genuine minimum at Y, so the
optimizer settles there instead of running off. And unlike the two hand-patches, Y is not a guessed constant —
it is *emitted by a network*, so it carries the right structure automatically.

So what should Y be? I want the unlearned model to behave like the *competent* model on `D_r` and like a
model that *never learned anything* on `D_f`. That asks for two teachers. On retain inputs, distill from a
competent teacher — a frozen copy of the original trained model — so the student keeps its retain knowledge
toward a full target distribution, a richer signal than NegGrad's argmax defense. On forget inputs, distill
from an *incompetent* teacher: a same-architecture network with fresh random weights. Its output on a class-0
image is not a semantic wrong label and not an inverted anti-fact; it is an untrained, largely uninformative
distribution — close to the generalization-level uncertainty I want, and structured the way a real fresh
network is rather than the artificially-flat uniform. Distilling toward *that* pushes the forget set toward
untrained randomness, not confident wrongness, so the MIA should land near 0.5 rather than overshooting. I
get the privacy-safe regime as a direct consequence of choosing the forget teacher random rather than
malicious — exactly the property NegGrad could not express.

Name the pieces. The competent teacher = a frozen deep copy of the model entering unlearning (it has seen
all of `D`). The incompetent teacher = a frozen, randomly-reinitialized copy of the same architecture (it has
seen nothing). The student = the model I update. Crucially the student *starts as the competent model* — it
is the very `model` the harness hands me — so retain utility is satisfied for free at step zero, and the
retain distillation's job shrinks from NegGrad's "repair against a rising tide of ascent damage" to merely
"maintain" a student that already matches the competent teacher. That is why I expect `retain_acc` to recover
toward the ceiling rather than crash.

Route each sample by an unlearning label `l_u` (1 for `D_f`, 0 for `D_r`): pull toward the competent teacher
when `l_u = 0` and toward the incompetent when `l_u = 1`. Because `l_u` is a hard 0/1 selector, exactly one
term is alive per sample, so I fold the selection into one masked target
`target = l_u·t_d + (1 − l_u)·t_s` and compute a single `KL(target ‖ student)` — exact, not a
convex-combination approximation, since `l_u` is never fractional.

Why the retain and forget samples must be fed *together* rather than forget alone: the random teacher's
noise, if the only signal, would bleed through the shared trunk and corrode nearby retained behavior — the
same coupling that destroyed NegGrad. Fed together, the corrosion is checked structurally: a class-0 aircraft
shares wings and fuselage with the other aircraft in `D_r`, and on those retain samples the competent teacher
simultaneously holds the student to the right "aircraft-ish" distribution. The forget signal erases the
*specific* class-0 competence; the retain distillation, live in the same step, preserves the *generic*
features. But I should be honest this is a *balance*, not a proof of safety: if on some architecture the
random target's gradient through the shared trunk is stronger than the competent teacher's restoring
gradient, the balance tips and the retain classes go down with the forget class. That is the one failure mode
I cannot rule out on paper, and I will watch for it.

Now the harness specifics. I get one retain and one forget minibatch per call, and no per-sample `l_u`
loader, so I build the mixed batch by concatenation — `x = cat([retain_x, forget_x])`, `is_forget =
cat([zeros(|retain|), ones(|forget|)])` — the student and both teachers emit `256×C` logits, `is_forget`
broadcasts as a `256×1` mask over the target mixture, the KL reduces to one scalar. Teachers are captured
lazily on the first step from the model: a frozen `eval()` deep copy for the competent teacher, and a second
deep copy whose weights I re-randomize for the incompetent one. The re-randomization must match the harness's
own init — Kaiming-normal `fan_out` on Conv2d, constant 1/0 on BatchNorm, Kaiming-normal `fan_in` on Linear
with zero bias — so the incompetent teacher is the same kind of fresh-init network the harness would build,
whose output is *representative* untrained behavior rather than an out-of-distribution scribble. Both
teachers run under `no_grad`; Adam owns only the student. Temperature `T = 1`: the usual reason to raise `T`
is to soften a sharp competent teacher, but the forget target is already an untrained diffuse distribution
that needs no softening, and the competent teacher's ordinary probabilities are exactly the retain behavior I
want — so no `T^2` term. The loss is `F.kl_div(log_softmax(student/T), target)` with `batchmean` reduction,
the true per-sample average the fixed `lr=0.001` Adam is calibrated against.

The starting point makes the early dynamics unusually clean. At step zero the student *is* the competent
teacher, so on the retain half `student(retain_x) = t_s` exactly, `KL(t_s ‖ student) = 0`, and the retain
distillation contributes zero loss and zero gradient — utility held for free. On the forget half the student
is confidently class 0 while `t_d` is diffuse, so all initial gradient comes from the forget region, pushing
only the forget behavior toward the random target. This is the opposite of NegGrad's start where both terms
fought immediately; here the retain term is silent until the forget push actually disturbs the shared trunk,
at which point its KL wakes up and pulls back — a self-activating restorer, the control loop NegGrad lacked.
And the boundedness this rung leans on is exactly that `KL(target ‖ student)` for a fixed finite target is
bounded below by 0 and attains it uniquely when the student matches the target, so the forget objective has a
reachable finite minimizer — the untrained distribution — with no `p → 0`, `−log p → ∞` runaway anywhere.

The KL direction is a deliberate choice. `F.kl_div(log_softmax(student), target)` computes
`KL(target ‖ student)` — teacher mixture as reference, student as approximation. This forward KL is
mean-seeking: it penalizes the student for putting *low* probability anywhere the target puts mass, forcing
it to cover the full support of whichever teacher it matches. On retain samples that means inheriting the
competent teacher's soft inter-class structure, a richer signal than the argmax. On forget samples it means
being pulled to spread probability the way an untrained network does rather than collapsing onto one
alternative class — exactly the generalization-level uncertainty I want. Reverse KL (mode-seeking) would let
the student collapse onto a single mode of the diffuse forget target, reintroducing confident-wrong risk, so
the forward direction is the one whose behavior matches the "spread out, don't commit" forgetting target.

So the delta from NegGrad: where it ascended an unbounded hard-label loss on `D_f` and ran the weights off, I
distill toward a bounded per-sample teacher mixture — competent original on retain, untrained random on
forget, fed together so the retain distillation holds the trunk while the random target erases the class.
Falsifiable against NegGrad's numbers: `retain_acc` should recover dramatically toward the ceiling rather
than collapsed to 0.0101 / 0.1726 / 0.1111, since the student starts competent and the competent teacher
actively holds the retained classes; `forget_acc` should stay low; and the forget MIA should sit near 0.5
rather than NegGrad's conspicuous 0.3635 / 0.1261 / 0.0381 dips, lifting `unlearn_score` back above the
passive baseline on the visible benchmarks. The genuine uncertainty is the balance I flagged: distilling
toward injected random noise is a weak, noisy target, and on an architecture where that noise couples
strongly into the retain features it could fail to settle — if any benchmark shows a collapsed `retain_acc`
and a spiked `forget_acc`, that is the random-teacher noise winning the tug-of-war through the shared trunk,
and it would say the *target itself* is the weak link. That would hand the next step a sharp problem: keep
the boundedness that fixed NegGrad's divergence, but find a forget target that imports no randomness into the
shared features. The full scaffold module is in the answer.
