NegGrad confirmed the failure I predicted, and it confirmed it in exactly the diagnostic shape that
points the way out. It forgot, hard — `forget_acc` went to 0.0 on all three benchmarks — but it paid by
wrecking the model. On vgg16bn-cifar100 `retain_acc` collapsed to 0.0101, a near-total destruction of the
100-class retained head; on resnet20-cifar10 it fell to 0.1726; on the hidden mobilenetv2-fmnist to
0.1111. Set those against the previous rung's utility ceiling — 0.8758 / 0.5345 / 0.9373 — and the loss
is catastrophic, and worst on the deepest, most-shared-trunk architecture (vgg16bn), precisely as the
unbounded-ascent-through-shared-features argument predicted.

The sharpest confirmation is that the break-even thresholds I computed last rung called this correctly.
I had derived that NegGrad beats the passive baseline only if `retain_acc` stays above `0.4246` on
resnet20, `0.0580` on vgg16bn, and `0.4556` on fmnist. The measured retain accuracies came in at
`0.1726`, `0.0101`, and `0.1111` — below all three floors, and the vgg case fell below even its
absurdly-low `0.0580` bar, which is the near-total head wipe I said only a catastrophic crash could
produce. So `unlearn_score` had to land below passive, and it did: `0.6822 / 0.5489 / 0.6910` against
retain finetuning's `0.8082 / 0.6860 / 0.8185`, below on every benchmark. Let me check one decomposition
to be sure I am reading the score the same way the harness computes it: resnet20 gives
`(0.1726 + 1 + (1 − 0.1261))/3 = (0.1726 + 1 + 0.8739)/3 = 2.0465/3 = 0.6822`, exactly the reported
value — so the arithmetic holds and the collapse is real, not a misread. The score punished the crashed
utility more than it rewarded the perfect forgetting, exactly the trade the break-even numbers priced.

And look at the MIA numbers, because they confirm the *second* prediction — the overshoot — on all three:
resnet20 AUC `0.1261`, vgg16bn `0.3635`, fmnist `0.0381`, every one *below* 0.5. That is not better
privacy; it is the confidently-wrong fingerprint I warned about — the model is now so systematically
inverted on class 0 that the attacker could read the scrubbing off the abnormality. The score design even
flatters it: fmnist's `(1 − 0.0381) = 0.9619` is a huge forgetting bonus, yet the benchmark still scored
only `0.6910` because the crashed `retain_acc = 0.1111` dragged it down — a clean demonstration that the
score is retain-bound once forgetting is saturated, and that chasing the MIA past 0.5 buys nothing while
costing everything. So the lesson is sharp and double: the forgetting signal must be (1) *bounded*, so it
cannot run the weights off, and (2) anchored to a *reference behavior* — generalization-level uncertainty,
MIA = 0.5 — so it stops at "looks like it never studied this" instead of overshooting into "confidently
wrong about this."

Before I reach for a new mechanism, let me ask whether I can rescue the *existing* one by simply bounding
it, because that is the cheapest possible fix and I should reject it explicitly rather than skip past it.
The failure was that `−log p` is unbounded, so the obvious patch is to cap it — clamp the forget loss at
some ceiling, or stop ascending once `p` on class 0 drops below a threshold. Walk it: a hard clamp turns
the ascent off once the forget images are "wrong enough," which does remove the divergence, but it
introduces exactly the free parameter I have no way to set — *how* wrong is wrong enough? — and worse, it
still has no notion of *where* to land. A clamp stops the push at whatever confidently-inverted point the
threshold happens to catch, which is still in the below-0.5 overshoot regime; it prevents the blow-up but
not the conspicuousness. A second patch, pushing the forget logits toward the *uniform* distribution
instead of away from the true label, is better — uniform is a bounded target with a real minimum, and it
is genuinely near "no information" — but it hard-codes that the ignorant behavior is exactly uniform,
which is not what an untrained network actually outputs (an untrained net has structured, non-uniform
biases from its random weights and the data statistics). So "push toward uniform" is a hand-built guess at
the reference behavior. Both patches are groping toward the same realization: what I actually want is not
to bound the ascent by hand but to *replace the objective with one that has the right target built in*.

Here is the question that reframes the whole problem. I have a trained model, and I want a *different*
model that behaves a *prescribed* way on `D_f` and a *prescribed* way on `D_r`. "Make a model behave like
a prescribed target distribution on given inputs" is exactly what distillation does: a student is trained
to match a teacher's softened output distribution rather than hard labels, and it copies *whatever
distribution the teacher emits*. The piece I underused with NegGrad is that the teacher does not have to
be *good* — it is just a source of target behavior. The student will faithfully imitate a wise teacher,
and it will just as faithfully imitate an idiot. That is the lever. The disaster with NegGrad was that
hard-label ascent has no target distribution at all — it just says "move away from the true label," with
no destination, hence unbounded. Distillation replaces "move away from X" with "move toward Y," and a
*bounded* target Y is exactly what kills the divergence: KL toward a fixed finite distribution has a
genuine minimum at Y itself, so the optimizer settles at the target instead of running off past it. And
unlike the two hand-patches above, the target Y is not a guessed constant — it is *emitted by a network*,
so it carries the right structure automatically.

So what should Y be? I want the unlearned model to behave like the *full, competent* model on `D_r` and
like a *model that never learned anything* on `D_f`. That asks for two teachers. On retain inputs, distill
from the competent teacher — a frozen copy of the original trained model — so the student keeps every bit
of its retain knowledge, *toward a target distribution* rather than just defending the argmax the way
NegGrad's retain term did. On forget inputs, distill from an *incompetent* teacher: a same-architecture
network with fresh random weights, never trained. Its output on a class-0 image is not a semantic wrong
label and it is not an inverted anti-fact; it is an untrained, largely uninformative distribution —
close to the generalization-level uncertainty I argued is the correct forgetting target, and structured
the way a real fresh network is rather than the artificially-flat uniform. Distilling the student toward
*that* on `D_f` pushes it toward untrained randomness, not toward confident wrongness, so the MIA should
land near 0.5 rather than overshooting below it. I get the privacy-safe regime not as a tuning afterthought
but as a *direct consequence* of choosing the forget teacher to be random rather than malicious — which is
exactly the property NegGrad could not express and the clamp/uniform patches could only fake.

Name the pieces. The competent teacher `T_s` = a frozen deep copy of the model as it enters unlearning
(it has seen all of `D`). The incompetent teacher `T_d` = a frozen, randomly-reinitialized copy of the
*same architecture* (it has seen nothing). And the student = the model I actually update. Crucial: the
student starts *as the competent model* — it is the very `model` the harness hands me, already at the
pretrained weights — so one of my two goals, retain utility, is satisfied for free at step zero, and I
only have to work the forget region without corrupting it. Compare that to NegGrad, where the retain term
was forever fighting a rising tide of ascent damage; here the retain distillation only has to *hold* a
student that already matches the competent teacher. That difference is exactly why I expect `retain_acc`
to recover toward the ceiling rather than crash: the retain objective's job shrank from "repair" to
"maintain," and it starts already satisfied.

The objective routes each sample to its teacher by an unlearning label `l_u`: 1 if the sample is in `D_f`,
0 if in `D_r`. Per sample, pull the student toward `T_s` when `l_u = 0` and toward `T_d` when `l_u = 1`.
Using KL as the distillation discrepancy, that is `(1 − l_u)·KL(T_s ‖ S) + l_u·KL(T_d ‖ S)`. Let me
verify the vectorization I want to use is exact rather than an approximation, because I am going to fold
the selection into a single target. Because `l_u` is a hard 0/1 selector, exactly one term is alive per
sample, so I can build a single per-sample target distribution `target = l_u·t_d + (1 − l_u)·t_s` from the
two teacher softmaxes and compute one `KL(target ‖ student)`. Check the two cases: for `l_u = 0`,
`target = 0·t_d + 1·t_s = t_s` and the KL is `KL(t_s ‖ S)`, the competent term; for `l_u = 1`,
`target = t_d` and the KL is `KL(t_d ‖ S)`, the incompetent term — identical to the two-term form in both
cases. So the fold is exact, not a convex-combination approximation (it would only be an approximation if
`l_u` were fractional, which it never is), and it collapses the whole batched objective to one masked-
target KL: build the mixed target once, call KL once.

There is a subtlety the random teacher handles that I should make explicit, because it is *why* I must
feed retain and forget *together* rather than forget alone. The random teacher's noise, if it were the
only signal, would bleed through the shared trunk and corrode nearby retained behavior — the same
shared-feature coupling that destroyed NegGrad. But here the corrosion is checked structurally: a class-0
aircraft shares wings and fuselage with the other aircraft in `D_r`, and on all those retain samples the
*competent* teacher is simultaneously holding the student to the right "aircraft-ish" distribution. The
forget signal erases the *specific* class-0 competence; the retain distillation, live in the same step,
preserves the *generic* features. That is why the harness feeds a balanced mixture of retain and forget
samples each step — without the live retain pressure, the random teacher's noise would spread. I should
be honest that this is a *balance*, not a proof of safety: if on some architecture the random target's
gradient through the shared trunk is stronger than the competent teacher's restoring gradient, the balance
tips and the retain classes go down with the forget class. That is the one failure mode I cannot rule out
on paper, and I will watch for it.

Now the harness specifics, because the edit surface is narrower than the general method and I must land
exactly its implementation. I do not get a separate unlearning DataLoader carrying `l_u` per sample, and
I do not get to run a divergence diagnostic against the random teacher — those live outside `unlearn_step`.
What I get is one retain minibatch and one forget minibatch per call. So I build the mixed batch by
*concatenation*: `x = cat([retain_x, forget_x])`, and the selector `is_forget = cat([zeros(|retain|),
ones(|forget|)])`. A shape check: if each minibatch is 128, the concatenated batch is 256, the student
and both teachers each emit `256×C` logits, the softmaxes are `256×C`, `is_forget` is length 256 and
broadcasts as a `256×1` mask over the target mixture, and the final KL reduces to one scalar — all
consistent. The teachers are captured lazily on the first step from the model itself — a deep copy frozen
in `eval()` for the competent teacher, and a second deep copy whose weights I re-randomize for the
incompetent teacher. The re-randomization must match the harness's own `initialize_weights`: Kaiming-normal
`fan_out` on Conv2d, constant 1/0 on BatchNorm weight/bias, Kaiming-normal `fan_in` on Linear with zero
bias — so the incompetent teacher is the same kind of fresh-init network the harness would have built,
not some arbitrary noise, which matters because I want its output to be *representative* untrained
behavior, the honest reference, not an out-of-distribution scribble. Both teachers run under `no_grad`;
the fixed Adam owns only the student (`model`) parameters. The temperature `T` I keep at 1: the usual
reason to raise `T` is to soften a sharp *competent* teacher so its dark knowledge shows, but here the
forget target is an untrained distribution that needs no softening, and the competent teacher's ordinary
probabilities are exactly the retain behavior I want to copy, sharpness and all — so `T = 1` is the
natural setting and there is no `T^2` gradient-scaling term to worry about since the objective is pure
distillation with no hard-label term mixed in. The loss is `F.kl_div(log_softmax(student/T), target)` —
student log-probabilities as the input, the mixed teacher *probabilities* as the target, the standard
direction; I use `batchmean` reduction so the KL is the true per-sample average (sum over classes, mean
over samples), the mathematically correct normalization that the fixed `lr=0.001` Adam is calibrated
against.

Let me trace the very first step to check that this objective does the right thing at initialization,
because the student's starting point makes the early dynamics unusually clean and worth verifying. At
step zero the student *is* the competent teacher, weight for weight. So on the retain half of the batch,
`student(retain_x) = T_s(retain_x)` exactly, which means `t_s` and the student softmax coincide and
`KL(t_s ‖ student) = 0` — the retain distillation contributes *zero loss and zero gradient* at the start,
confirming utility is held for free at step zero rather than needing repair. On the forget half, the
student (= competent) is confidently class 0 while the target `t_d` is the random teacher's diffuse
distribution, so `KL(t_d ‖ student)` is large and *all* of the initial gradient comes from the forget
region — the update pushes only the forget behavior, in the direction of the random target. This is the
qualitative opposite of NegGrad's start, where both terms fought from the first step; here the retain
term is silent until the forget push actually disturbs the shared trunk, at which point the retain KL
wakes up and pulls back. It is a self-activating restorer: it produces gradient exactly and only when the
forget push has moved the student off the competent function, which is the well-behaved control loop
NegGrad lacked.

I should also confirm the boundedness claim I am leaning the whole rung on, since it is what fixes
NegGrad. `KL(target ‖ student)` for a fixed finite `target` is bounded below by 0 and attains 0 uniquely
when the student softmax equals `target`. So the forget objective has a genuine, finite minimizer — the
untrained distribution — and once the student reaches it the gradient vanishes and the optimizer stops
pushing. There is no `p → 0`, `−log p → ∞` runaway anywhere, because the target is a proper distribution
the student can actually match, not a label it must flee forever. That single structural fact — a
reachable finite minimum instead of an unbounded ascent — is the entire mechanism by which this rung
should avoid the weight blow-up that crashed `retain_acc` last time.

The direction of the KL is a real choice and I want it deliberate, not incidental. I use
`F.kl_div(log_softmax(student), target)`, which computes `KL(target ‖ student)` — the teacher mixture as
the reference distribution, the student as the approximation. This is the "forward" KL relative to the
target, and it is mean-seeking: it penalizes the student for putting *low* probability anywhere the
target puts mass, so the student is forced to *cover the full support* of whichever teacher it is matching.
That is exactly what I want in both regions. On retain samples, covering the competent teacher's full
distribution means the student inherits the soft inter-class structure — the "dark knowledge" that this
airplane is a little bit like a bird — not just the argmax, which is a richer and more stabilizing retain
signal than the hard-label defense NegGrad used. On forget samples, covering the random teacher's diffuse
support means the student is pulled to spread its probability out the way an untrained network does,
rather than collapsing onto one alternative class — which is precisely the generalization-level
uncertainty I want and the opposite of NegGrad's confident single-class inversion. Reverse KL
(`KL(student ‖ target)`, mode-seeking) would let the student collapse onto a single mode of the diffuse
forget target and ignore the rest, reintroducing a confident-wrong risk; so the forward direction is not
just the library default, it is the one whose mean-seeking behavior matches the "spread out, don't commit"
forgetting target I am after.

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
classes, so I expect `retain_acc` far above NegGrad's on all three, and specifically back above the
break-even floors it fell through last rung. `forget_acc` should still be low (the random teacher gives
the forget set no reason to predict class 0), but the privacy story should be *better* than NegGrad's
overshoot: the forget-set MIA AUC should sit near 0.5 — the never-saw-it operating point — rather than
NegGrad's conspicuous 0.3635 / 0.1261 / 0.0381 dips below it. The net effect on `unlearn_score` should be
to climb back *above* the retain-finetune baseline on the visible benchmarks, since I now recover most of
the utility NegGrad threw away while keeping the forgetting. The one place I am genuinely unsure is the
balance I flagged: re-randomizing a teacher and distilling toward near-uniform noise is a *weak, noisy
target*, and on an architecture where that noise couples strongly into the retain features it could fail to
settle — if any benchmark shows a collapsed `retain_acc` and a spiked `forget_acc`, that is the
random-teacher noise winning the tug-of-war through the shared trunk, and it would say the *target* itself
is the weak link: distilling toward injected randomness is what a fragile trunk cannot absorb. That
outcome would hand the next step a sharply-defined problem — keep the boundedness this rung buys, which
fixed NegGrad's divergence, but find a forget target that imports no such randomness into the shared
features. What that cleaner target should be, I leave for the numbers to force rather than presume here.
The full scaffold module is in the answer.
