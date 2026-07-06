Label smoothing landed almost exactly where I feared, and the numbers say why. On
**resnet20-cifar10** it bought nothing: test accuracy slipped to 0.7878 (from ERM's 0.7972) and the
attack AUC actually *rose* to 0.7678 (from 0.7275), so the composite privacy_score fell to 0.5200,
below ERM's 0.5697. On **vgg16bn-cifar100** the AUC barely budged — 0.8686 versus ERM's 0.8677,
statistically the same wall — and although the privacy_score ticked up to 0.1572 it did so off a tiny
accuracy bump, not off any real defense; the privacy_gap even *widened* to 0.2866. On
**mobilenetv2-fmnist** it was a small net loss (0.8460 vs 0.8685). This is the mean-shift failure made
concrete: smoothing relocated the member-confidence distribution but did not change its *separability*
from the non-members, so the attack — which thresholds confidence and cares only about separability —
was untouched, and I paid a little accuracy for nothing. The lesson is precise. Bounding confidence
uniformly is not enough; I need a regularizer that applies its pressure *where the over-confidence
actually is*, adaptively, rather than pulling every output toward uniform by the same amount. The
member hump did not move toward the non-member hump in any way the attacker notices because smoothing
treats a sample the model has already memorized exactly like a sample it is unsure about.

Let me lay the deltas against ERM side by side, because the *sign* of each one is the real lesson and it
is worse than "flat." On resnet20-cifar10 smoothing moved accuracy `0.7878 − 0.7972 = −0.0094` and the
attack AUC `0.7678 − 0.7275 = +0.0403` — the AUC rose, so `Δprivacy_score = −0.0094 − 0.0403 = −0.0497`,
exactly the drop from 0.5697 to 0.5200. On vgg16bn-cifar100, AUC moved `0.8686 − 0.8677 = +0.0009` (dead
flat, the wall), accuracy `+0.0213`, so the score's `+0.0204` rise is `0.0213 − 0.0009` — *entirely* the
accuracy bump, zero defense. On mobilenetv2-fmnist, AUC `0.5810 − 0.5575 = +0.0235` against accuracy
`+0.0010`, net `−0.0225`. So the attack AUC rose on *all three* benchmarks. My rung-2 expectation was
that a uniform shift would leave the AUC roughly untouched; the data say a uniform shift left it slightly
*worse*, which is even more damning for target-editing than the invariance argument predicted.

The privacy_gap column carries a second surprise I have to sit with honestly, because I predicted the
opposite. My ceiling argument said smoothing caps member confidence at ~0.91, so the member mean should
fall and the gap should *shrink*. Instead the gap *widened* everywhere: resnet `0.0924 → 0.1377`, vgg
`0.1479 → 0.2866` (nearly double), fmnist `0.0211 → 0.0216`. The reconciliation is that the cap lowers
member confidence, yes, but as a regularizer smoothing also changes what the model does on *non-members* —
and it pushed the non-member mean down at least as much as the member mean, so the difference between them
grew rather than closed. I will not over-claim the exact mechanism from one seed, but the robust reading is
unambiguous and it is the thing that matters: editing the target moves *both* humps, and it does not move
them in a way that brings them together — on this evidence it pushed them slightly apart. A uniform,
sample-blind intervention is not merely insufficient; it can be counterproductive. That kills target-side
humbling as a family, not just at `ε = 0.1`.

So let me re-derive the regularizer from the symptom rather than from the cure. The disease shows up in
the *output distribution*: an overfit network puts almost all its softmax mass on one class — a
low-entropy spike — and that spike is exactly the high-confidence signal the attack reads. Smoothing
medicated this by editing the *target* (move mass to uniform), which acts identically on every example.
What I want instead is to penalize the spike *itself*, in proportion to how spiked each output is, so
the pressure concentrates on the confident predictions and ignores the ones already spread out. Entropy
is the scalar that measures the spike. For a softmax output,

  `H(p) = − Σ_i p_i log p_i`,

maximal at uniform, zero at a one-hot spike. Over-confidence is low entropy; the cure is to push
entropy up. It is worth noting the size of the headroom this opens, because it scales with `K`: the
maximum entropy is `log K`, so `log 10 = 2.303` nats on CIFAR-10 and FashionMNIST but `log 100 = 4.605`
nats on CIFAR-100 — twice the range for a memorized spike (`H ≈ 0`) to be pushed into. That cuts both
ways, and I should hold both edges: it is *more* room for the penalty to do useful flattening on the
100-class problem, and also a *longer road* toward the uniform attractor that a constant, undecayed push
can travel too far down. The same `K = 100` that made smoothing's grip thin makes the entropy penalty's
range wide. So take the cross-entropy I already minimize and *subtract* a multiple of the output
entropy:

  `L = L_CE − β·H(p)`.

Let me get the sign right because it is easy to flip. I minimize `L`. The term `−β·H` means that to make
`L` small I want `H` large — high entropy, less peaked outputs — so I am *penalizing low entropy*,
penalizing confidence. The scaffold's `compute_loss` returns this directly: cross-entropy minus a
weight times the mean per-sample entropy. `β` controls how hard I push.

Now is this surgical or a sledgehammer? The answer is in the gradient of the entropy with respect to
the logits, because that is what flows back. Using the softmax Jacobian
`∂p_j/∂z_i = p_j(δ_{ij} − p_i)`,

  `∂H/∂z_i = − Σ_j (∂p_j/∂z_i)(log p_j + 1) = − Σ_j p_j(δ_{ij} − p_i)(log p_j + 1)`,

and splitting the `δ_{ij}` and `−p_i` pieces, with `Σ_j p_j(log p_j + 1) = −H + 1`, this collapses to

  `∂H/∂z_i = p_i(− log p_i − H)`.

Sit with that. The quantity `− log p_i` is the surprisal of class `i`, and `H` is the *mean* surprisal,
so the entropy gradient on logit `i` is the deviation of `i`'s surprisal from the mean, weighted by
`p_i`. Read off the consequences, remembering I optimize `−β·H`, so descent moves opposite the entropy
gradient. For the class the model is confident about, `p_i` is large and `− log p_i` is small (below the
mean), so `∂H/∂z_i` is negative, the loss gradient is positive, and the update pulls that logit *down*.
For a class the model has nearly killed, `p_i ≈ 0`, the whole term is multiplied by `p_i ≈ 0` and the
push is negligible. So this does *not* yank every dead class toward uniform; it acts mostly on the
dominant, over-confident class. This is exactly the adaptivity label smoothing lacked. Smoothing forced
every wrong class toward the same `ε/K`; the confidence penalty weights its pressure by the model's own
current probabilities, so it concentrates precisely on the spiked, memorized predictions — the ones
generating the membership signal — and leaves the already-uncertain outputs alone.

Let me put a number on that concentration to be sure it is real and not a story. Take a mildly spiked
three-class output `p = [0.9, 0.05, 0.05]`. Its entropy is
`H = −0.9 ln 0.9 − 2(0.05 ln 0.05) = 0.0948 + 0.2996 = 0.394`. The entropy gradient on the dominant logit
is `p_i(−ln p_i − H) = 0.9(0.105 − 0.394) = 0.9(−0.289) = −0.260`; on each near-dead logit it is
`0.05(2.996 − 0.394) = 0.05(2.602) = +0.130`. First a consistency check the derivation demands: the
components must sum to zero, since `Σ_i ∂H/∂z_i = Σ_i p_i(−ln p_i) − H Σ_i p_i = H − H = 0`, and indeed
`−0.260 + 2(0.130) = 0`. Good — the arithmetic is self-consistent. Now the substance: under the objective
`L_CE − βH`, the entropy term contributes `−β ∂H/∂z_i`, so on the dominant logit the loss gradient is
`+0.260β` (pushes it *down*) and on each dead logit `−0.130β` (pushes it up, half as hard). The spike is
pressed down at twice the rate any single dead class is lifted — exactly the surgical profile I wanted,
and the opposite of smoothing, which put the *same* `1/K` weight on all of them. That contrast is stark on
CIFAR-100: smoothing's per-class weight there is `1/K = 0.01`, whereas the confidence penalty's weight on
a dominant class near `p ≈ 0.9` is `≈ 0.9` — about 90× more pressure aimed exactly at the leaking spike.
So on the very benchmark where I argued smoothing's grip was diluted 10×, the confidence penalty's grip is
roughly 90× stronger on the offending class. If adaptivity is the missing ingredient, this is where it
should show.

One subtlety the same formula reveals, and it matters for what happens late in training: the push
`p_i(−ln p_i − H)` does not grow without bound as the spike sharpens. As `p_i → 1`, both `−ln p_i → 0` and
`H → 0`, so the dominant-logit gradient `p_i(−ln p_i − H) → 0`. The penalty grips hardest on the
*half-formed* spikes and eases off once an output has already collapsed to near-one-hot. That is a
double-edged property: it means the penalty is gentle on the most extreme members (where the leak is
worst) yet fights hardest during the confidence build-up — which foreshadows both why it may under-defend
the most memorized samples and why a *constant* coefficient applied through the whole build-up can be
destabilizing.

I can make this exact by comparing the two as KL penalties to uniform, because that nails *why* the
confidence penalty should beat smoothing on the attack rather than just shifting things. Smoothing, up
to a constant, adds the *forward* KL `D_KL(u‖p) = Σ_i u_i log(u_i/p_i)`, whose per-class weight is the
*constant* `u_i = 1/K` — equal pressure on every class regardless of what the model is doing. The
confidence penalty is `−H(p)`, and `D_KL(p‖u) = Σ_i p_i log(p_i/u_i) = −H(p) + log K`, so up to a
constant the confidence penalty is the *reverse* KL to uniform, whose per-class weight is the model's
own `p_i` — large precisely on the classes the model is currently confident about. Same target
distribution (uniform), opposite KL direction, and the reversal is the formal reason one is adaptive
and the other uniform. Against this attack, adaptive is what I want: I want to flatten the spikes that
leak, not relocate every output.

Let me verify the identity `D_KL(p‖u) = −H(p) + log K` on its two extremes so I trust the reverse-KL
reading. At the uniform output `p = u`, the left side is `D_KL(u‖u) = 0`, and the right side is
`−H(u) + log K = −log K + log K = 0` — they agree, and the penalty is at its minimum (no pressure on an
already-flat output, correct). At a one-hot output, `H(p) = 0`, so the right side is `log K`, its maximum,
and `D_KL(one-hot‖u) = Σ_i p_i log(p_i/u_i) = 1·log(1/(1/K)) = log K` — they agree again, and the penalty
is maximal exactly on the fully-collapsed spike (before the `p_i → 1` grip-vanishing I noted; the *value*
is maximal even where the *gradient* eases). The identity checks out at both ends, so subtracting entropy
really is, up to the constant `log K`, penalizing the reverse KL from the model's output to uniform —
weighted by the model's own `p_i`, which is the adaptivity in one line of algebra.

Now an honesty check, because this is the rung where I expect to learn whether *any* mean-control
regularizer can break the attack, or only buy stability. The confidence penalty is still, at bottom, a
device that lifts the confidence floor — it pushes the spiked outputs down toward uniform. It is more
*targeted* than smoothing, so I expect it to dent the attack where smoothing could not, especially on
the easy/moderate benchmarks. But it has no explicit lever on the *variance* of the member-confidence
distribution: it pushes each over-confident sample toward uniform, which compresses the high end of the
member distribution, but it does not deliberately *spread* members out to overlap the non-members. So I
expect it to help, but to remain a mean-region intervention — and the harder the dataset, the more I
worry it will not be enough.

There is a sharper worry specific to CIFAR-100, and it comes from a tension the entropy bonus carries.
In reinforcement learning the entropy bonus is wanted *throughout* training — keep exploring — but in
supervised learning I actually want fast convergence on the easy examples and humility only near the
end, once the model starts memorizing. A *constant* `−β·H` from epoch one is a blunt instrument: early
it fights the very convergence I want, late it does its job. The richer version of this method anneals
`β` or replaces it with a hinge `+β·max(0, Γ − H)` that only switches on once entropy has dropped below
a threshold — "weak early, strong near convergence." But the scaffold edit I am landing here is the
*plain, fixed-`β`* form: `L_CE − β·H(p)` with `β` constant for all 300 epochs and no threshold. The
harness exposes `epoch` so I *could* anneal, but the baseline does not — it is the unscheduled entropy
penalty, and I should reason against exactly that. The risk that follows is real: a constant entropy
push on the highest-capacity model (VGG-16-BN, 100 classes) is the configuration most likely to fight
training hard enough to *destabilize* it — if the penalty keeps flattening outputs while the schedule
decays the learning rate, the model can fail to commit to the correct class at all, and on a 100-class
problem with low baseline accuracy that can tip into a degenerate solution where accuracy collapses
toward chance. A fixed `β` has no relief valve for that, where smoothing's strictly-positive target
never could destabilize. So I am trading smoothing's guaranteed safety for adaptivity, and on the
hardest benchmark that trade carries collapse risk.

Let me make the collapse mechanism quantitative, because the destination of a runaway entropy push is a
specific number, not a vague "chance." The entropy term is minimized by the *uniform* output, `p_i = 1/K`
for all `i`, whose confidence is `1/K` — that is the attractor `−βH` pulls toward. On CIFAR-100 that
attractor sits at `1/100 = 0.01` confidence, and if the penalty ever wins the tug-of-war the argmax
becomes an arbitrary tie-break, so test accuracy would fall to about `1/K = 0.01` — chance. So the failure
mode is not "accuracy degrades a bit"; it is a hard floor at 0.01 if the entropy term overpowers fitting.
Now weigh the two forces across the schedule. The cross-entropy pull is `p − y`, which *shrinks* toward
zero as the model fits, and the learning-rate schedule cuts the step size to 0.01 after epoch 150 and to
0.001 after epoch 225. So in the late phase the fitting gradient is both intrinsically small (samples
nearly fit) and scaled down by a 100× smaller learning rate — while the entropy coefficient `β = 0.1` is
*constant*, never decayed. A fixed-magnitude flattening force competing against a fitting force that the
schedule is deliberately shrinking is exactly the recipe for the flattening force to win late in training.
On CIFAR-10 the attractor is only at `1/10 = 0.1` confidence and the model fits easily, so there is margin;
on CIFAR-100, where baseline accuracy is already low (0.5045 at the floor) and the model is barely holding
100 classes together, the same constant push has far less margin before it drags outputs toward the 0.01
uniform point. That is the concrete asymmetry: same `β`, but the 100-class benchmark is the one whose
fitting force is weakest relative to the fixed entropy pull, so it is the one that can tip.

I should be explicit about why I take the confidence penalty *now* rather than jumping straight to a
method with a variance lever, since I already suspect mean-control may not be enough. What smoothing
actually falsified is narrow: *uniform* mean-control fails. It did not test whether *adaptive*
mean-control — pressure aimed at the spikes — can break the wall, and that is a genuinely different
hypothesis with a real chance of working, because the whole reason smoothing failed (it treats memorized
and uncertain samples alike) is precisely what adaptivity fixes. Reaching past this rung to a
fundamentally different lever now would leave that hypothesis untested and I would never know whether
adaptivity alone sufficed. So the disciplined move is to spend exactly one rung isolating the adaptivity
variable: keep the target (uniform) and the goal (raise entropy) identical to smoothing, change only
*forward KL to reverse KL* so the pressure becomes sample-adaptive, and read whether that one change moves
the AUC. Whatever the confidence penalty does or fails to do, it cleanly answers "is adaptivity the
missing ingredient, or do I need to attack variance directly?" — and the fixed-`β` unscheduled form is the
minimal, assumption-free version of that experiment.

The knob. `β` trades data-fitting against humility and is genuinely task-dependent, but the established
default for this penalty as a general regularizer is `β = 0.1`, and I keep it fixed across all three
benchmarks — same value, no per-dataset tuning, no schedule. The implementation is one extra reduction
over the logits I already have: form the softmax `p` once, take its `log` with a small floor on the
argument (a `1e-8` clamp so a near-dead class can never feed `log 0` into the sum), form the per-sample
entropy `−Σ_i p_i log p_i`, average over the batch, and return `L_CE − β·H̄`. No extra forward or backward passes, no auxiliary network, `epoch`
unused — which is itself the design choice under test: a single fixed coefficient, applied identically at
every epoch and every class count, so that whatever this rung shows is attributable to adaptivity alone
and not to any schedule I could have hidden inside it.

The falsifiable expectations against the smoothing numbers. Because the pressure is now adaptive and
concentrated on the spiked outputs, I expect the attack AUC to *drop* where smoothing left it flat —
on **resnet20-cifar10** below smoothing's 0.7678 and ERM's 0.7275, at roughly ERM-level accuracy
(~0.79), so privacy_score recovering above smoothing's 0.5200 and plausibly above ERM. To put an
arithmetic floor on that: if the adaptive pressure merely undoes smoothing's damage and returns the AUC to
ERM's 0.7275 at accuracy ~0.79, the score is `0.79 − (0.7275 − 0.5) = 0.79 − 0.2275 ≈ 0.5625`, already
comfortably above smoothing's 0.5200; any further AUC dent below 0.7275 pushes it past ERM's 0.5697. So the
bar for "adaptivity helped" is concrete and low, and clearing it would confirm that aiming pressure at the
spikes does what uniform target-editing could not. On
**mobilenetv2-fmnist** I expect a clean small win over smoothing's 0.8460 (the penalty trims the few
spiked outputs at almost no accuracy cost), back toward or above ERM's 0.8685. But I should be sober about
the ceiling here: FashionMNIST already sits close to the 0.5 corner where AUC reduction stops paying —
smoothing's AUC was 0.5810, ERM's 0.5575, so there is at most about 0.08 of margin to claw back, and once
the AUC touches 0.5 any accuracy the penalty costs is pure loss. So the *best* case on FashionMNIST is a
modest return toward ERM's 0.8685, not a breakthrough; the win, if it comes, is small by construction, and
the real risk on this benchmark is spending accuracy for AUC that was never there to win. The CIFAR-100 case is
where I am genuinely unsure, and where I expect the constant-`β` trade to bite: either a modest AUC dent
below 0.8686 at preserved low accuracy, *or* — the failure mode I just argued — a training collapse in
which the unscheduled entropy push prevents the 100-class model from committing, accuracy craters toward
chance, and the composite privacy_score becomes meaningless. If CIFAR-100 collapses while the two
easier benchmarks improve, that is the decisive evidence: a fixed, unscheduled mean-control penalty is
both insufficiently powerful *and* unsafe on the hardest case, and the next rung must do two new things
at once — control the loss *level* without crushing it to a degenerate point, and deliberately *spread*
the member distribution so it overlaps the non-members, instead of only shifting its mean. The distilled
scaffold fill is in the answer.
