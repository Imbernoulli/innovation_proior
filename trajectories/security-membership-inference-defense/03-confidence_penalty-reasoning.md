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

So let me re-derive the regularizer from the symptom rather than from the cure. The disease shows up in
the *output distribution*: an overfit network puts almost all its softmax mass on one class — a
low-entropy spike — and that spike is exactly the high-confidence signal the attack reads. Smoothing
medicated this by editing the *target* (move mass to uniform), which acts identically on every example.
What I want instead is to penalize the spike *itself*, in proportion to how spiked each output is, so
the pressure concentrates on the confident predictions and ignores the ones already spread out. Entropy
is the scalar that measures the spike. For a softmax output,

  `H(p) = − Σ_i p_i log p_i`,

maximal at uniform, zero at a one-hot spike. Over-confidence is low entropy; the cure is to push
entropy up. So take the cross-entropy I already minimize and *subtract* a multiple of the output
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

The knob. `β` trades data-fitting against humility and is genuinely task-dependent, but the established
default for this penalty as a general regularizer is `β = 0.1`, and I keep it fixed across all three
benchmarks — same value, no per-dataset tuning, no schedule. The implementation is one extra reduction
over the logits I already have: `log_softmax` once (numerically stable — never `log` of a separately
formed softmax), recover `p = exp(log p)`, the per-sample entropy `−Σ_i p_i log p_i`, average over the
batch, and return `L_CE − β·H̄`. No extra forward or backward passes, no auxiliary network, `epoch`
unused.

The falsifiable expectations against the smoothing numbers. Because the pressure is now adaptive and
concentrated on the spiked outputs, I expect the attack AUC to *drop* where smoothing left it flat —
on **resnet20-cifar10** below smoothing's 0.7678 and ERM's 0.7275, at roughly ERM-level accuracy
(~0.79), so privacy_score recovering above smoothing's 0.5200 and plausibly above ERM. On
**mobilenetv2-fmnist** I expect a clean small win over smoothing's 0.8460 (the penalty trims the few
spiked outputs at almost no accuracy cost), back toward or above ERM's 0.8685. The CIFAR-100 case is
where I am genuinely unsure, and where I expect the constant-`β` trade to bite: either a modest AUC dent
below 0.8686 at preserved low accuracy, *or* — the failure mode I just argued — a training collapse in
which the unscheduled entropy push prevents the 100-class model from committing, accuracy craters toward
chance, and the composite privacy_score becomes meaningless. If CIFAR-100 collapses while the two
easier benchmarks improve, that is the decisive evidence: a fixed, unscheduled mean-control penalty is
both insufficiently powerful *and* unsafe on the hardest case, and the next rung must do two new things
at once — control the loss *level* without crushing it to a degenerate point, and deliberately *spread*
the member distribution so it overlaps the non-members, instead of only shifting its mean. The distilled
scaffold fill is in the answer.
