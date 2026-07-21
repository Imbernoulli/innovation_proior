Label smoothing landed almost exactly where I feared, and the numbers say why. On
**resnet20-cifar10** it bought nothing: test accuracy slipped to 0.7878 (from ERM's 0.7972) and the
attack AUC actually *rose* to 0.7678 (from 0.7275), so the composite privacy_score fell to 0.5200,
below ERM's 0.5697. On **vgg16bn-cifar100** the AUC barely budged — 0.8686 versus ERM's 0.8677,
statistically the same wall — and although privacy_score ticked up to 0.1572 it did so off a tiny
accuracy bump, not any real defense; the privacy_gap even *widened* to 0.2866. On
**mobilenetv2-fmnist** it was a small net loss (0.8460 vs 0.8685). This is the mean-shift failure made
concrete: smoothing relocated the member-confidence distribution but did not change its *separability*
from the non-members, so the attack — which thresholds confidence and cares only about separability —
was untouched, and I paid a little accuracy for nothing. The lesson is precise: bounding confidence
uniformly is not enough; I need pressure applied *where the over-confidence actually is*, adaptively,
because smoothing treats a sample the model has memorized exactly like a sample it is unsure about.

Laid against ERM, the *sign* of each delta is worse than "flat." On resnet20-cifar10 accuracy moved
`−0.0094` and the attack AUC `+0.0403`, so `Δprivacy_score = −0.0497`. On vgg16bn-cifar100, AUC `+0.0009`
(dead flat, the wall) against accuracy `+0.0213`, so the score's rise is *entirely* the accuracy bump,
zero defense. On mobilenetv2-fmnist, AUC `+0.0235` against accuracy `+0.0010`, net `−0.0225`. The attack
AUC rose on *all three* benchmarks. My rung-2 expectation was that a uniform shift would leave the AUC
roughly untouched; the data say it left it slightly *worse*, which is even more damning for
target-editing than the invariance argument predicted.

The privacy_gap column carries a second surprise I have to sit with, because I predicted the opposite. My
ceiling argument said smoothing caps member confidence at ~0.91, so the member mean should fall and the
gap *shrink*. Instead it *widened* everywhere: resnet `0.0924 → 0.1377`, vgg `0.1479 → 0.2866` (nearly
double), fmnist `0.0211 → 0.0216`. The reconciliation is that the cap lowers member confidence, yes, but
as a regularizer smoothing also changes what the model does on *non-members* — and it pushed the
non-member mean down at least as much, so the difference between them grew rather than closed. I won't
over-claim the exact mechanism from one seed, but the robust reading is unambiguous: editing the target
moves *both* humps, and not in a way that brings them together — on this evidence, slightly apart. A
uniform, sample-blind intervention is not merely insufficient; it can be counterproductive. That kills
target-side humbling as a family, not just at `ε = 0.1`.

So re-derive the regularizer from the symptom rather than the cure. The disease shows up in the *output
distribution*: an overfit network puts almost all its softmax mass on one class — a low-entropy spike —
and that spike is exactly the high-confidence signal the attack reads. Smoothing medicated this by
editing the *target*, which acts identically on every example. What I want instead is to penalize the
spike *itself*, in proportion to how spiked each output is. Entropy measures the spike:
`H(p) = −Σ_i p_i log p_i`, maximal at uniform, zero at a one-hot spike. Over-confidence is low entropy;
the cure is to push entropy up. The headroom scales with `K`: the maximum entropy is `log K`, so
`log 10 = 2.303` nats on the 10-class problems but `log 100 = 4.605` on CIFAR-100 — twice the range for a
memorized spike to be pushed into, which cuts both ways (more room to flatten, but a longer road toward
the uniform attractor that a constant, undecayed push can travel too far down). So take the cross-entropy
I already minimize and *subtract* a multiple of the entropy:

  `L = L_CE − β·H(p)`.

Getting the sign right: I minimize `L`, and `−β·H` means that to make `L` small I want `H` large — less
peaked outputs — so I am penalizing low entropy, penalizing confidence. `β` controls how hard I push.

Is this surgical or a sledgehammer? The answer is in the entropy gradient. Using the softmax Jacobian
`∂p_j/∂z_i = p_j(δ_{ij} − p_i)` and `Σ_j p_j(log p_j + 1) = −H + 1`, it collapses to

  `∂H/∂z_i = p_i(−log p_i − H)`.

The quantity `−log p_i` is the surprisal of class `i` and `H` the *mean* surprisal, so the gradient on
logit `i` is the deviation of `i`'s surprisal from the mean, weighted by `p_i`. Since I optimize `−β·H`,
descent moves opposite the entropy gradient: for the class the model is confident about, `p_i` is large
and `−log p_i` is small (below the mean), so `∂H/∂z_i < 0`, the loss gradient is positive, and the update
pulls that logit *down*; for a nearly-killed class, `p_i ≈ 0` multiplies the whole term to negligible. So
this does not yank every dead class toward uniform — it acts mostly on the dominant, over-confident
class. Exactly the adaptivity smoothing lacked: smoothing forced every wrong class toward the same
`ε/K`, while the confidence penalty weights its pressure by the model's own current probabilities,
concentrating precisely on the spiked, memorized predictions and leaving the already-uncertain outputs
alone.

Put a number on the concentration. On a mildly spiked `p = [0.9, 0.05, 0.05]` (entropy 0.394) the entropy
gradient is `−0.260` on the dominant logit and `+0.130` on each near-dead one, so under `L_CE − βH` the
loss gradient presses the spike *down* at `0.260β` and lifts each dead class at `0.130β` — the spike
pushed down at twice the rate any single dead class is lifted, the opposite of smoothing's equal `1/K`
weight on all of them. That contrast is stark on CIFAR-100: smoothing's per-class weight there is
`1/K = 0.01`, whereas the confidence penalty's weight on a dominant class near `p ≈ 0.9` is `≈ 0.9` —
about 90× more pressure aimed exactly at the leaking spike. On the very benchmark where smoothing's grip
was diluted 10×, adaptivity is roughly 90× stronger on the offending class. If adaptivity is the missing
ingredient, this is where it should show.

One subtlety the same formula reveals matters for late training: the push `p_i(−log p_i − H)` does *not*
grow without bound as the spike sharpens — as `p_i → 1`, both `−log p_i → 0` and `H → 0`, so the
dominant-logit gradient vanishes. The penalty grips hardest on the *half-formed* spikes and eases off
once an output has already collapsed to near-one-hot. Double-edged: it is gentle on the most extreme
members (where the leak is worst), yet fights hardest during the confidence build-up — which foreshadows
both why it may under-defend the most memorized samples and why a *constant* coefficient applied through
the whole build-up can be destabilizing.

This is, formally, the *reverse* KL to uniform. Smoothing added the *forward* KL `D_KL(u‖p)`, whose
per-class weight is the constant `u_i = 1/K` — equal pressure on every class. The confidence penalty is
`−H(p)`, and `D_KL(p‖u) = −H(p) + log K`, so up to a constant it is the reverse KL, whose per-class
weight is the model's own `p_i` — large precisely on the classes the model is currently confident about.
Same target distribution (uniform), opposite KL direction, and the reversal is the formal reason one is
adaptive and the other uniform. Against a threshold attack I want adaptive: flatten the spikes that leak,
don't relocate every output.

An honesty check, because this is the rung where I learn whether *any* mean-control regularizer can break
the attack or only buy stability. The confidence penalty is still, at bottom, a device that lifts the
confidence floor by pushing spiked outputs toward uniform. It is more targeted than smoothing, so I
expect it to dent the attack where smoothing could not, especially on the easy/moderate benchmarks. But
it has no explicit lever on the *variance* of the member-confidence distribution — it compresses the high
end without deliberately spreading members to overlap the non-members. So I expect it to help, but to
remain a mean-region intervention; and the harder the dataset, the more I worry it will not be enough.

There is a sharper worry specific to CIFAR-100. In reinforcement learning the entropy bonus is wanted
*throughout* training; in supervised learning I want fast convergence on the easy examples and humility
only near the end, once memorization starts. A *constant* `−β·H` from epoch one is blunt: early it fights
the convergence I want, late it does its job. The richer version anneals `β` or replaces it with a hinge
`+β·max(0, Γ − H)` that switches on only once entropy drops below a threshold — but the scaffold edit here
is the *plain, fixed-`β`, unscheduled* form, `L_CE − β·H(p)` constant for all 300 epochs, and I should
reason against exactly that. The collapse mechanism is quantitative: the entropy term is minimized by the
uniform output, confidence `1/K`, so on CIFAR-100 its attractor sits at `0.01` — and if the penalty ever
wins the tug-of-war the argmax becomes an arbitrary tie-break and test accuracy falls to about
`1/K = 0.01`, chance. Not "accuracy degrades a bit" but a hard floor at 0.01. And the schedule sets that
up: the cross-entropy pull `p − y` shrinks as the model fits, and MultiStepLR cuts the step to 0.01 after
epoch 150 and 0.001 after 225, while `β = 0.1` is *never* decayed. A fixed-magnitude flattening force
against a fitting force the schedule is deliberately shrinking is the recipe for the flattening force to
win late. On CIFAR-10 the attractor is only `0.1` and the model fits easily, so there is margin; on
CIFAR-100, already at 0.5045 accuracy and barely holding 100 classes together, the same constant push has
far less margin before it drags outputs toward the 0.01 uniform point. Smoothing's strictly-positive
target never had this relief valve to lose; here I am trading that guaranteed safety for adaptivity, and
on the hardest benchmark that trade carries collapse risk.

I take the confidence penalty *now* rather than jumping to a variance lever because what smoothing
falsified is narrow: *uniform* mean-control fails. It never tested whether *adaptive* mean-control can
break the wall, and that is a genuinely different hypothesis with a real chance — the whole reason
smoothing failed (treating memorized and uncertain samples alike) is precisely what adaptivity fixes. So
the disciplined move is to spend one rung isolating the adaptivity variable: keep the target (uniform)
and the goal (raise entropy) identical to smoothing, change only forward KL to reverse KL, and read
whether that alone moves the AUC. The fixed-`β`, unscheduled form is the minimal, assumption-free version
of that experiment — whatever it shows is attributable to adaptivity, not to a schedule hidden inside it.

The knob `β` trades fitting against humility and is task-dependent, but the established default is
`β = 0.1`, held fixed across all three benchmarks with no schedule. Implementation is one extra reduction:
form the softmax `p`, take its log with a `1e-8` clamp so a near-dead class can't feed `log 0` into the
sum, form the per-sample entropy, average over the batch, and return `L_CE − β·H̄`. No extra forward or
backward pass, no auxiliary network, `epoch` unused — itself the design choice under test.

The expectations against the smoothing numbers. Because the pressure is now adaptive and aimed at the
spikes, I expect the attack AUC to *drop* where smoothing left it flat — on **resnet20-cifar10** below
smoothing's 0.7678 and plausibly below ERM's 0.7275, at roughly ERM-level accuracy, so privacy_score
recovering above smoothing's 0.5200: even if adaptivity merely undoes smoothing's damage and returns the
AUC to ERM's 0.7275 at accuracy ~0.79, the score is already `0.79 − 0.2275 ≈ 0.5625`, above smoothing,
and any further dent pushes past ERM's 0.5697. On **mobilenetv2-fmnist** a clean small win over 0.8460,
back toward ERM's 0.8685 — but sober about the ceiling, since FashionMNIST already sits close to the 0.5
corner (at most ~0.08 of margin to claw back) and once the AUC touches 0.5 any accuracy the penalty costs
is pure loss; the best case is a modest return, not a breakthrough. CIFAR-100 is where I am genuinely
unsure, and it forks: either a modest AUC dent below 0.8686 at preserved low accuracy, *or* the failure I
argued — the unscheduled entropy push prevents the 100-class model from committing, accuracy craters
toward chance, and the composite becomes meaningless. If CIFAR-100 collapses while the two easier
benchmarks improve, that is the decisive evidence: a fixed, unscheduled mean-control penalty is both
insufficiently powerful *and* unsafe on the hardest case, and the next rung must do two new things at once
— control the loss *level* without crushing it to a degenerate point, and deliberately *spread* the
member distribution so it overlaps the non-members, instead of only shifting its mean. The distilled
scaffold fill is in the answer.
