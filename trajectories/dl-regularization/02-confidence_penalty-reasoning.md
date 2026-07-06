The DropBlock-inspired weight penalty came back exactly where I feared it would: 72.45 on
ResNet-56/CIFAR-100, 73.37 on VGG-16-BN/CIFAR-100, 94.69 on MobileNetV2/FashionMNIST. Read against
what I expected, every prediction landed. It is the floor of the field — the lowest of the three on both
hard CIFAR-100 pairs, and lowest again on the saturated FashionMNIST pair where everything crowds into
the mid-90s. Let me actually read the three numbers as error rates before I move on, because the pattern
is instructive. The CIFAR-100 pairs are missing `27.55%` (ResNet) and `26.63%` (VGG) of the test set;
FashionMNIST is missing `5.31%`. So the two CIFAR-100 tasks carry roughly *five times* the error of the
FashionMNIST task — there is an enormous generalization gap still open on CIFAR-100 and almost none left
on FashionMNIST, which is why the mid-90s number barely moves for anyone. That tells me where any real
regularizer has leverage: on the CIFAR-100 pairs, not on the near-ceiling FashionMNIST. And the `~0.9`
gap *between* the two CIFAR-100 numbers (VGG's 73.37 above ResNet's 72.45, despite ResNet being far
deeper) says the two hard pairs are failing for different reasons — VGG has more raw capacity to overfit
in its dense head, ResNet is deep and residual — so a single regularizer will not move them by the same
amount, and I should expect its gains to be uneven across the two.

The largest shortfall this rung showed is exactly where I said it would be, on the harder CIFAR-100
models: VGG-16-BN at 73.37 is the weakest single number, and that is the architecture whose real capacity
sink — the 512-wide dense classifier head — my penalty never even touched, because it only regularizes
`Conv2d` filters. ResNet-56 at 72.45 is barely a baseline-level number; the residual paths let the network
route around any filter-shape preference, and the penalty reached full strength only after the cosine
schedule had nearly frozen the weights. So the diagnosis is clean and it is a diagnosis about *location*: I
was acting on the spatial shape of convolutional weights, which is two steps removed from the failure, on
a pipeline that already shrinks those same weights with L2 and normalizes their activations with BN, and I
left the place where over-confidence actually shows up — the output — completely unregularized. The penalty
was too indirect to out-muscle the over-fitting it was supposed to fight. The fix is not a better
weight-shape penalty; it is to move the point of action to where the disease is visible.

So let me look at where the disease is actually visible. When one of these networks over-fits, the
symptom I can literally see is in the output: an over-fit classifier puts almost all of its softmax mass
on one class. Histogram the softmax probabilities over the validation set and the mass piles up at 0 and
1 — the outputs are near-deterministic spikes. Every regularizer I have reached for so far reaches into
the *insides* of the network — weight decay penalizes the weights, BN normalizes the activations, my
DropBlock penalty reshaped the filters — but the thing the network hands to the world is the distribution
`p(y|x)` that falls out of the softmax, and I am doing nothing to it. That is the blind spot the prior
art flagged and the one I just confirmed the hard way: I spent rung one acting on the weights and the
output stayed unregularized, so the over-confidence that drives the generalization gap was never
addressed. VGG's dense head over-fitting at 73.37 is precisely an *output* over-confidence I could not
reach by penalizing conv kernels. Acting on the output distribution directly should reach it.

Why is the output a good place to act, beyond "that is where the symptom shows"? Two reasons that are
not cosmetic, and that contrast sharply with the weight penalty I just ran. First, the output
distribution has a natural, fixed scale — it is a probability vector, it sums to one no matter what —
whereas the weights do not: the meaning of any single weight depends on every other weight, so a penalty
on the weights is entangled with how I happened to parameterize the network. That entanglement is part
of why rung one's `lambda_max` was such a delicate guess against the L2 already on those weights, and why
I had to divide by a layer count just to make one coefficient mean the same thing across a 55-layer
ResNet and a 13-layer VGG. A penalty on the output is invariant to the parameterization underneath: it
sees a length-`K` probability vector regardless of whether the net that produced it has 55 conv banks or
13, so one strength should travel across ResNet, VGG, and MobileNetV2 far more cleanly than a weight-scale
coefficient did. Second, the output *contains* information I must not destroy: the probabilities the
network assigns to the *wrong* classes encode how it generalizes — a net that puts `1e-3` on a visually
similar class and `1e-9` on an unrelated one is genuinely better than one that flips those, because the
ratios among the wrong-class probabilities are real knowledge. So whatever I add must discourage the
peaked spike without crushing those wrong-class ratios. File that as a hard constraint on the design.

What does "regularize the output distribution" concretely want? The over-fit network is *over-confident*
— its output is a low-entropy spike. The scalar that measures exactly this is entropy. For a softmax
output, `H(p(y|x)) = − sum_i p_i log p_i` is maximal at the uniform distribution and zero at a one-hot
spike. Over-confidence is low entropy; the cure is to push entropy up. And I have seen this move before,
just not in supervised classification. In reinforcement learning, when the policy is a softmax over
actions, people add the entropy of the policy to the objective — a `+ beta H(pi)` term — to stop the
policy from collapsing onto one action too early so it keeps exploring; Williams and Peng did it in 1991
and it is standard in modern policy-gradient training. "Premature convergence to a deterministic policy"
in RL is the same failure pattern as "an over-confident, low-entropy output distribution" in supervised
learning. The RL people add entropy to keep their softmax from spiking; I want to keep my softmax from
spiking. Same term, a setting where it transfers directly.

Before I commit to entropy I should walk the other ways to "penalize confidence," because a couple are
more literal and I want to know why I am passing them over. The most literal is a hinge on the top
probability: penalize `max_i p_i` whenever it exceeds some threshold. But `max` is non-smooth, it only
ever touches one class, and it says nothing about the *shape* of the rest of the distribution — it would
happily accept a `[0.5, 0.5, 0, ..., 0]` output as "unconfident" while that is still a collapsed,
information-poor prediction. Entropy is the smooth, whole-distribution version of the same intent, and it
carries the RL and KL pedigree that the max-hinge does not. A second option is to attack the *cause* of
peaking rather than the peak: over-confident outputs come from large-magnitude logits, so penalize the
logit norm, an L2 on `outputs`. I reject this on the dark-knowledge constraint I just filed: shrinking the
whole logit vector toward zero squashes *every* class's logit uniformly, including the informative gaps
between wrong classes, so it crushes exactly the ratios I promised to preserve. Entropy, as I am about to
show, does not. A third option is the closest relative and it is *forbidden* rather than merely inferior:
label smoothing, which I will place carefully in a moment, rewrites the target — and this edit surface
freezes the target. So the field of "act on the output" candidates collapses onto the entropy term, which
is smooth, whole-distribution, ratio-preserving, and additive.

So the candidate is almost embarrassingly direct: take the cross-entropy I already minimize and subtract
a multiple of the output entropy. As a total loss, `L = CE − beta H(p)`. Let me make sure the signs say
what I mean, because this is exactly the kind of thing that is easy to flip. I *minimize* `L`. The term
`− beta H` means that to make `L` small I want `H` large — high entropy, less-peaked outputs. So I am
penalizing low entropy, which is penalizing confidence. Since this edit surface adds my term onto a fixed
cross-entropy, my `compute_regularization` should return `− beta H(p)`: the loop computes `CE + (− beta
H)`, exactly the objective I want. The sign is right.

Is this surgical or a sledgehammer that flattens everything to uniform and destroys the model? Look at
what the entropy term does to the gradient on the logits, because that is what flows back. With the
softmax Jacobian `∂p_j/∂z_i = p_j(δ_ij − p_i)`, the entropy gradient works out to
`∂H/∂z_i = p_i(− log p_i − H)`. Sit with that. `− log p_i` is the surprisal of class `i`, and `H` is the
mean surprisal under `p`, so the gradient on logit `i` is the deviation of class `i`'s surprisal from the
mean, *weighted by `p_i`*. Let me not just assert the surgery — let me put numbers on it. Take a peaked
CIFAR-100-shaped output: one class at `p = 0.9`, the other 99 sharing `0.1` (each `≈ 1.01e-3`). Then
`H ≈ 0.784` nats. The gradient on the dominant logit is `0.9·(−log 0.9 − 0.784) = 0.9·(0.105 − 0.784) ≈
−0.611` — a firm push *down* on the confident class. The gradient on any one of the near-dead classes is
`1.01e-3·(6.90 − 0.784) ≈ +0.0062` — a push *up*, but `~100×` smaller. So the term acts almost entirely on
the one over-confident logit, pulling it down, and touches each of the ninety-nine suppressed classes with
a hundredth of that force. Two checks confirm this is a real gradient and not something that will drift.
First, the pushes must balance: `Σ_i ∂H/∂z_i = Σ_i p_i(−log p_i) − H·Σ_i p_i = H − H = 0`, and indeed
`−0.611 + 99·0.0062 ≈ 0` — every bit of down-push on the spike is redistributed across the tail, nothing
leaks. Second, the term has a fixed point exactly at uniform: at `p_i = 1/K` all surprisals equal `log K
= H`, so `∂H/∂z_i = 0` for every `i` — the penalty pushes toward uniform but has *no* gradient once it
gets there, so it cannot overshoot past uniform and invert the prediction. That is the surgical property
the weight penalty could never have: it concentrates exactly on the over-confidence that drove the gap,
in proportion to the mass each class already carries, and leaves the long tail of near-zero wrong classes
— the dark knowledge — essentially alone.

There is one more thing I should check about the gradient, because it decides where in training the term
does its work, and I would rather know it than assume it. I have two fixed points where the gradient
vanishes — uniform (all surprisals equal `H`) and a perfect one-hot (the dominant term `p_1·(−log p_1)`
runs to `1·0` and everything else is `≈ 0·large`). So the push is not monotone in confidence; it is zero
at both ends and maximal somewhere between. Putting numbers on the dominant-logit gradient across
confidence: at `p_max = 0.5` it is `≈ −1.15`, at `0.7` `≈ −1.14`, at `0.9` `≈ −0.61`, at `0.99` `≈ −0.09`.
So the term pushes *hardest* on outputs in the middle of committing — the `0.5`-to-`0.7` band where a
prediction is turning from unsure into confident — and eases off both on the early near-uniform outputs
and, notably, on the ones that have already fully collapsed to a near one-hot. That reads as exactly the
right allocation of pressure: it resists the *approach* to over-confidence, where resistance still changes
the outcome, and does not waste force pushing on an output whose battle is already lost. A training net on
CIFAR-100 spends most of its time with outputs in precisely that intermediate band before memorization
finishes, so the penalty is strong where it counts. The only honest caveat is the fully-collapsed tail:
on training examples the net has already driven to `p_max ≈ 0.99`, the term has nearly let go — but those
are a minority, and the aggregate effect is to hold the bulk of outputs off the spike rather than to rescue
the few that reached it.

I should place this against the closest existing thing, label smoothing, because if I cannot say why I am
not just reinventing it I have not understood my own term — and the prior art listed label smoothing as a
gap precisely because *this edit surface forbids it*. Label smoothing replaces the one-hot target with a
softened one and, up to a constant, adds the *forward* KL `D_KL(u || p)` to the loss; but it does so by
*changing the target / the loss itself*, which I cannot do here — the base cross-entropy is fixed. My
confidence penalty adds `− beta H(p)`, and computing the *reverse* KL to uniform gives
`D_KL(p || u) = − H(p) + log K`. So minimizing `D_KL(p || u)` is, up to the constant `log K`, exactly
minimizing `− H(p)` — my penalty *is* a KL toward uniform, with the direction reversed relative to label
smoothing, and crucially it is an *additive* term, not a modified loss, so it is the version that
actually fits this contract. The reversal is not a curiosity: in `D_KL(u || p)` the log-ratio is weighted
by the constant `u_i = 1/K`, equal fixed pressure on every class, forcing every wrong class toward the
same target; in `D_KL(p || u)` it is weighted by the model's own `p_i`, adaptive pressure concentrated on
the currently over-confident classes, with no target imposed on the wrong classes at all. That is the
formal reason it is adaptive where label smoothing is uniform, and the reason it preserves the wrong-class
ratios — the same `p_i`-weighting I just read off the gradient.

Now the strength, and here the arithmetic warns me about a trap that is easy to walk into. The RL analogy
says RL wants the entropy bonus *on throughout* because it never wants to converge; in supervised learning
I *do* want fast convergence on the easy examples and humility only near the end, when memorization sets
in. A very large constant `beta` from step one would fight the convergence I want. But I just learned
something from rung one's failure about over-engineering the schedule: my DropBlock penalty was *all*
schedule — a delayed 20% start and a linear ramp — and the schedule is part of why it reached strength
only when the weights had stopped moving and had no leverage. So my instinct is to reach for an anneal
here too, and I want to check whether I actually need one before I bolt it on. Look at the term at
initialization. At init the logits are near-random and small, so the softmax is near-uniform, `H ≈ log K`,
and — by the fixed-point check I just did — the entropy gradient `p_i(−log p_i − H)` is *near zero*
because no class dominates. So early in training the penalty produces almost no gradient on its own; it
`self-attenuates` exactly when I would have wanted a warm-up to suppress it, and it only develops a real
gradient once the outputs start to spike, which is precisely when I want it to bite. This is a subtle
point worth stating cleanly: the term's *value*, `−beta H`, is actually *largest* early (`H` is large) and
*smallest* late (`H` shrinks as outputs peak) — but the value does not move weights, the gradient does,
and the gradient runs the opposite way, near-zero early and largest late. So the effective schedule is
already built into the geometry of entropy; I do not need to add one, and adding one would repeat rung
one's mistake of decoupling strength from the dynamics. A thresholded variant — only penalize once `H`
drops below some `H_0` — is the same idea made explicit, and I pass on it for the same reason: the
`p_i`-weighting already delivers the thresholding softly, so an explicit threshold is a redundant knob
that buys nothing but a hyperparameter to tune.

What value of `beta`, then? It trades the data-fitting term against the humility term, and I want one value
that travels across all three pairs, the way the parameterization-invariance of the output argued it
should. Let me sanity-check the magnitude it induces: at init `−beta H ≈ −0.1·log 100 ≈ −0.46`, about a
tenth of the `CE ≈ 4.6` — a noticeable but bounded offset whose gradient I just showed is near zero, so it
shifts the loss without steering the weights; as outputs peak, `H` falls toward `~0.8` and the term
shrinks to `~−0.08` while its gradient grows into the surgical push I computed. A modest `beta = 0.1` is
the natural default: large enough to register a real pressure on the spike but small enough not to flatten
genuine confidence on easy examples, and it sits at the conservative end of the range one would sweep on
validation data. One knob, the model's other hyperparameters untouched, no schedule and no threshold — the
simplest form the geometry allows.

A note on the actual computation, because it has to drop into the loop cleanly and cheaply. I have the
logits `outputs` of shape `[B, K]`. The entropy is one extra reduction over logits I already computed for
the cross-entropy: form `probs = softmax(outputs)`, then `H = − sum_i p_i log p_i` per row, averaged over
the batch, and return `− beta H`. The order of those two reductions is not cosmetic and I want it right:
I sum over classes *first* to get each example's own entropy, then average over the batch — the mean of
per-example entropies. The tempting-looking alternative, averaging the softmaxes over the batch and taking
the entropy of *that* mean distribution, measures something different and wrong for my purpose: it rewards
class balance across the batch and would fight a legitimately imbalanced batch, penalizing the network for
being collectively certain that a mini-batch happens to contain mostly one class. What I want to discourage
is each *individual* prediction being an over-confident spike, which is exactly the per-row entropy summed
then meaned. So `probs.log`-weighted sum over `dim=-1`, then `.mean()` over the batch, in that order.
There is no extra forward or backward pass, no auxiliary network —
unlike the input-Jacobian or adversarial penalties, this is essentially free, which matters because it
runs every step on three full 200-epoch runs. I compute the log with a small `1e-8` floor inside the log
to keep it finite when a probability underflows toward zero; a peaked output is exactly when some `p_i` is
tiny, and `log(p_i + 1e-8)` stays finite there while barely perturbing the healthy entries. The penalty
reads only `outputs`, ignores `model`, `inputs`, `targets`, returns a scalar on the right device, and
changes nothing else in the pipeline. The full scaffold body is in the answer.

Now the falsifiable expectations against the numbers I just got. The mechanism is finally pointed at the
right place — the output over-confidence that the weight penalty could not reach — so I expect this rung
to *beat* DropBlock's floor, and to beat it most clearly where DropBlock failed hardest and where the
leverage is largest: VGG-16-BN, whose over-fitting lives in a dense head my conv-only penalty ignored
entirely but whose softmax this penalty acts on directly, and which is the CIFAR-100 pair with the most
open error to close. So I expect VGG-16-BN to move up from 73.37 by the most — call it a point or so, into
the mid-74s. On ResNet-56 I expect a gain over 72.45 as well, since the entropy penalty is parameterization-
invariant and does not care about residual routing, though residual nets are already well-regularized so
the gain may be smaller — into the low-to-mid 72s, comfortably above DropBlock's 72.45. On MobileNetV2 /
FashionMNIST I expect the smallest move from 94.69, because the task is nearly saturated — only `5.31%`
error left — and there is little gap for a humility term to close: a slight bump at most, into the
high-94s. If instead the entropy penalty *failed* to beat DropBlock, or worse, flattened genuine
confidence and dropped accuracy below the floor, that would falsify my reading that output-distribution
over-confidence is the live failure here and that acting on it directly is the right move — and would push
me back toward the weights. I do not expect that; the gradient analysis says the penalty is surgical,
concentrated on the spike, ratio-preserving, and pointed at the right place. What I do expect is a real
improvement over rung one, and a new question it cannot answer: the entropy penalty fixes the *output* but
says nothing about the *internal* conditioning of the network — the spectrum of the weight matrices that
decides how signal and gradient propagate. That is the next place the prior art's menu points.
