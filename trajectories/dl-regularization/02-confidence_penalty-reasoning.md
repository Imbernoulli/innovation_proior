The DropBlock-inspired weight penalty came back exactly where I feared it would: 72.45 on
ResNet-56/CIFAR-100, 73.37 on VGG-16-BN/CIFAR-100, 94.69 on MobileNetV2/FashionMNIST. Read against
what I expected, every prediction landed. It is the floor of the field — the lowest of the three on both
hard CIFAR-100 pairs, and lowest again on the saturated FashionMNIST pair where everything crowds into
the mid-90s. The largest shortfall is exactly where I said it would be, on the harder CIFAR-100 models:
VGG-16-BN at 73.37 is the weakest single number, and that is the architecture whose real capacity sink —
the 512-wide dense classifier head — my penalty never even touched, because it only regularizes `Conv2d`
filters. ResNet-56 at 72.45 is barely a baseline-level number; the residual paths let the network route
around any filter-shape preference, and the penalty reached full strength only after the cosine schedule
had nearly frozen the weights. So the diagnosis is clean and it is a diagnosis about *location*: I was
acting on the spatial shape of convolutional weights, which is two steps removed from the failure, on a
pipeline that already shrinks those same weights with L2 and normalizes their activations with BN, and I
left the place where over-confidence actually shows up — the output — completely unregularized. The
penalty was too indirect to out-muscle the over-fitting it was supposed to fight. The fix is not a better
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
of why rung one's `lambda_max` was such a delicate guess against the L2 already on those weights. A
penalty on the output is invariant to the parameterization underneath, so one strength should travel
across ResNet, VGG, and MobileNetV2 far more cleanly than a weight-scale coefficient did. Second, the
output *contains* information I must not destroy: the probabilities the network assigns to the *wrong*
classes encode how it generalizes — a net that puts `1e-3` on a visually similar class and `1e-9` on an
unrelated one is genuinely better than one that flips those, because the ratios among the wrong-class
probabilities are real knowledge. So whatever I add must discourage the peaked spike without crushing
those wrong-class ratios. File that as a constraint.

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
mean, *weighted by `p_i`*. For the class the model is confident about, `p_i` is large and `− log p_i` is
small (below the mean), so the deviation is negative and the term pushes that logit down. For a class the
model has nearly killed, `p_i ≈ 0`, so even though its surprisal is huge, the whole thing is multiplied
by `p_i ≈ 0` and the push is negligible. So this does not yank every dead class up toward uniform
indiscriminately — it acts mostly on the dominant class, pulling it down, with action concentrated
wherever `p_i` is non-negligible, exactly the over-confident classes. It flattens toward uniform in
proportion to the mass each class already carries, leaving the long tail of near-zero wrong classes
essentially alone and *preserving their ratios* — the dark knowledge I told myself not to crush. That is
the surgical property the weight penalty could never have: it concentrates exactly on the over-confidence
that drove the gap.

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
ratios — the same behavior I read off the gradient.

Now the strength. The RL analogy also warns me where it differs: in RL you want the entropy bonus *on
throughout* because you never want to converge; in supervised learning I *do* want fast convergence on
the easy examples and humility only near the end, when memorization sets in. A very large constant `beta`
from step one would fight the convergence I want. But I just learned something from rung one's failure
about over-engineering the schedule: my DropBlock penalty was *all* schedule — a delayed 20% start and a
linear ramp — and the schedule is part of why it reached strength only when the weights had stopped
moving and had no leverage. So I do not want to repeat that mistake by wrapping this in an elaborate
anneal that arrives too late. The entropy penalty is far better-behaved than the weight penalty for a
reason the gradient already showed: early in training the outputs are *naturally* high-entropy (the
network is unsure), so `H` is already near its max and `max`-ish, the penalty `− beta H` is small in
effect and its gradient `p_i(− log p_i − H)` is tiny because no class dominates yet. The penalty
*self-attenuates* early and only bites once the outputs start to spike — it has a built-in schedule. So I
do not need an explicit warm-up at all; a fixed modest `beta`, applied every step, is weak exactly when I
want it weak (early, high entropy) and strong exactly when I want it strong (late, peaked outputs). That
is the opposite of rung one's problem, where the strength was decoupled from the dynamics.

What value of `beta`? It trades the data-fitting term against the humility term, and the right value is
task-dependent — but I want one value that travels across all three pairs, the way the parameterization-
invariance of the output argued it should. A modest `beta = 0.1` is the natural default: it is large
enough to register a real pressure on the spike but small enough not to flatten genuine confidence on
easy examples, and it sits at the conservative end of the range one would sweep on validation data. One
knob, the model's other hyperparameters untouched.

A note on the actual computation, because it has to drop into the loop cleanly and cheaply. I have the
logits `outputs` of shape `[B, K]`. The entropy is one extra reduction over logits I already computed for
the cross-entropy: form `probs = softmax(outputs)`, then `H = − sum_i p_i log p_i` per row, averaged over
the batch, and return `− beta H`. There is no extra forward or backward pass, no auxiliary network —
unlike the input-Jacobian or adversarial penalties, this is essentially free, which matters because it
runs every step on three full 200-epoch runs. (I will compute the log with a small `1e-8` floor inside
the log to keep it finite when a probability underflows toward zero; a peaked output is exactly when some
`p_i` is tiny.) The penalty reads only `outputs`, ignores `model`, `inputs`, `targets`, returns a scalar
on the right device, and changes nothing else in the pipeline. The full scaffold body is in the answer.

Now the falsifiable expectations against the numbers I just got. The mechanism is finally pointed at the
right place — the output over-confidence that the weight penalty could not reach — so I expect this rung
to *beat* DropBlock's floor, and to beat it most clearly where DropBlock failed hardest: VGG-16-BN, whose
over-fitting lives in a dense head my conv-only penalty ignored entirely but whose softmax this penalty
acts on directly. So I expect VGG-16-BN to move up from 73.37 by the most — call it a point or so, into
the mid-74s. On ResNet-56 I expect a gain over 72.45 as well, since the entropy penalty is parameterization-
invariant and does not care about residual routing, though residual nets are already well-regularized so
the gain may be smaller — into the low-to-mid 72s, comfortably above DropBlock's 72.45. On MobileNetV2 /
FashionMNIST I expect the smallest move from 94.69, because the task is nearly saturated and there is
little gap left to close — a slight bump at most, into the high-94s. If instead the entropy penalty
*failed* to beat DropBlock, or worse, flattened genuine confidence and dropped accuracy below the floor,
that would falsify my reading that output-distribution over-confidence is the live failure here and that
acting on it directly is the right move — and would push me back toward the weights. I do not expect
that; the gradient analysis says the penalty is surgical, concentrated on the spike, and the right place
to act. What I do expect is a real improvement over rung one, and a new question it cannot answer: the
entropy penalty fixes the *output* but says nothing about the *internal* conditioning of the network — the
spectrum of the weight matrices that decides how signal and gradient propagate. That is the next rung.
