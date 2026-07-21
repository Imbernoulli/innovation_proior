The DropBlock-inspired weight penalty came back exactly where I feared it would: 72.45 on
ResNet-56/CIFAR-100, 73.37 on VGG-16-BN/CIFAR-100, 94.69 on MobileNetV2/FashionMNIST. Read against
what I expected, every prediction landed. It is the floor of the field — the lowest of the three on both
hard CIFAR-100 pairs, and lowest again on the saturated FashionMNIST pair where everything crowds into
the mid-90s. Read as error rates the pattern is instructive: the CIFAR-100 pairs are missing `27.55%`
(ResNet) and `26.63%` (VGG) of the test set;
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

So where is the disease actually visible? When one of these networks over-fits, the symptom I can
literally see is in the output: an over-fit classifier puts almost all of its softmax mass on one class —
histogram the probabilities over the validation set and the mass piles up at 0 and 1, near-deterministic
spikes. Every regularizer reached for so far acts on the *insides* — weight decay on the weights, BN on
the activations, my DropBlock penalty on the filters — but the thing the network hands to the world is the
distribution `p(y|x)` that falls out of the softmax, and I am doing nothing to it. That is the blind spot
the prior art flagged, and VGG's 73.37 is precisely an output over-confidence I could not reach by
penalizing conv kernels. Acting on the output distribution directly should reach it.

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

There are other ways to "penalize confidence," a couple more literal, and I want to know why I pass them
over. The most literal is a hinge on the top probability: penalize `max_i p_i` above some threshold. But
`max` is non-smooth, only ever touches one class, and says nothing about the *shape* of the rest — it
would accept a `[0.5, 0.5, 0, ..., 0]` output as "unconfident" while that is still a collapsed,
information-poor prediction. Entropy is the smooth, whole-distribution version of the same intent. A
second option attacks the *cause* of peaking rather than the peak: over-confident outputs come from
large-magnitude logits, so penalize the logit norm, an L2 on `outputs`. I reject this on the dark-knowledge
constraint I just filed: shrinking the whole logit vector toward zero squashes every class's logit
uniformly, including the informative gaps between wrong classes, so it crushes exactly the ratios I
promised to preserve. The closest relative, label smoothing, is *forbidden* rather than merely inferior:
it rewrites the target, and this edit surface freezes the target. So the field of "act on the output"
candidates collapses onto the entropy term — smooth, whole-distribution, ratio-preserving, and additive.

So the candidate is almost embarrassingly direct: take the cross-entropy I already minimize and subtract
a multiple of the output entropy, `L = CE − beta H(p)`. The sign has to be right, since it is easy to
flip: I *minimize* `L`, and `− beta H` means making `L` small wants `H` large — high entropy, less-peaked
outputs, i.e. penalizing confidence. Since the loop adds my term onto a fixed cross-entropy,
`compute_regularization` returns `− beta H(p)` and the loop computes `CE + (− beta H)`, the objective I
want.

Is this surgical or a sledgehammer that flattens everything to uniform? What flows back is the gradient on
the logits. With the softmax Jacobian `∂p_j/∂z_i = p_j(δ_ij − p_i)`, the entropy gradient works out to
`∂H/∂z_i = p_i(− log p_i − H)` — the deviation of class `i`'s surprisal `−log p_i` from the mean surprisal
`H`, *weighted by `p_i`*. Put numbers on a peaked CIFAR-100-shaped output: one class at `p = 0.9`, the
other 99 sharing `0.1` (each `≈ 1.01e-3`), so `H ≈ 0.784` nats. The dominant logit's gradient is
`0.9·(0.105 − 0.784) ≈ −0.611` — a firm push *down* on the confident class; each near-dead class gets
`1.01e-3·(6.90 − 0.784) ≈ +0.0062`, a push up `~100×` smaller. So the term acts almost entirely on the one
over-confident logit and touches each of the ninety-nine suppressed classes with a hundredth of that
force. Two properties matter. The pushes balance — `Σ_i ∂H/∂z_i = H − H = 0`, so every bit of down-push on
the spike is redistributed across the tail, nothing leaks. And the fixed point is exactly uniform: at
`p_i = 1/K` all surprisals equal `log K = H`, so `∂H/∂z_i = 0` everywhere — the penalty pushes toward
uniform but has *no* gradient once there, so it cannot overshoot and invert the prediction. That is the
surgical property the weight penalty could never have: it concentrates on the over-confidence that drove
the gap, in proportion to the mass each class carries, and leaves the long tail of near-zero wrong classes
— the dark knowledge — essentially alone.

Where in training does the term do its work? The gradient vanishes at both fixed points — uniform, and a
perfect one-hot (the dominant `p_1·(−log p_1)` runs to `1·0`) — so the push is zero at both ends and
maximal between. The dominant-logit gradient across confidence: `≈ −1.15` at `p_max = 0.5`, `≈ −1.14` at
`0.7`, `≈ −0.61` at `0.9`, `≈ −0.09` at `0.99`. So it pushes *hardest* on outputs in the middle of
committing — the `0.5`-to-`0.7` band where a prediction is turning from unsure into confident — and eases
off both on early near-uniform outputs and on ones that have already collapsed to a near one-hot. That is
the right allocation: it resists the *approach* to over-confidence, where resistance still changes the
outcome, and does not waste force on an output whose battle is already lost. A CIFAR-100 net spends most
of training with outputs in that intermediate band, so the penalty is strong where it counts; the
fully-collapsed examples where it has let go are a minority.

I should place this against its closest relative, label smoothing, which the prior art listed as a gap
precisely because *this edit surface forbids it*. Label smoothing replaces the one-hot target with a
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

Now the strength — and whether it needs a schedule. RL wants the entropy bonus *on throughout* because it
never wants to converge; in supervised learning I *do* want fast convergence on easy examples and humility
only near the end, when memorization sets in, so a large constant `beta` from step one would fight the
convergence I want. My instinct is to reach for an anneal. But rung one just taught me the cost of
over-scheduling — its penalty was *all* schedule and reached strength only when the weights had stopped
moving — so I check whether I need one before bolting it on. At init the logits are near-random and small,
the softmax is near-uniform, `H ≈ log K`, and the entropy gradient `p_i(−log p_i − H)` is near zero
because no class dominates. So early in training the penalty produces almost no gradient on its own; it
self-attenuates exactly when a warm-up would have suppressed it, and develops a real gradient only once
outputs start to spike — precisely when I want it to bite. The term's *value* runs the opposite way
(largest early where `H` is large, smallest late), but the value does not move weights, the gradient does.
So the effective schedule is already built into the geometry of entropy; adding one would repeat rung
one's mistake of decoupling strength from the dynamics. A thresholded variant — penalize only once `H`
drops below `H_0` — is the same idea made explicit, and I pass on it: the `p_i`-weighting already delivers
the thresholding softly, so an explicit threshold is a redundant knob.

What value of `beta`, then? It trades the data-fitting term against the humility term, and I want one value
that travels across all three pairs, as parameterization-invariance argued it should. The magnitude it
induces: at init `−beta H ≈ −0.1·log 100 ≈ −0.46`, about a tenth of `CE ≈ 4.6` — a bounded offset whose
gradient is near zero, so it shifts the loss without steering the weights; as outputs peak, `H` falls
toward `~0.8` and the term shrinks to `~−0.08` while its gradient grows into the surgical push. A modest
`beta = 0.1` is
the natural default: large enough to register a real pressure on the spike but small enough not to flatten
genuine confidence on easy examples, and it sits at the conservative end of the range one would sweep on
validation data. One knob, the model's other hyperparameters untouched, no schedule and no threshold — the
simplest form the geometry allows.

The computation drops into the loop cheaply: from the logits `outputs` of shape `[B, K]`, form
`probs = softmax(outputs)`, then `H = − sum_i p_i log p_i` per row, mean over the batch, return `− beta H`.
The order of those reductions is not cosmetic — I sum over classes *first* for each example's own entropy,
then average over the batch. The alternative, averaging the softmaxes over the batch and taking the
entropy of *that* mean, measures something else and wrong for my purpose: it rewards class balance across
the batch and would penalize the network for being collectively certain that a mini-batch happens to
contain mostly one class. What I want to discourage is each *individual* prediction being a spike, which
is exactly the per-row entropy summed then meaned. There is no extra forward or backward pass, no
auxiliary network — unlike the input-Jacobian or adversarial penalties this is essentially free, which
matters at every step of three full 200-epoch runs. A `1e-8` floor inside the log keeps it finite when a
peaked output drives some `p_i` toward underflow, barely perturbing the healthy entries. The penalty reads
only `outputs`, returns a scalar on the right device, and changes nothing else. The full scaffold body is
in the answer.

Now the falsifiable expectations against the DropBlock floor. The mechanism is finally pointed at the
right place — the output over-confidence that the weight penalty could not reach — so I expect this to
*beat* that floor, and to beat it most clearly where DropBlock failed hardest and the leverage is
largest: VGG-16-BN, whose over-fitting lives in a dense head my conv-only penalty ignored entirely but
whose softmax this penalty acts on directly, and which is the CIFAR-100 pair with the most open error to
close. So the ordering I predict is VGG gaining most, clearly above its 73.37; ResNet-56 gaining too but
by less, since the entropy penalty is parameterization-invariant and does not care about residual routing
yet residual nets are already well-regularized; and MobileNetV2 / FashionMNIST moving least off 94.69,
the task being nearly saturated with only `5.31%` error left for a humility term to close. If instead the
penalty *failed* to beat DropBlock, or flattened genuine confidence and dropped below the floor, that
would falsify my reading that output over-confidence is the live failure and push me back toward the
weights. I do not expect that; the gradient analysis says the penalty is surgical, concentrated on the
spike, ratio-preserving. What it cannot answer is the *internal* conditioning of the network — the
spectrum of the weight matrices that decides how signal and gradient propagate. That is the next place
the prior art's menu points.
