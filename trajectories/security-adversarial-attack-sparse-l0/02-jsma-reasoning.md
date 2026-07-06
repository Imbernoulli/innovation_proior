The Pixle floor came back almost exactly as I feared, and the numbers tell me *why* in detail.
`asr = 0.0067` on `Rebuffi-R18-L2`, `0.020` on `Augustin-L2`, `0.0067` on `Engstrom-L2` — a mean of
about `0.011`. Let me read those as raw counts, because at this magnitude the fractions are literally
integers over the sample size and that is more honest than treating them as continuous rates. The harness
runs up to `150` correctly-classified samples per model, and `1/150 = 0.00667`, `3/150 = 0.020`. So Pixle
flipped *one* image on `Rebuffi`, *three* on `Augustin`, *one* on `Engstrom` — five successful attacks in
roughly `450` attempts. This is not noise around some real signal; it is a search that essentially never
found anything, sitting one or two counts off absolute zero. And the diagnosis is precise: Pixle spends
*no* information deciding which pixels to touch. With `pixel_mapping="random"` it copies an existing pixel
value onto a *uniformly random* destination, and with only ~15 evaluations per image it gets fifteen blind
throws. On an undefended network that is enough, because fragile pixels are everywhere; on these
`L2`-adversarially-trained models, whose whole training objective was to flatten the loss surface in a
neighborhood of each input, the fragile pixels are scarce and small, and fifteen random throws almost
never land on one. The Augustin model being the softest of the three (`3` flips versus `1`) is consistent
with that reading — it is the least robust target, so blind luck pays off three times instead of once —
but "three out of 150" against "one out of 150" is the difference between essentially-never and
essentially-never; the three-to-one ratio is real but it is a ratio of tiny integers, not a signal I can
build on. The lesson is unambiguous: against robust models I cannot guess locations. I have to *compute*
which pixels matter and spend my budget on them.

So the next rung has to introduce a per-pixel importance signal. The harness hands me full model access,
including gradients, so I no longer have to stay black-box — and now I should choose deliberately among the
ways a gradient can become a *sparse* attack, not just grab the first one. Three routes are on the table.
I could read a single loss gradient `grad_x J` and keep its top-`24` coordinates by magnitude — the
dense-attack reflex. I could run a differential-evolution or random search that only *evaluates* the
objective and never differentiates it. Or I could compute the full input-output Jacobian and build a
saliency map that asks, per pixel, a signed and class-aware question. The first route I can reject on the
spot with a shape argument: a single loss gradient collapses the whole `10`-dimensional output vector and
the label into one scalar `J` and hands me one direction over all `3072` input features — it tells me how
to move *every* pixel a little for a dense `L_inf` step, but it carries no per-class structure, so ranking
its coordinates by magnitude selects pixels that are sensitive *on average*, not pixels that specifically
trade true-class score for some other class. That is the wrong object for *selecting* a tiny support. The
second route, pure evaluative search, is exactly the black-box regime Pixle just came from; jumping
straight to it would skip the question of whether the gradient the harness now permits is worth anything
at all, and I want that measured before I abandon it. So the third route is the disciplined next step: use
the gradient, but through the object built for *selection* — the Jacobian `dF_j/dx_i`, the forward
derivative. It is per-feature *and* per-class, and crucially it keeps the *sign*. With it I can ask, for
each individual pixel, the targeted question that a single loss gradient cannot express: does pushing this
pixel raise some chosen class while dragging the others down? That is exactly the question a sparse,
label-flipping attack lives on.

Let me make sure I can actually compute the forward derivative under this threat model before I lean on
it, because if I cannot it is a fantasy. Each entry `dF_j/dx_i` is the derivative of an *output* (a
logit) with respect to an *input feature*, and the chain rule threads it through every layer: it is the
activation derivative at each neuron times the weighted sum of the previous layer's input-derivatives,
recursive in depth, with base case `dx/dx_i = e_i`. Every term — weights, biases, activations at the
current image, activation derivatives — is available with full model access, and in practice I do not
hand-roll the recursion: each row `j` of the Jacobian is one backward pass with the output seeded at
neuron `j`, so the whole `n_classes x num_features` matrix costs ten backward passes on CIFAR-10. That
is cheap in absolute terms, and it is exactly the white-box capability Pixle declined to use — but let me
cost the *whole* attack, not one Jacobian, because the greedy loop recomputes it. The network is
non-linear, so after saturating a pixel the sensitivities shift and I must rebuild the Jacobian; with the
budget forcing twelve iterations (I derive that count below), the attack spends `10 * 12 = 120` backward
passes per image. That is an order of magnitude more model work than Pixle's ~15 forward passes, and it is
the price of *directed* placement — I am paying `120` gradient evaluations to choose `24` pixels
deliberately instead of `15` forward passes to place them blind. Whether that trade pays is exactly what
this rung measures.

Now set up the targeted saliency precisely, because the both-signs structure is the whole point. To
flip the label toward a target class `t`, a useful feature, when I increase it, must do two things at
once: raise the target output (`dF_t/dx_i > 0`) and lower the rest (`sum_{j!=t} dF_j/dx_i < 0`). A
feature that fails either test is useless or counterproductive, so it scores zero; among features passing
both, the score is the product `(dF_t/dx_i) * |sum_{j!=t} dF_j/dx_i|`, large when target-help and
others-hurt are both large. Let me convince myself the product rather than the sum is right with a small
worked comparison, because it is the kind of choice that quietly wrecks a saliency map. Take two candidate
features. Feature `A` has `dF_t/dx_i = 5.0` and `sum_{j!=t} dF_j/dx_i = -0.1`: it shoves the target score
up hard but barely suppresses the competitors, so increasing it mostly inflates a class that is already
winning-adjacent while the true class stays high — a wasted pixel on a robust model. Feature `B` has
`dF_t/dx_i = 1.0` and `sum_{j!=t} dF_j/dx_i = -1.0`: a balanced trade that genuinely moves probability
mass from the field toward the target. Under a *sum* rule the scores would be `5.0 - 0.1 = 4.9` for `A`
against `1.0 - 1.0 = 0.0` for `B`, so the sum ranks the useless feature far above the useful one. Under
the *product* rule (with the sign gate already passed) they are `5.0 * 0.1 = 0.5` for `A` against
`1.0 * 1.0 = 1.0` for `B`, correctly preferring the balanced trade. The product demands *both* factors be
substantial, and the absolute value on the others-sum turns "more negative (more helpful)" into "higher
score" once the sign gate has passed. One subtlety pins down a real choice: the two conditions only carry
independent information if I differentiate the *logits*, not the softmax probabilities — the probabilities
sum to one, so raising one mechanically drops the rest (`sum_j dp_j/dx_i = 0` identically), which forces
`dp_t/dx_i > 0` to imply `sum_{j!=t} dp_j/dx_i < 0` for *free*, making the second gate vacuous, and the
softmax's saturated derivatives flatten the ranking on top of that. On the logits the outputs are
unconstrained, so requiring *both* signs genuinely selects the rare favorable features. And because a
single feature is rarely favorable on both axes — most pixels strongly help the target but slightly help a
competitor too, and get gated out — the saliency search modifies *two* features at a time, letting one
pixel's strongly-negative others-sum compensate the other's slightly-positive one. Each selected feature
is saturated to its extreme in one shot (`theta = +1`, increasing being more reliable than decreasing
because adding intensity is more confidently misclassified than removing it), then dropped from the search
domain because a saturated pixel has nothing left to give. Recompute the Jacobian every iteration, because
the network is non-linear and the sensitivities shift after every move, and stop when the prediction
reaches the target or the feature budget is spent.

Let me make the two-feature pairing concrete with numbers, because "one pixel compensates the other's
sign flaw" is the mechanism that keeps the gate from starving on a robust surface and I want to see it
work. On a flattened model very few single features clear both gates — many have `dF_t/dx_i > 0` but a
slightly *positive* others-sum, say a feature `C` with `dF_t/dx_C = 2.0` and
`sum_{j!=t} dF_j/dx_C = +0.3`, which fails the second gate alone. Pair it with a feature `D` that
over-suppresses the field, `dF_t/dx_D = 0.4` and `sum_{j!=t} dF_j/dx_D = -0.9`. Individually `C` is
rejected (positive others-sum) and `D` is a weak target-helper. But changing *both* moves the target by
`2.0 + 0.4 = 2.4` and the field by `+0.3 - 0.9 = -0.6` — a net favorable pair that neither member could
form alone. That is exactly why the search operates on pairs and why saturating them together, then
recomputing, is not the same as two independent greedy single-pixel steps: the pairing is what lets the
attack find leverage on a surface where singletons are almost all gated out. It also tells me where the
method will still fail — when *no* pair, not just no singleton, clears the combined gate, which is
precisely the local-optimum stall I expect on the hardest model.

It helps to size the saliency object once, because the whole per-iteration cost rides on it. The Jacobian
is `n_classes x num_features = 10 x 3072`, a `30720`-entry matrix rebuilt each of the twelve iterations.
From it, for a fixed target `t`, the target row `dF_t/dx_i` is one length-`3072` vector and the
others-sum `sum_{j!=t} dF_j/dx_i` is another (the column sums of the other nine rows), so the per-feature
saliency reduces to two length-`3072` vectors, a sign gate, and a product — cheap arithmetic once the ten
backward passes have produced the rows. The pair search is over the *surviving* features after gating,
which on a robust surface is a small set, so even the pairwise step is inexpensive. The dominant cost is
the twelve Jacobian rebuilds, `120` backward passes, which I already accounted for; everything downstream
is vector arithmetic on `3072`-length arrays. So the attack is gradient-heavy but not search-heavy: its
entire intelligence is in *which* two features it picks each step, and its entire fragility is that those
picks come from a first-order reading of a surface trained to lie to first-order readings.

Now the part that matters most for *this* task: how the harness actually configures JSMA, because it
diverges from the textbook in two consequential ways, and both are bug-fixes that a naive fill would get
wrong. First, `torchattacks.JSMA` is *targeted-only* by construction — it needs a target class per
sample. The textbook choice is a fixed shift like `(y+1) % n_classes`, but that is a weak target: forcing
the image toward an arbitrary neighbor class can be far harder than just pushing it off its own class, and
on a robust model "harder target" means "fails more." Let me reason about why least-likely is the right
untargeted proxy with a concrete picture of the ten logits. For a correctly-classified image the true
class sits at the top; the arbitrary neighbor `(y+1)%10` might be the *second*-highest logit — a class the
model already half-believes — or it might be somewhere in the middle. Aiming at a mid-pack class asks the
attack to overtake every class above it *and* stay ahead of those below, a needlessly constrained target.
The least-likely class is the one the model rates *lowest*, so its logit has the largest gap to close —
but here is the point: I do not actually need to *reach* it. My success metric is untargeted, so the image
is fooled the instant *any* class overtakes the true one. Driving mass toward the class the model is most
confident is wrong is the most aggressive way to *collapse the true class*, because the both-signs
saliency it induces systematically suppresses the true logit hardest, and along the way some easier class
usually overtakes first and triggers the untargeted flip before the least-likely target is ever reached.
So `set_mode_targeted_least_likely` turns a targeted-only primitive into the strongest untargeted proxy
available, and it works for any `n_classes`.

Second — and this is the failure mode the harness explicitly guards against — the `L0` budget. JSMA
counts in *feature* space: `torchattacks` computes `num_features = C*H*W = 3*32*32 = 3072` on CIFAR and
sets `max_iters = ceil(num_features * gamma / 2)`, modifying two features per iteration, so the total
features it can touch is `num_features * gamma`. A spatial pixel is counted as changed if *any* of its
three channels moves, so touching `num_features * gamma` features can cover up to that many *distinct
spatial pixels*. Let me walk the arithmetic both ways so the landmine is explicit. The fix the edit lands
is `gamma = pixels / (C*H*W) = 24 / 3072 = 0.0078125`, which gives
`max_iters = ceil(3072 * 0.0078125 / 2) = ceil(24/2) = 12` iterations, and `12 * 2 = 24` features touched
at most, which upper-bounds the distinct spatial pixels by `24` — exactly the budget, with the pixel count
possibly *lower* if two touched features share a spatial pixel. Now the careless version: suppose I
copy a constant `gamma` from a `28x28`-grayscale reference where "10 pixels of 1024" reads as
`gamma = 10/1024 = 0.009766`. Plug it into *this* task's `3072`-feature space and
`max_iters = ceil(3072 * 0.009766 / 2) = ceil(15) = 15` iterations touching `30` features, hence up to
`30` distinct spatial pixels — six over budget. The harness validates the `L0` count channel-wise *after*
the attack and rejects any sample exceeding `24` as an attack failure, so every over-budget sample scores
as a miss and the measured ASR collapses toward zero regardless of how well the saliency worked. Getting
`gamma` right is therefore not a tuning nicety; it is the difference between a valid attack and one the
harness throws out wholesale — and it is precisely the budget bookkeeping that a same-named fill would
miss. This is the literal scaffold edit: `JSMA(model, theta=1.0, gamma=pixels/num_features)`, then
`set_mode_targeted_least_likely(quiet=True)`, then `attack(images, labels)`. The full module is in the
answer.

One more design choice deserves its reasoning spelled out rather than asserted: `theta = +1`, saturating
each chosen feature *up* to its maximum rather than down to zero. A pixel channel lives in `[0,1]`, and at
a randomly-lit natural-image location the clean value is typically somewhere mid-range, so both directions
have room — but they are not symmetric in effect. Driving a channel to `1.0` injects a bright, high-
contrast impulse that a convolutional filter reads as a strong positive activation; driving it to `0.0`
mostly *removes* signal, and a removed pixel is easily inpainted by the surrounding context that
adversarial training taught the model to rely on. Empirically across sparse attacks, adding intensity
produces a more confident misclassification than removing it, so committing all chosen features to `+1`
extracts the most per-pixel leverage under a budget that charges nothing for magnitude. It also keeps the
saliency's sign convention consistent — the whole gate was written for *increasing* a feature — so
`theta = +1` is the choice that matches the derivation rather than fighting it.

I should also close off the tempting shortcut of simply *retuning Pixle* instead of switching methods,
because it is the obvious cheaper move and I want a computed reason to reject it. Pixle failed at five
flips in `450` attempts because it inspected only about three percent of locations blind; the naive fix is
to crank `restarts` and patch sizes until it covers more of the grid. But covering the grid by brute
sampling scales linearly in queries, and to inspect even half the `1024` locations I would need on the
order of hundreds of probes per image — and each probe would still be *blind*, testing a location with no
reason to believe it is fragile. JSMA, by contrast, reads the sensitivity of all `3072` features in ten
backward passes and spends its `24` pixels on the highest-scoring ones. So the comparison is not "more
blind probes versus fewer" but "blind coverage that scales linearly and never learns versus a directed
reading that ranks every feature at once." On a surface where fragile locations are a fraction of a
percent, directed ranking is the only route that could plausibly do better than luck, which is why the
next rung has to *compute* importance rather than sample harder.

So where does that leave my expectations against the Pixle floor? JSMA is a genuine step up on the axis
Pixle failed: it *uses* a per-pixel importance signal instead of guessing locations, and it uses the
gradient the harness now lets me read. It should beat `0.011` — placing 24 saturated pixels chosen by
forward-derivative saliency is a far better bet than fifteen random patch throws, and unlike Pixle's blind
three-percent coverage it inspects the sensitivity of *every* one of the `3072` features before spending a
single pixel. But I am not expecting a large number, and the reason is structural to *this* setting.
JSMA's saliency is a *first-order, local* quantity computed at the clean image, and it is greedy with no
backtracking: it commits to the best-scoring pair, saturates them, recomputes, and repeats, never
reconsidering a committed pixel. On a robust `L2` model the first-order signal is exactly what training was
hardened against — the gradient is small and the both-signs gate finds few features that clear it — so the
greedy walk is choosing among weak candidates and can stall in a local optimum where no next pair looks
good even though a different support would have worked. There is also a real risk that on the hardest
model the least-likely target is still too hard to reach within 12 iterations, in which case those samples
simply fail. My falsifiable expectation: JSMA's mean ASR clears the Pixle floor of `0.011` — anything that
*uses* saliency should — but lands low, somewhere in the few-percent range, and stays well short of a
search method that can escape local optima. Concretely I would expect single-digit counts per model — a
handful to a dozen flips out of `150` rather than Pixle's one-to-three — with the per-model ordering no
longer guaranteed to track Pixle's, since a first-order signal is suppressed differently by each model's
training than blind luck is. If it comes back only marginally above Pixle, the diagnosis for the next rung
is already written: greedy first-order saliency is too brittle on robust surfaces, and I need either a
richer gradient move that does not commit irrevocably, or a search that can reconsider.

The delta from the previous rung, concretely: where Pixle returned `attack(images, labels)` from a
random-mapping `Pixle` instance and scored `0.011`, this rung returns `attack(images, labels)` from a
`JSMA` instance with `gamma = pixels/(C*H*W)` and a least-likely target — trading a blind black-box
random search for a white-box, saliency-guided, budget-exact greedy support selection that spends `120`
backward passes to place its `24` pixels deliberately. I expect it to beat the floor by *using* the
gradient Pixle ignored, while exposing the next weakness — the brittleness of a greedy first-order signal
against models trained to flatten exactly that signal.
