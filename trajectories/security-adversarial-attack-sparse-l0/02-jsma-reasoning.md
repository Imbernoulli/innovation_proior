The Pixle floor came back almost exactly as I feared, and the numbers tell me *why* in detail.
`asr = 0.0067` on `Rebuffi-R18-L2`, `0.020` on `Augustin-L2`, `0.0067` on `Engstrom-L2` — a mean of
about `0.011`. That is one or two successes out of 150 on two of the three models, three on the third.
This is not noise around some real signal; it is a search that essentially never found anything. And the
diagnosis is precise: Pixle spends *no* information deciding which pixels to touch. With
`pixel_mapping="random"` it copies an existing pixel value onto a *uniformly random* destination, and
with only ~15 evaluations per image it gets fifteen blind throws. On an undefended network that is
enough, because fragile pixels are everywhere; on these `L2`-adversarially-trained models, whose whole
training objective was to flatten the loss surface in a neighborhood of each input, the fragile pixels
are scarce and small, and fifteen random throws almost never land on one. The Augustin model being the
softest of the three (`0.020` versus `0.0067`) is consistent with that reading — it is the least robust
target, so blind luck pays off slightly more often — but "slightly more often than essentially never" is
still essentially never. The lesson is unambiguous: against robust models I cannot guess locations. I
have to *compute* which pixels matter and spend my budget on them.

So the next rung has to introduce a per-pixel importance signal. The harness hands me full model access,
including gradients, so I no longer have to stay black-box — and the cheapest way to turn a gradient into
a *sparse* attack is the saliency-map idea. The reflex from the dense world is to read one loss gradient,
but that collapses the whole output vector and the label into a single scalar `J` and gives me one
direction over all pixels — useful for a dense `L_inf` step, useless for *selecting* a tiny support. What
I actually want, and what the saliency construction keeps, is the full sensitivity of *every* output to
*every* input feature: the Jacobian `dF_j/dx_i`, the forward derivative. It is per-feature *and*
per-class, and crucially it keeps the *sign*. With it I can ask, for each individual pixel, the targeted
question that a single loss gradient cannot express: does pushing this pixel raise some chosen class while
dragging the others down? That is exactly the question a sparse, label-flipping attack lives on.

Let me make sure I can actually compute the forward derivative under this threat model before I lean on
it, because if I cannot it is a fantasy. Each entry `dF_j/dx_i` is the derivative of an *output* (a
logit) with respect to an *input feature*, and the chain rule threads it through every layer: it is the
activation derivative at each neuron times the weighted sum of the previous layer's input-derivatives,
recursive in depth, with base case `dx/dx_i = e_i`. Every term — weights, biases, activations at the
current image, activation derivatives — is available with full model access, and in practice I do not
hand-roll the recursion: each row `j` of the Jacobian is one backward pass with the output seeded at
neuron `j`, so the whole `n_classes x num_features` matrix costs ten backward passes on CIFAR-10. That
is cheap, and it is exactly the white-box capability Pixle declined to use.

Now set up the targeted saliency precisely, because the both-signs structure is the whole point. To
flip the label toward a target class `t`, a useful feature, when I increase it, must do two things at
once: raise the target output (`dF_t/dx_i > 0`) and lower the rest (`sum_{j!=t} dF_j/dx_i < 0`). A
feature that fails either test is useless or counterproductive, so it scores zero; among features passing
both, the score is the product `(dF_t/dx_i) * |sum_{j!=t} dF_j/dx_i|`, large when target-help and
others-hurt are both large. The product, not a sum, is deliberate: a sum would let a huge target-
derivative paper over a near-zero others-effect, selecting a feature that pushes the target up while
doing nothing to suppress the competitors — which on a robust model is precisely a feature that wastes
one of my 24 pixels. The product demands *both* be substantial, and the absolute value on the others-sum
turns "more negative (more helpful)" into "higher score" once the sign gate has already passed. One subtlety pins down a real choice: the two conditions only carry
independent information if I differentiate the *logits*, not the softmax probabilities — the probabilities
sum to one, so raising one mechanically drops the rest and the gate becomes vacuous, and the softmax's
saturated derivatives flatten the ranking. On the logits the outputs are unconstrained, so requiring
*both* signs genuinely selects the rare favorable features. And because a single feature is rarely
favorable on both axes — most pixels strongly help the target but slightly help a competitor too, and get
gated out — the saliency search modifies *two* features at a time, letting one pixel's strongly-negative
others-sum compensate the other's slightly-positive one. Each selected feature is saturated to its
extreme in one shot (`theta = +1`, increasing being more reliable than decreasing because adding
intensity is more confidently misclassified than removing it), then dropped from the search domain
because a saturated pixel has nothing left to give. Recompute the Jacobian every iteration, because the
network is non-linear and the sensitivities shift after every move, and stop when the prediction reaches
the target or the feature budget is spent.

Now the part that matters most for *this* task: how the harness actually configures JSMA, because it
diverges from the textbook in two consequential ways, and both are bug-fixes that a naive fill would get
wrong. First, `torchattacks.JSMA` is *targeted-only* by construction — it needs a target class per
sample. The textbook choice is a fixed shift like `(y+1) % n_classes`, but that is a weak target: forcing
the image toward an arbitrary neighbor class can be far harder than just pushing it off its own class,
and on a robust model "harder target" means "fails more." The harness instead calls
`set_mode_targeted_least_likely`, which picks, per sample, the class the model currently rates *least*
likely as the attack target. That sounds backwards until you see it: a *least-likely* target is the
strongest possible *untargeted* proxy here, because driving probability mass toward the class the model is
most confident is wrong is the most aggressive way to collapse the true class. It is the right adaptation
of a targeted-only primitive to an untargeted success metric, and it works for any `n_classes`.

Second — and this is the failure mode the harness explicitly guards against — the `L0` budget. JSMA
counts in *feature* space: `torchattacks` computes `num_features = C*H*W = 3*32*32 = 3072` on CIFAR and
sets `max_iters = ceil(num_features * gamma / 2)`, modifying two features per iteration, so the total
features it can touch is `num_features * gamma`. A spatial pixel is counted as changed if *any* of its
three channels moves, so touching `num_features * gamma` features can cover up to that many *distinct
spatial pixels*. If `gamma` is set carelessly — say a constant meant as "10/1024" — the attack can
perturb ~30 features and therefore up to ~30 distinct pixels, blowing past the budget; the harness then
rejects *every* such sample as invalid and ASR collapses to zero. The fix the edit lands is
`gamma = pixels / (C*H*W) = 24 / 3072`, so `max_iters = ceil(24/2) = 12` iterations, `12 * 2 = 24`
features touched at most, which upper-bounds the distinct spatial pixels by 24 — exactly the budget. This
is the literal scaffold edit: `JSMA(model, theta=1.0, gamma=pixels/num_features)`, then
`set_mode_targeted_least_likely(quiet=True)`, then `attack(images, labels)`. The full module is in the
answer. Getting `gamma` right is not a tuning nicety here; it is the difference between a valid attack and
one the harness throws out wholesale — and it is precisely the budget bookkeeping that a same-named paper
fill would miss.

So where does that leave my expectations against the Pixle floor? JSMA is a genuine step up on the axis
Pixle failed: it *uses* a per-pixel importance signal instead of guessing locations, and it uses the
gradient the harness now lets me read. It should beat `0.011` — placing 24 saturated pixels chosen by
forward-derivative saliency is a far better bet than fifteen random patch throws. But I am not expecting a
large number, and the reason is structural to *this* setting. JSMA's saliency is a *first-order, local*
quantity computed at the clean image, and it is greedy with no backtracking: it commits to the
best-scoring pair, saturates them, recomputes, and repeats, never reconsidering a committed pixel. On a
robust `L2` model the first-order signal is exactly what training was hardened against — the gradient is
small and the both-signs gate finds few features that clear it — so the greedy walk is choosing among weak
candidates and can stall in a local optimum where no next pair looks good even though a different support
would have worked. There is also a real risk that on the hardest model the least-likely target is still
too hard to reach within 12 iterations, in which case those samples simply fail. My falsifiable
expectation: JSMA's mean ASR clears the Pixle floor of `0.011` — anything that *uses* saliency should —
but lands low, somewhere in the few-percent range, and stays well short of a search method that can
escape local optima. If it comes back only marginally above Pixle, the diagnosis for the next rung is
already written: greedy first-order saliency is too brittle on robust surfaces, and I need either a richer
gradient move that does not commit irrevocably, or a search that can reconsider.

The delta from the previous rung, concretely: where Pixle returned `attack(images, labels)` from a
random-mapping `Pixle` instance and scored `0.011`, this rung returns `attack(images, labels)` from a
`JSMA` instance with `gamma = pixels/(C*H*W)` and a least-likely target — trading a blind black-box
random search for a white-box, saliency-guided, budget-exact greedy support selection. I expect it to beat
the floor by *using* the gradient Pixle ignored, while exposing the next weakness — the brittleness of a
greedy first-order signal against models trained to flatten exactly that signal.
