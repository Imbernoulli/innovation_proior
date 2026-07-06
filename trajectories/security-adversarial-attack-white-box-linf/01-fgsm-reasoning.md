I am filling exactly one function — `run_attack` — in a frozen white-box harness, and the scaffold
default returns the clean images, which by construction flips nothing: its ASR is zero. So the first
rung has a clean job: produce the cheapest attack that beats zero, the floor every later rung will be
measured against. I want the minimal, principled move, because every stronger
adversary I could build later is a refinement of this first primitive, and if I start from the wrong
primitive the entire climb inherits the mistake. The primitive has to be a single, closed-form,
gradient-based move whose optimality I can state exactly, so that whatever I do to strengthen it later
is visibly a correction to *one* named approximation rather than a pile of unrelated tricks.

What is actually available to me through this interface? The harness hands me a fresh deep copy of the
model, the clean `images` in `[0,1]`, the `labels`, and `eps = 2/255`. Let me pin that budget to a
number so I stop thinking of it as a symbol: `2/255 = 0.00784`, so every pixel may move by at most
about eight thousandths of its full `[0,1]` range — roughly two 8-bit levels. That is a tiny cube
around each clean image, and the design problem is entirely "what is the single best corner of that
cube to jump to." White-box means I can run a forward pass and call `torch.autograd.grad` to read the
gradient of any scalar I build with respect to the input pixels — one backward pass, the same
machinery training uses, just stopped one step early at the input instead of the weights. That is the
entire toolkit: a forward pass and an input-gradient. The only prior recipe I know for crafting an
adversarial example is the existence-proof method — drive `x` to a wrong label by minimizing
`c·||r||_2 + loss_f(x+r, l)` over the box with box-constrained L-BFGS and a line search over `c` until
the label flips. It works and it is the reason I believe these examples exist, but let me actually
count what it costs before I inherit it. Box-constrained L-BFGS needs on the order of tens of
forward/backward evaluations to converge, each wrapped in a line search that itself spends several
forward passes, and the whole thing is re-run for each value of `c` in the outer bisection — call it
optimistically a hundred forward/backward passes *per image*. The harness feeds up to 1000 samples in
each of six `(model, dataset)` scenarios: `1000 × 6 = 6000` attacks, so ~`6·10^5` network evaluations,
and every one is a fresh deep copy of the model. Against that, a single-gradient attack costs one
forward and one backward per *batch* of 100: `10` batches per scenario, `60` batches total, `60`
forward/backward passes for the whole benchmark. That is a four-orders-of-magnitude gap, and L-BFGS
also hands me no cheap closed-form move to build a ladder on. I want the `60`-pass method, not the
`6·10^5`-pass one.

So let me ask the prior question, because I suspect the cheap method and the *reason these examples
exist* are the same insight. Why do adversarial examples exist at all, and why does the same crafted
perturbation fool different models? The reflexive story is that deep nets are wildly nonlinear and
carve the input space into bizarre high-curvature pockets, and adversarial examples are needles hidden
in those pockets. But that story fights the evidence. If each model's blind spots were idiosyncratic
high-curvature pockets, a needle found for one model would not land in a needle for a *different* model
with different depth and weights — independent quirks do not line up like that — and yet they do, and
the models even agree on the wrong class. And a plain linear softmax classifier on raw pixels has the
same vulnerability, where there is no extreme nonlinearity to blame. The nonlinearity story is not
just unproven, it points the wrong way.

Let me take the opposite hypothesis seriously and work it out for the simplest model — a linear one —
to see if adversarial examples fall out of *linearity* rather than nonlinearity. A linear unit
computes `w^T x`. Feed it `x̃ = x + η`: the activation moves by exactly `w^T η`. What is the most
damage I can do to that activation under the constraint that no single pixel moves by more than the
budget — `||η||_inf ≤ eps`? I want to maximize `w^T η = Σ_i w_i η_i` with each `|η_i| ≤ eps`. The
terms decouple: each `w_i η_i` is maximized independently by pushing `η_i` to the corner of its
allowed interval in the direction of `w_i` — set `η_i = eps` if `w_i > 0`, `η_i = −eps` if `w_i < 0`.
That is `η = eps·sign(w)`, giving `w^T η = eps·Σ_i |w_i| = eps·||w||_1`. Stare at the scaling. With
`n` weight components of average magnitude `m`, `w^T η = eps·m·n`: the perturbation size `||η||_inf =
eps` does not grow with `n`, but the output swing grows *linearly* in `n`. Put the CIFAR number to it:
the input is `3 × 32 × 32 = 3072` pixels, so `n = 3072`, and the swing is `eps·m·3072 = 0.00784 · m ·
3072 ≈ 24.1·m` — the linear readout moves by about twenty-four times the *mean* weight magnitude even
though every pixel moved by less than one percent of its range. In high dimension I can make a huge
number of changes, each individually below the precision threshold — each invisible — that add up
coherently into one large swing because I aligned every coordinate with its weight. Adversarial
examples do not need curvature; they need *dimension* and *linearity*. That alone explains the
vulnerable linear softmax with no appeal to anything deep, and it says the vulnerability *grows* with
input dimension, which is why images — thousands of pixels — are such easy targets.

Before I lift this to the network I should settle *which* scalar I attack, because the choice quietly
determines the gradient. Three candidates sit in front of me: the negative true-class logit `−z_y`, a
margin `max_{i≠y} z_i − z_y`, or the training cross-entropy `J = −log p_y = −z_y + log Σ_j exp(z_j)`.
The bare logit `−z_y` ignores where the mass is going — it can push `z_y` down while some already-large
wrong logit sits unchanged, spending gradient on a class that will not become the argmax. The margin is
the sharpest object to attack because flipping the label is exactly making that margin negative, but at
the clean point `max_{i≠y} z_i` selects a single competitor and the `max` is non-differentiable where
two wrong logits tie, so its gradient can be jumpy. Cross-entropy is the smooth compromise: its input
gradient is `∇_x J = (p_y − 1)∇_x z_y + Σ_{i≠y} p_i ∇_x z_i`, a probability-weighted vote that lowers
the true logit and raises *every* wrong logit in proportion to how plausible it already is — a soft
version of the margin that automatically concentrates on the nearest competitors. It is also the loss
the model was trained on, so its geometry near `x` is the best-conditioned surface I have, and it needs
no class bookkeeping. For a single linearized step where I only want the steepest honest direction,
cross-entropy is the right primitive: smooth, well-conditioned near `x`, and the loss the model was
trained to have gradients for. So `J` is `F.cross_entropy(logits, labels)` and `g = ∇_x J`.

The linearity hypothesis also has to explain the cross-model agreement I used to reject the curvature
story, and it does so cleanly, which is a further check on it. Two independently trained near-linear
classifiers that both get high accuracy on the same data must both align their weight vectors with the
same discriminative directions of the input distribution — that is what "generalize well" means
geometrically. Their input-gradients at a correctly classified point are therefore correlated, not
independent, so `sign(g^{(A)})` and `sign(g^{(B)})` agree on a large majority of the `3072` coordinates
rather than the `~50%` two random sign vectors would share. A perturbation `eps·sign(g^{(A)})` then has
positive inner product with `g^{(B)}` as well, raises model B's loss too, and pushes both toward the
*same* dominant off-class — which is exactly the observed "different models, same wrong label." Under
the curvature story those signs would be uncorrelated and no such agreement could arise. So the same
linearity that hands me the closed form also predicts the transfer phenomenon that motivated it; here I
attack and am graded by one model, so I do not need transfer, but its consistency with my hypothesis is
reassurance that I have the right primitive.

Now the bridge to the real classifiers I am attacking — ResNet20, VGG11-BN, MobileNetV2. They are not
linear, but the modern building blocks are deliberately *near*-linear because that is what makes them
trainable: ReLU is piecewise linear, BatchNorm is affine, the convolutions are linear. So my working
hypothesis is that these models are too linear to resist a linear perturbation. If that holds, the
closed-form perturbation that is optimal for a linear model should also damage them, because near the
input they behave linearly. So I want to do for the classification loss `J(θ, x, y)` what I just did
for a single linear activation. The thing I am attacking is the scalar loss the model was trained
with, and I want to *raise* it — raising the loss is pushing the logits away from the true label. I do
not have a global linear form for `J`, but I have its first-order behavior around the current input.
Taylor-expand the loss in the input: `J(θ, x + η, y) ≈ J(θ, x, y) + η^T ∇_x J(θ, x, y)`. The part I
control is `η^T g` with `g = ∇_x J` — one input-gradient, exactly the `torch.autograd.grad` the
harness lets me call. Maximize the linearized increase under the per-pixel budget:
`max_{||η||_inf ≤ eps} η^T g`.

This is the linear-model problem again, with `g` in the role `w` played. I can solve it coordinate by
coordinate as before, but the clean general statement is worth stating because it tells me the norm is
doing real work. By Hölder's inequality with conjugate exponents `∞` and `1`,
`η^T g ≤ ||η||_inf·||g||_1 ≤ eps·||g||_1`, with equality when every nonzero-gradient coordinate has
the same sign as `g_i` and is pushed to the boundary `|η_i| = eps`. So a canonical maximizer is forced:
`η = eps·sign(∇_x J(θ, x, y))`, achieving `η^T g = eps·||g||_1`. There it is in closed form, no inner
loop. The geometry, said out loud so I do not reach for the wrong thing: the feasible set
`{η : ||η||_inf ≤ eps}` is an axis-aligned cube, the objective is linear, so the maximum sits at a
*corner* — a vector all of whose coordinates are `±eps`, i.e. `eps·sign(g)`.

Let me make sure I am not fooling myself that the sign is genuinely better than the two obvious
competitors, because this is the choice the whole ladder rests on and "sign" looks almost arbitrary
next to "just step along the gradient." Two candidates are tempting. The first is the plain scaled
gradient, `η = eps·g/||g||_∞`, which keeps the raw gradient direction and just rescales it to touch
the box. The second is the `L_2`-steepest step, `η = eps·g/||g||_2`, the move I would take if the
budget were a ball. Let me run both against the sign step on a tiny concrete case so the difference is
a number, not a vibe. Take two pixels, `g = (3, 1)`, and `eps = 1`. The sign step is `η = (1, 1)` with
`η^T g = 3 + 1 = 4 = eps·||g||_1`, the Hölder bound met exactly. The scaled-gradient step must not
exceed `1` on either coordinate, so I divide `g` by its max component `3`: `η = (1, 1/3)`, giving
`η^T g = 3 + 1/3 = 3.33` — it wastes the second coordinate, spending only a third of its allowance,
and loses `17%` of the achievable increase. The `L_2` step `η = (3,1)/√10 = (0.95, 0.32)` gives
`η^T g = 10/√10 = 3.16`, worse still, because it pours budget into the already-large coordinate that
the box does not reward. So the ranking is `4 > 3.33 > 3.16`: under the `L_inf` box the sign step is
strictly optimal and the intuitive gradient-following steps leave budget on the table. That is why it
is the *sign* and not the gradient itself. If the budget had been `L_2` the ball's maximizer
`eps·g/||g||_2` would be correct and large-magnitude coordinates *should* soak up the budget; but under
`L_inf` each coordinate has its own independent allowance `eps`, and the right move is to spend each
one fully in its gradient's direction — magnitude is irrelevant, only the sign of each partial
derivative matters. That is the fast gradient sign step: one forward pass, one backward pass, an
elementwise sign, scale by `eps`.

It is worth checking the "imperceptible" claim as an actual number rather than an assertion, because
the whole justification for the `L_inf` box rests on it. At `eps = 2/255` the largest change any pixel
can undergo is `2` out of `255` 8-bit levels — two clicks on a channel that has `256` of them. The
coordinated part is that *all* `3072` pixels move by (up to) that amount at once, each toward its own
gradient sign, so the aggregate `L_2` size of the perturbation is as large as `eps·√3072 = 0.00784 ·
55.4 = 0.434` — a Euclidean displacement that would be very visible if it were concentrated in a few
pixels, but is instead smeared two-levels-thin across every pixel and reads as imperceptible dither.
That is the dimension amplification made tactile: the attack is loud in `L_2` and silent in `L_inf`,
and `L_inf` is the norm human perception and 8-bit storage actually enforce. It is exactly the gap
between those two norms that FGSM converts into an output swing.

Why is `L_inf` the right budget here in the first place, not merely the convenient one? Because of
feature precision, and because the task fixed it. An 8-bit image throws away everything below `1/255`
of its dynamic range; a per-pixel change smaller than the sensor can represent is semantically
meaningless by construction. At `eps = 2/255` I am spending exactly two of those quantization levels
per pixel — right at the floor of what a display can even show. The honest notion of imperceptible is
therefore *per-coordinate* — no single feature moves by more than the precision floor — which is
exactly a max-norm constraint, and the harness scores me against precisely that constraint, rejecting
any sample where `||x_adv − x||_inf` exceeds `eps`. The norm is not a mathematical convenience; it is
the formalization of "below the precision of the features," and it happens to give the cleanest answer.
It also explains why the context fixes the *tight* `2/255` and calls the looser `8/255` saturated: at
`8/255` the box is four times wider per pixel, the linearized swing `eps·||g||_1` quadruples, and even
a crude single step flips almost everything, so attack quality stops being measurable. At `2/255` the
budget is tight enough that how well I spend it actually shows up in the score.

One worry before I trust this on a real network: it is only a first-order method. I linearized `J` and
then took `η` as large as the whole budget allows — a full `eps` jump in every coordinate at once. The
true loss surface is curved; over a jump that big the linear model is approximate, so `x + eps·sign(g)`
maximizes the *approximation*, not the true loss inside the box. Could it just fail? Let me check the
one case where I can compute the answer exactly. For binary logistic regression the loss is genuinely
of the form `J(x) = softplus(−y·w^T x)` with `∇_x J = −y·σ(−y·w^T x)·w`, whose sign is `−y·sign(w)`
(the scalar `σ(·) > 0` and `−y·(·)` do not depend on `x`) — a constant direction independent of where
`x` sits. So the sign step is `eps·sign(∇_x J) = −eps·y·sign(w)`, the perturbation moves `w^T x` by
exactly `−y·eps·||w||_1` regardless of step size, the Taylor "approximation" has no higher-order error
in the direction that matters, and the sign step is the *exact* worst case in the box. There is no
curvature to underfit in the linear case, which is precisely the regime my hypothesis says the deep
models approximate. And the method and the hypothesis test each other: if the single step reliably
raises the loss on the real nets, that success is itself evidence for the linearity explanation; if it
underfits, the shortfall localizes exactly where curvature is strongest.

That shortfall is worth predicting concretely, because it tells me where a stronger attack would have
to push. The linearization wastes budget wherever the loss bends fast away from its tangent plane over
an `eps` jump. Architecturally, the bottlenecked ResNet20 and the depthwise-separable MobileNetV2 propagate a
small coordinated `L_inf` perturbation readily through their narrow channels, so the tangent plane
stays a good guide across the whole cube and I expect the single step to flip a large majority there.
The wider, shallower VGG11-BN spreads the same signal across broad, low-resolution feature maps that
average out small per-pixel perturbations, so its loss curves away from the clean-point tangent faster
and one full-budget jump lands at a corner that is high on the *plane* but not on the *true* loss —
leaving the most survivors there. So FGSM is the right *primitive* but a deliberately weak *adversary*:
the floor, not the finish, and the architecture-dependent gap it leaves is exactly the headroom a
stronger adversary would have to reclaim.

Now, what does the harness force me to add beyond the bare math? The scaffold-level discipline. The
mathematical step is `x + eps·sign(g)`, but the harness rejects any sample that leaves `[0,1]` or
violates the budget `||x_adv − x||_inf ≤ eps + 1e-6`, treating it as an attack failure. The clean
reference single-step attack just clamps the resulting image to `[0,1]`. The projection I write is: clip
the *perturbation* to `[-eps, eps]`, add it to the image, then clamp the image to `[0,1]`. For my signed
step the perturbation clip binds at exactly the budget — `|eps·sign(g)_i| = eps` for every coordinate,
right on the boundary — so it is the `[0,1]` clamp that does the visible work here, pulling back any
pixel the step would have driven below `0` or above `1`. Writing the budget clip as its own explicit
line rather than trusting that a step of magnitude `eps` cannot overshoot keeps the guarantee legible:
the clip to `[-eps, eps]` is exactly the line that enforces `||x_adv − x||_inf ≤ eps`, the constraint
the harness checks, and I would rather see it stated than inferred. The `[0,1]` clamp, by contrast, can bite even here: a clean pixel already
at `0.003` cannot move down by the full `eps = 0.00784` without going negative, so the clamp silently
returns a sub-budget move on boundary pixels, which is correct and costs nothing. One more framing the harness fixes for me:
it scores `asr = 1 − robust_acc`, counting a sample flipped the moment the prediction is *any* wrong
class, so I want an *untargeted* attack — raise the loss of the true label and let the argmax fall
wherever the geometry sends it, rather than steering toward a chosen wrong class and spending budget on
a specific target the scoring does not reward. Cross-entropy ascent is exactly untargeted in this sense.
I also do the
gradient bookkeeping the interface needs: detach and clone the inputs, set `requires_grad_`, build the
cross-entropy with `F.cross_entropy`, take exactly one `torch.autograd.grad`, and do the pixel
arithmetic under `torch.no_grad` so the projection is not itself recorded on the graph. `n_classes` and
`device` go unused for this attack — the gradient already knows the class count through the loss, and
the tensors arrive on-device — so I bind them to a throwaway to keep the signature honest. The full
scaffold fill is in the answer.

What I expect, stated against the floor it must beat. The default returns clean images and scores
exactly 0 ASR; one signed gradient step must beat that comfortably on every scenario, because raising
a linearized loss by `eps·||g||_1 ≈ 24·(mean gradient)` is a real push toward the boundary even when
crude. But I expect a clear *split by architecture*: on the bottlenecked ResNet and depthwise
MobileNetV2 the single step should already flip a large majority; on the wider, shallower VGG11-BN the
single step should leave the most headroom — the clearest sign that one linearized jump underfits the
curved loss. That architecture-dependent shortfall, and the general gap between "raise the tangent
plane" and "raise the true loss," is precisely the headroom a stronger adversary must reclaim, and the
VGG columns are where I will be watching for it.
