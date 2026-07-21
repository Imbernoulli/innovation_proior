I am filling exactly one function — `run_attack` — in a frozen white-box harness, and the scaffold
default returns the clean images, which flips nothing: ASR zero. So the first job is the cheapest attack
that beats zero, the floor every stronger attempt is measured against. I want the minimal, principled move: a
single, closed-form, gradient-based step whose optimality I can state exactly, so that whatever
strengthens it later is visibly a correction to *one* named approximation rather than a pile of
unrelated tricks.

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

The linearity hypothesis also explains the cross-model agreement I used to reject the curvature story:
two independently trained near-linear classifiers that both generalize must align their weight vectors
with the same discriminative directions, so their input-gradients at a correct point are correlated and
`eps·sign(g^{(A)})` raises model B's loss too — exactly the observed "different models, same wrong
label," which uncorrelated curvature quirks could never produce. Here I attack and am graded by one
model, so transfer is not on the table; the phenomenon just corroborates that I have the right
primitive.

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

This is the linear-model problem again, with `g` in the role `w` played. By Hölder's inequality with
conjugate exponents `∞` and `1`, `η^T g ≤ ||η||_inf·||g||_1 ≤ eps·||g||_1`, with equality when every
nonzero-gradient coordinate has the same sign as `g_i` and is pushed to the boundary `|η_i| = eps`. So
a canonical maximizer is forced: `η = eps·sign(∇_x J(θ, x, y))`, achieving `η^T g = eps·||g||_1` —
closed form, no inner loop. The feasible set `{η : ||η||_inf ≤ eps}` is an axis-aligned cube, the
objective is linear, so the maximum sits at a *corner* — a vector all of whose coordinates are `±eps`,
i.e. `eps·sign(g)`.

The sign step looks almost arbitrary next to "just step along the gradient," so it is worth seeing why
the two obvious competitors lose. The scaled gradient `η = eps·g/||g||_∞` keeps the raw direction and
rescales to touch the box; the `L_2`-steepest step `η = eps·g/||g||_2` is the move for a ball budget.
On `g = (3,1)`, `eps = 1`: the sign step `(1,1)` gives `η^T g = 4 = eps·||g||_1`, the Hölder bound; the
scaled gradient `(1, 1/3)` gives `3.33`, wasting two-thirds of the second coordinate's allowance; the
`L_2` step `(0.95, 0.32)` gives `3.16`, worse, because it pours budget into the already-large
coordinate the box does not reward. Under `L_inf` each coordinate has its own independent allowance
`eps`, so the right move spends each one fully in its gradient's direction — magnitude is irrelevant,
only the sign matters. That is the fast gradient sign step: one forward pass, one backward pass, an
elementwise sign, scale by `eps`.

The `L_inf` box is the honest budget: the perturbation is loud in `L_2` (its size is
`eps·√3072 = 0.434`, a Euclidean displacement that would be glaring if concentrated in a few pixels)
but silent per-coordinate, smeared two-levels-thin across every pixel, and it is the per-coordinate
8-bit floor that perception and the harness's max-norm scoring enforce. The tight `2/255` is what keeps
attack quality measurable — under the looser `8/255` even a crude step flips almost everything.

The one honest caveat: this is first-order, and I take `η` as large as the whole budget — a full `eps`
jump — so `x + eps·sign(g)` maximizes the *linearization*, not the curved true loss. In the linear case
there is nothing to underfit: for binary logistic regression `J(x) = softplus(−y·w^T x)`, `∇_x J`'s sign
is `−y·sign(w)`, a constant direction, so the sign step is the *exact* worst case in the box with no
higher-order error — which is precisely the regime my hypothesis says the deep models approximate. Where
they curve, the single step will underfit, and that shortfall predicts where a stronger attack must push:
the bottlenecked ResNet20 and depthwise MobileNetV2 propagate a small coordinated `L_inf` perturbation
readily through narrow channels, so their tangent plane stays a good guide and I expect a large majority
flipped; the wider, shallower VGG11-BN spreads the same signal across broad low-resolution feature maps
that average out per-pixel perturbations, so its loss bends away from the clean-point tangent faster and
one full-budget jump lands high on the plane but not on the true loss — the most survivors. So FGSM is
the right *primitive* but a deliberately weak *adversary*: the floor, and the architecture-dependent gap
it leaves is the headroom a stronger adversary must reclaim.

Beyond the bare math, the harness forces the projection: it rejects any sample that leaves `[0,1]` or
violates `||x_adv − x||_inf ≤ eps + 1e-6`. So I clip the *perturbation* to `[-eps, eps]`, add it, then
clamp the image to `[0,1]`. For a signed step the perturbation clip binds at exactly the budget on every
coordinate, so it is the `[0,1]` clamp that does the visible work — a clean pixel at `0.003` cannot move
down the full `eps` without going negative, and the clamp silently returns a correct sub-budget move
there. Writing the budget clip as its own line keeps the guarantee legible and it is the projection the
iterative attacks will reuse. The scoring `asr = 1 − robust_acc` counts a flip to *any* wrong class, so I
want an *untargeted* attack — raise the true-label loss and let the argmax fall where the geometry sends
it — which cross-entropy ascent is exactly. The bookkeeping the interface needs: detach and clone the
inputs, `requires_grad_`, `F.cross_entropy`, one `torch.autograd.grad`, the pixel arithmetic under
`torch.no_grad`. `n_classes` and `device` go unused — the loss already knows the class count and the
tensors arrive on-device — so I bind them to a throwaway. The full fill is in the answer. Against the
zero floor one signed step must beat it comfortably everywhere, with the architecture split above the
signature I will be watching.
