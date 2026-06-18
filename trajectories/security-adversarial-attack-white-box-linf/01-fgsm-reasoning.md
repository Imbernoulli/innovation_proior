I am filling exactly one function — `run_attack` — in a frozen white-box harness, and the scaffold
default returns the clean images, which by construction flips nothing: its ASR is zero. So the first
rung has a clean job: produce the cheapest attack that beats zero, the floor every later rung will be
measured against. I want the minimal, principled move, because the whole ladder above me is going to
be "this, but iterated / momentum-stabilized / made adaptive," and if I start from the wrong primitive
the entire climb inherits the mistake.

What is actually available to me through this interface? The harness hands me a fresh deep copy of the
model, the clean `images` in `[0,1]`, the `labels`, and `eps = 2/255`. White-box means I can run a
forward pass and call `torch.autograd.grad` to read the gradient of any scalar I build with respect to
the input pixels — one backward pass, the same machinery training uses, just stopped one step early at
the input instead of the weights. That is the entire toolkit: a forward pass and an input-gradient.
The only prior recipe I know for crafting an adversarial example is the existence-proof method — drive
`x` to a wrong label by minimizing `c·||r||_2 + loss_f(x+r, l)` over the box with box-constrained
L-BFGS and a line search over `c` until the label flips. It works and it is the reason I believe these
examples exist, but it runs an iterative constrained optimizer *with a line search* per single image.
Across 1000 samples per scenario and six scenarios, that is dead on arrival, and it gives me no cheap
closed-form move to build a ladder on. I want something I can compute in about the cost of one
backward pass.

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
eps` does not grow with `n`, but the output swing grows *linearly* in `n`. In high dimension I can
make a huge number of changes, each individually below the precision threshold — each invisible —
that add up coherently into one large swing because I aligned every coordinate with its weight.
Adversarial examples do not need curvature; they need *dimension* and *linearity*. That alone explains
the vulnerable linear softmax with no appeal to anything deep.

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
*corner* — a vector all of whose coordinates are `±eps`, i.e. `eps·sign(g)`. This is why it is the
*sign* and not the gradient itself. If the budget had been `L_2` the feasible set would be a ball, the
maximizer `eps·g/||g||_2`, and large-magnitude coordinates would soak up the budget. But under `L_inf`
each coordinate has its own independent allowance `eps`, and the right move is to spend each one fully
in its gradient's direction — magnitude is irrelevant, only the sign of each partial derivative
matters. That is the fast gradient sign step: one forward pass, one backward pass, an elementwise
sign, scale by `eps`.

Why is `L_inf` the right budget here in the first place, not merely the convenient one? Because of
feature precision, and because the task fixed it. An 8-bit image throws away everything below `1/255`
of its dynamic range; a per-pixel change smaller than the sensor can represent is semantically
meaningless by construction. The honest notion of imperceptible is therefore *per-coordinate* — no
single feature moves by more than the precision floor — which is exactly a max-norm constraint, and
the harness scores me against precisely that constraint, rejecting any sample where
`||x_adv − x||_inf` exceeds `eps`. The norm is not a mathematical convenience; it is the formalization
of "below the precision of the features," and it happens to give the cleanest answer.

One worry before I trust this on a real network: it is only a first-order method. I linearized `J` and
then took `η` as large as the whole budget allows — a full `eps` jump in every coordinate at once. The
true loss surface is curved; over a jump that big the linear model is approximate, so `x + eps·sign(g)`
maximizes the *approximation*, not the true loss inside the box. Could it just fail? Two things
reassure me. In the one case where the model genuinely is linear in `x` — logistic regression — there
is no approximation and the sign step is the exact worst case in the box. And my whole hypothesis is
that the nonlinear models are too linear to resist, so the first-order step *should* transfer; if it
reliably raises the loss in practice, that success is itself evidence for the linearity explanation.
The method and the hypothesis test each other. The cost of the worry is that the linearization wastes
budget on the harder, wider architectures: I expect the single step to underfit the model wherever the
loss bends fast away from its tangent plane — which is exactly the gap the next rung will attack by
iterating. So FGSM is the right *primitive* but a deliberately weak *adversary*; I want it as the
floor, not the finish.

Now, what does the harness force me to add beyond the bare math? The scaffold-level discipline. The
mathematical step is `x + eps·sign(g)`, but the harness rejects any sample that leaves `[0,1]` or
violates the budget, treating it as an attack failure. The clean reference single-step attack just
clamps the resulting image to `[0,1]`. I will be slightly more careful and project the *perturbation*
to `[-eps, eps]` first, then add it and clamp the image to `[0,1]`: with a single signed step of size
exactly `eps` the perturbation already lies in the box, so the delta-clamp is a no-op here, but it
makes the budget projection explicit and identical in form to the iterative rungs that follow, where
it stops being a no-op. I also do the gradient bookkeeping the interface needs: detach and clone the
inputs, set `requires_grad_`, build the cross-entropy with `F.cross_entropy`, take exactly one
`torch.autograd.grad`, and do the pixel arithmetic under `torch.no_grad`. `n_classes` and `device` go
unused for this attack — the gradient already knows the class count through the loss — so I bind them
to a throwaway to keep the signature honest. The full scaffold fill is in the answer.

What I expect, stated against the floor it must beat. The default returns clean images and scores
exactly 0 ASR; one signed gradient step must beat that comfortably on every scenario, because raising
a linearized loss by `eps·||g||_1` is a real push toward the boundary even when crude. But I expect a
clear *split by architecture*: on the bottlenecked ResNet and depthwise MobileNetV2, where small
`L_inf` perturbations propagate readily, the single step should already flip a large majority; on the
wider, shallower VGG11-BN, whose low-resolution feature maps absorb small per-pixel perturbations more
robustly, the single step should leave the most headroom — the clearest sign that one linearized jump
underfits the curved loss. That architecture-dependent shortfall, and the general gap between "raise
the tangent plane" and "raise the true loss," is precisely what the next rung must close by replacing
one big leap with many small re-linearized steps.
