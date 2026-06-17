Let me start from the inner loop, because that is the one thing I get to design and the one thing the
dominant method leaves dumbest. The frame I am working in is settled and I like it: meta-learn a shared
initialization `θ`, adapt to each new task by a few gradient steps on its support set, and train `θ` so
that the *query* loss after adaptation is small across tasks, differentiating the post-adaptation loss
back through the inner step into `θ`. It scales, it is architecture-agnostic, it works. But the inner
update is plain SGD: `θ' = θ − α ∇L`, one global scalar `α`, and the step direction is exactly the raw
gradient. Recent results say the choice of inner-loop optimizer matters to final performance, and that
nags at me, because I am leaving the inner optimizer at its crudest possible setting while pouring all
the meta-learning into the initialization. So the question is sharp: can I make the inner step *better
conditioned* — a smarter direction than the raw gradient — and fold that into the same meta-learning
frame?

What does "better conditioned" mean concretely? Plain gradient descent is bad exactly when the loss
surface is ill-conditioned: a long narrow valley, where the gradient points mostly across the valley
and barely along it, so SGD zig-zags. The classical cures precondition the gradient. Newton's method
steps `θ − α H^{-1} ∇L`, using the loss Hessian to rescale the step into the local quadratic geometry;
natural gradient steps `θ − α F^{-1} ∇L`, using the Fisher information matrix to take a steepest-descent
direction in distribution space. Both turn the zig-zag into a straight shot down the valley. So the
shape of the answer is clear: replace `∇L` in the inner step with `M ∇L` for some matrix `M` that
encodes curvature. The two questions are *where M comes from* and *how I afford it*.

Take "where M comes from" first, because here the few-shot setting forces a decision the classical
methods get wrong. Newton and natural gradient *compute* `H` or `F` from the current training loss and
data. That is fine with abundant data, but in few-shot I have `K` examples — one or five per class. A
curvature estimated from five points is curvature of *the support set*, and moving "as far as possible"
along it is moving as far as possible to fit five points, which is precisely how I overfit. The whole
pathology of few-shot learning is that fitting the support set is the wrong objective; a curvature
computed *to fit the support set faster* makes that worse, not better. So I must not compute `M` from
the task's training loss at all. But I have something the classical methods do not: a whole distribution
of tasks. My hypothesis — the same one that justifies a shared initialization — is that there are
*curvatures broadly applicable across tasks*, determined by the architecture, the loss, and the task
family, not by any one task's five points. If a shared *initialization* transfers across tasks, a
shared *preconditioner* should too. So I will *learn* `M` across the task distribution, jointly with
`θ`, by the same meta-objective: not "fit the support faster" but "transform the inner gradient so the
*post-adaptation query* loss generalizes." `M` becomes a meta-parameter, trained for generalization,
not estimated from data. That is the conceptual move, and it is exactly why this should beat dropping
K-FAC or a Hessian into the inner loop: those compute curvature from the support statistics and overfit;
mine is meta-learned for held-out performance.

Now "how I afford it," which is where the whole engineering problem lives. A full `M` is `dim(θ) ×
dim(θ)`. For a conv net that is the square of a hundred-thousand-dimensional vector — impossible to
store, let alone meta-learn, and the classical approximations (online Hessian updates, diagonal
approximations) exist precisely because the full matrix is hopeless. The cheapest learned
preconditioner already in the air is the *diagonal*: a per-parameter learning-rate vector, the inner
step `θ' = θ − α ∘ ∇L` with `α` learned. That captures per-coordinate rescaling and even tilts the step
off the raw gradient when the entries differ. But a diagonal `M` says every gradient coordinate is
rescaled *independently* — it cannot express that the gradient on one input channel should be combined
with the gradient on another, or that filter positions within a kernel covary. Real loss curvature is
not diagonal; the off-diagonal blocks are exactly the coordinate dependencies a diagonal throws away.
So I want something strictly richer than diagonal but vastly cheaper than the full matrix.

Here is the structural observation that breaks it open. A layer's parameters are not a flat vector —
they are a *tensor*. A conv layer's weight is `W ∈ R^{C_out × C_in × d}` (output channels, input
channels, filter size `d = h·w`), and its gradient `G` has the same shape; a linear layer is the `d =
1` case, `R^{C_out × C_in}`. The full `M` flattens this tensor and mixes all `C_out · C_in · d`
coordinates with all others. But the *natural* dependencies are along the three modes separately:
dependencies among the `d` elements *within* a filter (spatial structure of a kernel), dependencies
among the `C_in` input channels, and dependencies among the `C_out` output channels (which filters
relate to which). So instead of one giant matrix mixing everything, factor the transform into *three
small matrices*, one per mode, and apply each along its own axis of the gradient tensor. Define three
meta-curvature matrices per layer: `M_f ∈ R^{d × d}` (filter mode), `M_i ∈ R^{C_in × C_in}` (input
mode), `M_o ∈ R^{C_out × C_out}` (output mode). The tool for "multiply a tensor along one mode by a
matrix" is the `n`-mode product `X ×_n M`, which is just: unfold the tensor along mode `n` into a
matrix, multiply by `M`, fold back. So the transformed gradient is

  MC(G) = G ×_3 M_f ×_2 M_i ×_1 M_o,

three `n`-mode products in sequence. Read what each does. The 3-mode product `G ×_3 M_f` linearly
recombines the `d` elements within each filter — it captures the parameter dependencies *inside* a
convolutional kernel. The 2-mode product `×_2 M_i` recombines the gradient across input channels — each
input channel's contribution becomes a linear combination of the others'. The 1-mode product `×_1 M_o`
recombines across output channels — each filter's gradient becomes a linear combination of the gradients
of all filters in the layer. Composed, the three together examine the dependencies of *all* the layer's
gradient coordinates — but parameterized by `d² + C_in² + C_out²` numbers instead of `(C_out C_in d)²`.
That is the affordability win: structured curvature at the cost of three small matrices per layer.

I should check this is genuinely a transform over all coordinates and not three independent diagonal-ish
rescalings dressed up. A useful property of `n`-mode products is that the order is irrelevant for
distinct modes: `G ×_3 M_f ×_2 M_i ×_1 M_o = G ×_1 M_o ×_2 M_i ×_3 M_f`, so the three are applied "all
together," not as a privileged sequence. And if I wanted to see it as a single matrix-vector product (I
cannot afford to, but to understand it), `vec(MC(G)) = M_mc vec(G)` with `M_mc = (M_o ⊗ I_{C_in} ⊗ I_d)
(I_{C_out} ⊗ M_i ⊗ I_d)(I_{C_out} ⊗ I_{C_in} ⊗ M_f)`, i.e. a product of three Kronecker-structured
matrices, equivalently `M_mc = M_o ⊗ M_i ⊗ M_f` for this vectorization. So `M_mc` is a Kronecker
product of the three small matrices — with order fixed by the vectorization convention — and a
structured full matrix over the whole layer is never materialized. This is the same
*structure* K-FAC uses to approximate the Fisher (`A ⊗ G`), which reassures me the inductive bias is
sound; the difference is decisive — K-FAC computes its factors from training-loss statistics and inverts
them, whereas I *learn* my factors for generalization and never invert anything. And K-FAC's `A` block
is `R^{C_in d × C_in d}`, which is the expensive part; my factorization splits the input mode and the
filter mode into two separate small matrices `M_i ∈ R^{C_in × C_in}` and `M_f ∈ R^{d × d}`, avoiding
that large block entirely — cheaper, and (because the smaller factors have less capacity to memorize)
less prone to overfitting the meta-training set.

Now the inner update. Replace the raw gradient with its meta-curvature transform:

  θ' = θ − α · MC(∇L_train(θ)) = θ − α · (∇L ×_3 M_f ×_2 M_i ×_1 M_o),

per layer, with `α` the fixed inner learning rate. If I take several inner steps, the simplest version
reuses the same transform each time; if I need more capacity in a small head, I can allocate
step-specific transforms. The meta-parameters are the model initialization `θ` *and* all the `{M_o,
M_i, M_f}` across layers, all meta-trained together.

Two decisions the construction forces. First, initialization of the `M` matrices. At the start of
meta-training I have no reason to prefer any curvature, and I want to begin from behavior I trust —
which is MAML, plain gradient descent. If every `M` is the *identity*, then `MC(G) = G` exactly, the
inner step is `θ − α ∇L`, and I am running MAML. So initialize all three matrices per layer to identity;
meta-training deforms them away from identity as it discovers which coordinate recombinations help the
post-adaptation query loss generalize. This makes the method a strict generalization of MAML at
identity. I have to be more careful with Meta-SGD: Meta-SGD is an arbitrary same-size diagonal
preconditioner, while diagonal mode factors in this Kronecker form give a separable diagonal across
modes. So the right statement is that Meta-SGD is the diagonal-preconditioner baseline and MC is the
structured non-diagonal version, not that every Meta-SGD vector is contained by diagonal MC factors.
Second, the outer optimization. The whole point of folding `M` into the
existing frame is that the meta-gradient of the *transformed* inner step is computed by the same autodiff
machinery MAML already uses — I build `MC(∇L)` as a node in the graph (the `n`-mode products are
differentiable), subtract `α · MC(∇L)` keeping the subtraction in the graph, forward the query set, and
backprop the query loss into both `θ` and the `M` matrices. No new higher-order machinery beyond the
Hessian-vector product MAML already pays; no matrix inversion ever. Use Adam for the outer loop, as in
MAML.

Let me reason about why I should expect this to *generalize* better and not just fit faster, because
that is the claim that distinguishes it from naive second-order. Take the gradient of the meta-objective
with respect to a single `M` (call the flattened transform matrix `M`, the per-task inner update `θ^{τ}
= θ − α M ∇L_train(θ)`). The chain rule gives `∇_M L_val^τ(θ^τ) = −α ∇_θ L_val^τ(θ^τ) ∇_θ L_train^τ(θ)^⊤`
— an outer product between the gradient of the *validation* (query) loss at the adapted point and the
gradient of the *training* (support) loss at the initialization. Compare this to the empirical Fisher,
`F = E[∇_θ log p(y|x) ∇_θ log p(y|x)^⊤]`, an outer product of training-loss gradients with themselves.
Three differences, and each is a feature. The meta-gradient uses *separate* training and validation
sets — the validation set is never available to a within-task Fisher estimate, so `M` is shaped to help
*held-out* points, not the support it adapted on. The validation gradient is evaluated at the
*post-adaptation* parameters `θ^τ`, not at `θ` — it knows where adaptation lands. And the Fisher is
positive semidefinite by construction (always a descent direction), whereas `M` is not forced to be — I
*let* it be indefinite, trading the descent guarantee for the freedom to transform the gradient in
whatever way most helps generalization. So meta-curvature is "a Fisher-shaped object, but trained on
held-out loss after adaptation instead of estimated from training loss" — which is exactly the
generalization-vs-fitting distinction the few-shot setting demands.

So the picture is settled and it came straight out of the tension. The inner step in the meta-learning
frame is plain SGD, and the choice of inner optimizer matters; a better-conditioned step means
preconditioning the gradient with curvature, but classical curvature is computed from the support set
(overfits) and full matrices do not scale. The resolution is to *learn* the preconditioner across the
task distribution for generalization, and to make it affordable by factoring it over the natural tensor
modes of each layer — three small matrices `M_f`, `M_i`, `M_o` applied as `n`-mode products,
`MC(G) = G ×_3 M_f ×_2 M_i ×_1 M_o`, equivalently a Kronecker-factored `M_o ⊗ M_i ⊗ M_f` never
materialized under the chosen vectorization. The inner update becomes `θ' = θ − α MC(∇L)`; the `M`
matrices initialize to identity (so it starts as MAML) and are meta-trained jointly with `θ` by the same
autodiff outer loop (no inversion, no new higher-order cost). The result is a structured, learned,
generalization-oriented preconditioner of the inner-loop gradient that captures coordinate dependencies a
coordinate-wise diagonal cannot, at the cost of a few small matrices per layer.

Let me write it as the inner-loop optimizer it actually is, without adding decorative parameters. For a
convolutional weight in PyTorch layout `[C_out, C_in, kh, kw]`, I fold the spatial dimensions into
`d = kh·kw`, create identity-initialized `M_o`, `M_i`, and `M_f`, reshape the gradient to `[C_out,
C_in, d]`, apply the three mode products, and reshape back. For a 2D linear weight, `d = 1` makes the
filter-mode factor a redundant scalar, so I use only the input and output matrices. For 1D bias and
normalization parameters, an elementwise factor initialized to ones is enough; a dense bias matrix would
invent cross-coordinate coupling I did not justify. If I translate between layouts, `[kh, kw, C_in,
C_out]` naturally right-multiplies the output-mode factor, while `[C_out, C_in, ...]` puts the same
output-mode multiplication on the left. The differentiable update is still `p ← p − α MC(g)`, and the
inner optimizer returns only the curvature factors as optimizer-state meta-parameters while the outer
loop separately optimizes the shared initialization `θ`.

The causal chain, start to finish: in the meta-learning frame the inner step is plain SGD and the
inner optimizer is known to matter, so I want a better-conditioned step — precondition the gradient with
curvature, `θ − α M ∇L` — but classical curvature (Hessian, Fisher, K-FAC) is computed from the few
support points and so overfits, and a full `M` does not scale; the fix is to *learn* `M` across the task
distribution for generalization (its meta-gradient is a Fisher-shaped outer product but of the *query*
gradient after adaptation against the support gradient, with separate train/val sets and no forced
positive-definiteness), and to *afford* it by factoring `M` over each layer's natural tensor modes into
three small matrices `M_f, M_i, M_o` applied as `n`-mode products, equivalently a Kronecker product
never materialized; initialized to identity it starts as MAML, diagonal factors give a separable
coordinate-wise approximation rather than arbitrary Meta-SGD, and it folds into the existing autodiff
outer loop with no matrix inversion and no new higher-order cost beyond MAML's.
