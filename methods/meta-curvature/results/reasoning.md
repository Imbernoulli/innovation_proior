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
shape of the move is to replace `∇L` in the inner step with `M ∇L` for some matrix `M` that
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
shared *preconditioner* plausibly should too. So I will *learn* `M` across the task distribution, jointly
with `θ`, by the same meta-objective: not "fit the support faster" but "transform the inner gradient so
the *post-adaptation query* loss generalizes." `M` becomes a meta-parameter, trained for generalization,
not estimated from data. I will need to come back and check that this learning signal really is
generalization-shaped and not secretly just refitting — I will hold that thought and verify it later,
once I have the update written down.

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

Here is a structural observation that might break it open. A layer's parameters are not a flat vector —
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
matrix, multiply by `M`, fold back. So the candidate transformed gradient is

  MC(G) = G ×_3 M_f ×_2 M_i ×_1 M_o,

three `n`-mode products in sequence. Read what each does. The 3-mode product `G ×_3 M_f` linearly
recombines the `d` elements within each filter — it captures the parameter dependencies *inside* a
convolutional kernel. The 2-mode product `×_2 M_i` recombines the gradient across input channels — each
input channel's contribution becomes a linear combination of the others'. The 1-mode product `×_1 M_o`
recombines across output channels — each filter's gradient becomes a linear combination of the gradients
of all filters in the layer. Parameterized by `d² + C_in² + C_out²` numbers instead of `(C_out C_in d)²`.

Before I lean on this I want to know whether the composition is doing what I just said, because the
story only holds if (a) the three products genuinely act "all together" rather than as one privileged
ordering, and (b) the whole thing is a single linear transform of all the layer's coordinates, not three
diagonal-ish rescalings dressed up. Let me take a small concrete tensor and check by direct computation
rather than trust the picture. Take `C_out = 2, C_in = 3, d = 2` and random `M_o, M_i, M_f` of the
matching sizes. Implement `×_3 M_f` as `(o,i,d),(f,d) → (o,i,f)`, `×_2 M_i` as `(o,i,d),(j,i) →
(o,j,d)`, `×_1 M_o` as `(o,i,d),(j,o) → (j,i,d)`. Computing `G ×_3 M_f ×_2 M_i ×_1 M_o` and then the
reversed order `G ×_1 M_o ×_2 M_i ×_3 M_f` and a third arbitrary permutation, all three numerically
agree to machine precision (max abs difference `~4e-16`). So distinct-mode products do commute, and the
ordering in the formula is cosmetic — the three are applied together.

Now (b), the "is it really a full transform" question. If I wanted to see `MC(G)` as a single
matrix-vector product `vec(MC(G)) = M_mc vec(G)` (I cannot afford to materialize `M_mc`, but I want to
know what it *is*), the natural guess given the Kronecker structure of mode products is `M_mc = M_o ⊗
M_i ⊗ M_f`. The order in that Kronecker product is not something I can read off by inspection — it
depends on the vectorization convention — so I tested it. Flattening `G` in C-order with index `(o, i,
d)` and `d` running fastest, the prediction `(M_o ⊗ M_i ⊗ M_f) · vec(G)` reproduces `vec(MC(G))` exactly
(max abs difference `~1.8e-15`), while the swapped orders `M_f ⊗ M_i ⊗ M_o` and `M_o ⊗ M_f ⊗ M_i` do
*not* match. (For the column-major convention the matching order flips to `M_f ⊗ M_i ⊗ M_o`, which is
why I have to state the convention rather than just assert "Kronecker product".) So `M_mc` is a genuine
Kronecker product of the three small matrices over the whole layer — a dense linear map mixing all
`C_out · C_in · d` coordinates, with order pinned by the `(o,i,d)` C-order vectorization — never
materialized. Two things follow. First, this is the same *structure* K-FAC uses to approximate the
Fisher (`A ⊗ G`), which reassures me the inductive bias is at least the one curvature methods already
trust; the difference is that K-FAC computes its factors from training-loss statistics and inverts them,
whereas I *learn* my factors for generalization and never invert anything. Second, K-FAC's `A` block is
`R^{C_in d × C_in d}`, the expensive part; my factorization splits the input mode and the filter mode
into two separate small matrices `M_i ∈ R^{C_in × C_in}` and `M_f ∈ R^{d × d}`, avoiding that large
block entirely — cheaper, and (because the smaller factors have less capacity to memorize) plausibly less
prone to overfitting the meta-training set.

Is the affordability win actually large, or did I just talk myself into it? The factored cost is `C_out²
+ C_in² + d²` versus the full `(C_out C_in d)²`. For my toy `2,3,2` that is `4 + 9 + 4 = 17` versus
`144` — already an order of magnitude. For one realistic conv layer with `C_out = C_in = 64, d = 9` it
is `64² + 64² + 9² = 8273` parameters versus `(64·64·9)² ≈ 1.36 × 10⁹`. So the factorization is the
difference between roughly eight thousand learnable numbers per layer and over a billion: the full matrix
is not merely expensive, it is five orders of magnitude out of reach, and the factored form is the thing
that makes a *learned* curvature even storable. Good — the structure is what buys feasibility, not a nice-to-have.

Now the inner update. Replace the raw gradient with its meta-curvature transform:

  θ' = θ − α · MC(∇L_train(θ)) = θ − α · (∇L ×_3 M_f ×_2 M_i ×_1 M_o),

per layer, with `α` the fixed inner learning rate. If I take several inner steps, the simplest version
reuses the same transform each time; if I need more capacity in a small head, I can allocate
step-specific transforms. The meta-parameters are the model initialization `θ` *and* all the `{M_o,
M_i, M_f}` across layers, all meta-trained together.

Two decisions the construction forces. First, initialization of the `M` matrices. At the start of
meta-training I have no reason to prefer any curvature, and I want to begin from behavior I trust —
plain gradient descent, i.e. MAML. The natural choice is the identity: `G ×_n I = G` along each mode, so
`MC(G)` should collapse to `G` and the inner step to `θ − α ∇L`. I do not want to merely assert this, so
I ran the same toy tensor through the three mode products with `M_o, M_i, M_f` all set to identity, and
`MC(G)` came back equal to `G` to machine precision. So identity init reproduces MAML exactly, and this
method is a strict generalization of MAML that meta-training then deforms away from identity as it
discovers which coordinate recombinations help the post-adaptation query loss generalize. I have to be
more careful with Meta-SGD: Meta-SGD is an arbitrary same-size diagonal preconditioner, while diagonal
mode factors in this Kronecker form give a *separable* diagonal across modes (the layer-wise diagonal is
the Kronecker product of the three mode diagonals, not an arbitrary vector). So the honest statement is
that Meta-SGD is the diagonal-preconditioner baseline and MC is the structured non-diagonal version, not
that every Meta-SGD vector is contained by diagonal MC factors.

Second, the outer optimization. The point of folding `M` into the existing frame is that the
meta-gradient of the *transformed* inner step is computed by the same autodiff machinery MAML already
uses — I build `MC(∇L)` as a node in the graph (the `n`-mode products are differentiable), subtract `α ·
MC(∇L)` keeping the subtraction in the graph, forward the query set, and backprop the query loss into
both `θ` and the `M` matrices. No new higher-order machinery beyond the Hessian-vector product MAML
already pays; no matrix inversion ever. Use Adam for the outer loop, as in MAML.

Now I owe myself the verification I deferred: does the meta-learning signal on `M` actually shape it for
*generalization* rather than for refitting the support — and is it really "Fisher-shaped" the way I keep
claiming? Let me derive the meta-gradient of `M` and check the derivation numerically before I trust the
story built on it. Write the flattened transform as `M`, the per-task inner update `θ^τ = θ − α M
∇L_train(θ)`, and the meta-loss as the query loss `L_val^τ(θ^τ)`. By the chain rule, since `θ^τ` depends
on `M` only through the `−α M ∇L_train(θ)` term and `∇L_train(θ)` is evaluated at the *initialization*
(it does not depend on `M`),

  ∇_M L_val^τ(θ^τ) = −α ∇_θ L_val^τ(θ^τ) · ∇_θ L_train^τ(θ)^⊤,

an outer product between the gradient of the *validation* (query) loss at the adapted point and the
gradient of the *training* (support) loss at the initialization. That is a clean claim and exactly the
kind of thing I get wrong by a sign or a transpose, so I checked it by finite differences: pick `n = 4`,
random quadratic train/val losses, `α = 0.3`, a random `M`, form `θ^τ = θ − α M ∇L_train(θ)`, and
compare the analytic outer product against a central-difference gradient over every entry of `M`. They
agree to `~1e-8`, which is finite-difference-limited, so the identity holds.

With the identity confirmed, the comparison to curvature is sharp. The empirical Fisher is `F = E[∇_θ
log p(y|x) ∇_θ log p(y|x)^⊤]`, an outer product of training-loss gradients *with themselves*. The
meta-gradient is also an outer product of two gradients — but the two gradients are different objects,
and each difference is a feature. The meta-gradient uses *separate* training and validation sets: the
validation gradient is the left factor, and the validation set never enters a within-task Fisher
estimate, so `M` is being shaped to help *held-out* points rather than the support it adapted on. The
validation gradient is evaluated at the *post-adaptation* parameters `θ^τ`, not at `θ` — it knows where
adaptation lands, which a static curvature does not. And the Fisher is positive semidefinite by
construction (always a descent direction), whereas this outer product is not forced to be — I *let* `M`
be indefinite, trading the descent guarantee for the freedom to transform the gradient in whatever way
most helps generalization. So `M` is a Fisher-shaped object, but driven by held-out loss after
adaptation instead of estimated from training loss — which is exactly the generalization-versus-fitting
distinction the few-shot setting demanded, and it answers the worry I parked earlier: the learning signal
is structurally a held-out signal, not a disguised refit.

So the picture closes, and it closed by computation at each load-bearing step rather than by assertion.
The inner step in the meta-learning frame is plain SGD, and the choice of inner optimizer matters; a
better-conditioned step means preconditioning the gradient with curvature, but classical curvature is
computed from the support set (overfits) and full matrices do not scale. The resolution is to *learn* the
preconditioner across the task distribution for generalization (the meta-gradient is the held-out
outer product I just verified, not a within-task Fisher), and to make it affordable by factoring it over
the natural tensor modes of each layer — three small matrices `M_f`, `M_i`, `M_o` applied as `n`-mode
products, `MC(G) = G ×_3 M_f ×_2 M_i ×_1 M_o`, equivalently a Kronecker-factored `M_o ⊗ M_i ⊗ M_f`
(C-order `(o,i,d)` vectorization) never materialized. The inner update becomes `θ' = θ − α MC(∇L)`; the
`M` matrices initialize to identity (so it starts as MAML, which I checked numerically) and are
meta-trained jointly with `θ` by the same autodiff outer loop (no inversion, no new higher-order cost).
The result is a structured, learned, generalization-oriented preconditioner of the inner-loop gradient
that captures coordinate dependencies a coordinate-wise diagonal cannot, at the cost of a few small
matrices per layer.

Let me write it as the inner-loop optimizer it actually is, without adding decorative parameters. For a
convolutional weight in PyTorch layout `[C_out, C_in, kh, kw]`, I fold the spatial dimensions into
`d = kh·kw`, create identity-initialized `M_o`, `M_i`, and `M_f`, reshape the gradient to `[C_out,
C_in, d]`, apply the three mode products, and reshape back. For a 2D linear weight, `d = 1` makes the
filter-mode factor a redundant scalar, so I use only the input and output matrices. For 1D bias and
normalization parameters, an elementwise factor initialized to ones is enough; a dense bias matrix would
invent cross-coordinate coupling I did not justify. Before trusting the einsum that implements this, I
ran the conv branch on a random `[2,3,2,2]` weight against the verified Kronecker form `(M_o ⊗ M_i ⊗
M_f) vec(G)` and the two matched, and the identity-factor case returned the gradient unchanged — so the
code computes the transform I derived, not a transposed or mis-ordered cousin of it. If I translate
between layouts, `[kh, kw, C_in, C_out]` naturally right-multiplies the output-mode factor, while
`[C_out, C_in, ...]` puts the same output-mode multiplication on the left. The differentiable update is
still `p ← p − α MC(g)`, and the inner optimizer returns only the curvature factors as optimizer-state
meta-parameters while the outer loop separately optimizes the shared initialization `θ`.

The causal chain, start to finish: in the meta-learning frame the inner step is plain SGD and the
inner optimizer is known to matter, so I want a better-conditioned step — precondition the gradient with
curvature, `θ − α M ∇L` — but classical curvature (Hessian, Fisher, K-FAC) is computed from the few
support points and so overfits, and a full `M` does not scale (over a billion numbers for one conv layer
versus eight thousand factored); the fix is to *learn* `M` across the task distribution for
generalization (its meta-gradient is the verified outer product of the *query* gradient after adaptation
against the support gradient, with separate train/val sets and no forced positive-definiteness), and to
*afford* it by factoring `M` over each layer's natural tensor modes into three small matrices `M_f, M_i,
M_o` applied as commuting `n`-mode products, equivalently a Kronecker product never materialized;
initialized to identity it reproduces MAML exactly, diagonal factors give a separable coordinate-wise
approximation rather than arbitrary Meta-SGD, and it folds into the existing autodiff outer loop with no
matrix inversion and no new higher-order cost beyond MAML's.
</content>
</invoke>
