ANIL's numbers close one axis and, by closing it, point straight at the one I have not touched. The bet
paid off exactly as I framed it. At 5-shot ANIL recovered everything Meta-SGD lost and then some:
miniImageNet 5-shot back to 0.6366 (Meta-SGD had sagged to 0.6237) and CIFAR-FS 5-shot up to 0.7138 —
past MAML's 0.7067 and the best 5-shot CIFAR number on the ladder — with the seed spread that Meta-SGD
had widened (std 0.0113 on CIFAR) tightening back to 0.0035. So freezing the body did reverse the
meta-overfitting, confirming feature reuse was the whole story at 5-shot. And at 1-shot ANIL did not
just hold Meta-SGD's gain, it improved on it: 0.4815 against 0.4760, the tightest 1-shot spread yet
(std 0.0029, seeds {0.4814, 0.4852, 0.4780}). So ANIL is the strongest rung on every benchmark. That is
the good news and also the problem: I have now optimized two of the three axes of the inner loop. MAML
learned *where to start*. Meta-SGD learned a per-coordinate *rate*. ANIL settled *which parameters*
adapt — the head, not the body. What I have *not* learned, on any rung, is the *direction*: every method
so far steps along the raw gradient (Meta-SGD only rescales it coordinate-by-coordinate, which is a
diagonal tilt at best). The inner step is still, geometrically, "follow the gradient." That is the axis
left on the table, and it is exactly where ill-conditioning lives.

Let me look hard at what the surviving inner loop actually does, because ANIL sharpened the question
rather than answering it. ANIL freezes the body and adapts only the head — a single linear classifier
mapping the 1600-dim frozen feature to five logits — by plain SGD, ten steps at evaluation. So the entire
adaptation, at test time, is *gradient descent on a five-way logistic regression over a fixed
1600-dimensional feature*. And that is precisely the setting where the *direction* of the step matters
most, because the conditioning of that little problem is set by the feature covariance, which I do not
control and which is generally far from isotropic: some feature directions are high-variance and
correlated across the 1600 inputs, others are nearly dead. Plain SGD on an ill-conditioned logistic
regression zig-zags — it moves fast along the cheap directions and crawls along the informative ones —
and ten steps from a single example per class is not many steps to waste on a zig-zag. ANIL's own
gain came from *removing* the body's pointless inner loop; what remains is a head whose inner loop is
still geometrically naive. So the next move is not another axis of *what* to adapt — it is to fix *how*
the surviving inner step moves: precondition the gradient so the step points down the valley instead of
across it.

What does "precondition" mean concretely, and where does the preconditioner come from? The classical
answer is second-order: Newton steps `θ − α H^{-1} ∇L`, rescaling the gradient by the inverse loss
Hessian; natural gradient steps `θ − α F^{-1} ∇L`, rescaling by the inverse Fisher. Both turn the
zig-zag into a straight shot. But I cannot lift them in unchanged, and the reason is the same few-shot
pathology that has shaped every rung: Newton and natural gradient *compute* `H` or `F` from the current
task's data — and the current task is one or five examples. A curvature estimated from five points is the
curvature of *the support set*, and stepping "as far as possible" along it is stepping as far as possible
to fit five points, which is exactly how I overfit. I saw this already in another guise: Meta-SGD's extra
capacity, tuned to the support, meta-overfit at 5-shot. So I must *not* compute the preconditioner from
the task's loss. But the meta-learning frame gives me the escape it always has — the task *distribution*.
If a shared *initialization* transfers across tasks (it does — that is MAML), and a shared per-coordinate
*rate* transfers (it does — that is Meta-SGD), then a shared *preconditioner matrix* should transfer too.
So I learn the preconditioner `M` across tasks, jointly with θ, by the same meta-objective: not "fit the
support faster" but "transform the inner gradient so the *post-adaptation query* loss generalizes." `M`
becomes a meta-parameter, trained for held-out performance, not estimated from data. That is the move
that distinguishes this from naively dropping a Hessian into the head's inner loop — which would estimate
curvature from five points and overfit the same way Meta-SGD did.

Now the affordability problem, which is where this has to be made concrete against the harness. A full
preconditioner `M` is `dim(θ) × dim(θ)`. Even restricted to the head, that is `(5·1600)² ≈ 6.4×10^7`
numbers — and the harness budget caps the optimizer's learnable state at roughly the model size (it is
sized for Meta-SGD's one-scalar-per-parameter footprint, times 1.2). A full matrix is out, for the same
reason it was out everywhere. The structural observation that breaks it open is that a layer's parameters
are not a flat vector — they are a *tensor*. A conv weight is `R^{C_out × C_in × d}` (output channels,
input channels, filter size `d = h·w`), the head is the `d = 1` case `R^{C_out × C_in}`, and the gradient
has the same shape. The natural dependencies run along the three modes *separately*: among the `d`
elements within a filter, among the `C_in` input channels, among the `C_out` output channels. So instead
of one giant matrix mixing all `C_out·C_in·d` coordinates, factor the transform into three small matrices
per layer — `M_f ∈ R^{d×d}`, `M_i ∈ R^{C_in×C_in}`, `M_o ∈ R^{C_out×C_out}` — and apply each along its
own axis with an `n`-mode product. The transformed gradient is `MC(G) = G ×_3 M_f ×_2 M_i ×_1 M_o`. Read
each factor: `×_3 M_f` linearly recombines the elements within each filter (spatial structure of a
kernel); `×_2 M_i` recombines across input channels; `×_1 M_o` recombines across output channels (each
filter's gradient becomes a combination of all filters'). Composed, the three examine the dependencies of
*all* the layer's gradient coordinates — but parameterized by `d² + C_in² + C_out²` numbers instead of
`(C_out C_in d)²`. As a single never-materialized matrix this is the Kronecker product `M_o ⊗ M_i ⊗ M_f`,
the same structure K-FAC uses for the Fisher (`A ⊗ G`); the decisive difference is that K-FAC computes
its factors from training-loss statistics and inverts them, whereas I *learn* mine for generalization and
never invert anything.

This is strictly richer than every rung so far, and the special cases make that precise. If every `M` is
the *identity*, then `MC(G) = G` and the inner step is `θ − α ∇L` — exactly MAML. If every `M` is
*diagonal*, the transform is a per-coordinate rescaling — exactly Meta-SGD. The full Kronecker-factored
`M` keeps the *off-diagonal* coordinate dependencies a diagonal throws away: it can combine the gradient
on one input channel with another's, which is precisely the off-valley correction plain SGD (and even
Meta-SGD's diagonal) cannot make. So initializing all factors to identity starts the method as MAML and
lets meta-training deform them away as it discovers which coordinate recombinations help the
post-adaptation query loss — the same "start from behavior I trust" principle that made me initialize
Meta-SGD's rates uniformly. And I should expect this to *generalize* rather than just fit faster, for a
reason I can read off the meta-gradient: the gradient of the meta-loss with respect to a flattened `M`
(inner step `θ^τ = θ − α M ∇L_train`) is `−α ∇_θ L_val(θ^τ) · ∇_θ L_train(θ)^⊤`, an outer product that
*resembles* the empirical Fisher but differs in three features that all favor generalization — it uses
*separate* train and validation sets (the validation set never enters a within-task Fisher), it evaluates
the validation gradient at the *post-adaptation* parameters, and it does not force `M` positive-definite
(I let it be indefinite, trading the descent guarantee for the freedom to transform the gradient however
most helps held-out points).

Now ground it in *this task's* edit surface and budget, because the harness forces two concrete
decisions and the budget forces a third. The contract is `__init__` (create learnable factors), `adapt`
(differentiable inner step), `meta_parameters` (hand factors to the outer Adam). In `__init__` I attach,
per model parameter, the three factor matrices sized to its modes, all identity-initialized: a 4D conv
weight `[C_out, C_in, kh, kw]` folds its spatial dims into `d = kh·kw`; the 2D head is the `d = 1` case;
1D bias and BatchNorm parameters carry only the output-mode `M_o`. The inner step computes the support
cross-entropy, takes `torch.autograd.grad` over the parameters with the graph kept, transforms each
gradient by its factors via the three `n`-mode products (plain `einsum` matrix multiplies along each
axis), and applies the differentiable update `p ← p − α MC(g)` by re-routing parameter tensors so the
outer loop can backprop into both θ and the factors; `meta_parameters()` returns all the factors. This is
the literal generalization of the earlier rungs' edits: where MAML used a scalar in `maml_update` and
Meta-SGD used a per-parameter vector in `update_module`, I use a per-parameter *transform* and the same
differentiable re-route.

The budget forces the one deviation from the textbook transform, and it is the right one. The factors are
tiny everywhere the input dimension is small — every conv layer has `C_in ≤ 64`, so `M_i` is at most
`64×64`, and the full Kronecker curvature is essentially free there. The single oversized factor is the
head's input mode: `C_in = 1600` makes a dense `M_i` a `1600×1600 = 2.56M`-parameter matrix, twenty
times the whole model, which blows the budget by itself. This is exactly the cost my factorization was
meant to *avoid* in the first place — splitting the input and filter modes into separate small factors
precisely so as not to carry a single large `C_in·d` block. So for any mode larger than a threshold I degrade its
factor to a *diagonal* — a per-element rate over that mode, Meta-SGD's footprint — while keeping the dense
factors on every affordable mode. Concretely the head keeps a dense `5×5` output-mode `M_o` (it can
freely recombine the five class-logit gradients, which is the cheap and meaningful part for a five-way
head) and a diagonal input-mode rate over the 1600 features. With that, the total optimizer state lands at
~80K against a model of ~121K, comfortably inside the ~291K budget — full structured curvature where it
is affordable, a diagonal fallback only on the one mode that cannot be afforded. The factors still
initialize to identity (diagonal factors to ones), so the method still begins as MAML. The full scaffold
module is in the answer.

So the delta from ANIL is the axis ANIL left untouched: ANIL fixed *which* parameters adapt and left the
surviving head's step as plain gradient descent; meta-curvature fixes *how* that step moves by
preconditioning the gradient with a learned, Kronecker-factored, generalization-trained curvature —
applied across all layers, not only the head, so the body's outer-loop-learned features are themselves
shaped by a better-conditioned meta-objective. The bar this must clear is ANIL's real numbers, and the
falsifiable claims are pointed at the regime where the *direction* of the step should matter most. At
**1-shot** — the hardest, most ill-conditioned setting, where ten steps from a single example per class
are most wasted on a zig-zag — meta-curvature should clear ANIL's 0.4815: a preconditioned step that
points down the valley should extract more from those ten steps than plain SGD, so I expect 1-shot to be
the largest gain, and I would treat *failure* to beat 0.4815 at 1-shot as the method's central claim
falsified. At **5-shot** I expect to at least match ANIL's strong 0.6366 (miniImageNet) and 0.7138
(CIFAR-FS) and likely edge past, since the structured curvature is a strict superset of both the identity
(MAML) and diagonal (Meta-SGD) steps and is meta-trained for generalization rather than support-fitting —
though the risk I will watch is meta-overfitting from the larger factor count, the same failure that bit
Meta-SGD at 5-shot, which the identity initialization and the diagonal fallback on the oversized head mode
are meant to contain. This task's leaderboard has no meta-curvature row to confirm it; what the trajectory
ends on is the construction that, having exhausted *where to start*, *how far per coordinate*, and *which
parameters*, finally learns *which direction* — the structured, generalization-oriented preconditioner of
the inner gradient that captures the coordinate dependencies every earlier rung's diagonal-or-identity
step could not.
