ANIL's numbers close one axis and, by closing it, point straight at the one I have not touched. The bet
paid off exactly as I framed it. At 5-shot ANIL recovered everything Meta-SGD lost: miniImageNet 5-shot
back to 0.6366 (Meta-SGD had sagged to 0.6237, so $+0.0129$) and CIFAR-FS 5-shot up to 0.7138 (from
0.6885, $+0.0253$) — past MAML's 0.7067 and the best 5-shot CIFAR number yet — with the seed
spread that Meta-SGD had widened (std 0.0113 on CIFAR) tightening back to 0.0035. And at 1-shot ANIL did
not just hold Meta-SGD's gain, it improved on it: 0.4815 against 0.4760, the tightest 1-shot spread yet
(std 0.0029, seeds {0.4814, 0.4852, 0.4780}). The three-benchmark average makes it unambiguous —
$0.6106$ for ANIL against $0.5961$ Meta-SGD and $0.5937$ MAML — and, unlike Meta-SGD's wash, this is a
uniform lift rather than a redistribution: the strongest or tied-strongest method on all three. The one
soft spot: miniImageNet 5-shot at 0.6366 is a hair *under* MAML's 0.6379 (a $-0.0013$ I read as a tie
inside ANIL's own $0.0112$ spread there, seed 42 dipping to 0.6208), so that column is recovered but
not conquered. Still, ANIL is the strongest so far, and that is the good news and also the problem: I
have now optimized two of the three axes of the inner loop. MAML learned *where to start*. Meta-SGD
learned a per-coordinate *rate*. ANIL settled *which parameters* adapt — the head, not the body. What I
have *not* learned yet is the *direction*: every method so far steps
along the raw gradient (Meta-SGD only rescales it coordinate-by-coordinate, which is a diagonal tilt at
best, at most the thirty-nine-degree reorientation a diagonal can manage). The inner step is still,
geometrically, "follow the gradient." That is the axis left on the table, and it is exactly where
ill-conditioning lives.

Let me look hard at what the surviving inner loop actually does, because ANIL sharpened the question
rather than answering it. ANIL freezes the body and adapts only the head — a single linear classifier
mapping the 1600-dim frozen feature to five logits — by plain SGD, ten steps at evaluation. So the
entire adaptation, at test time, is *gradient descent on a five-way logistic regression over a fixed
1600-dimensional feature*. And that is precisely the setting where the *direction* of the step matters
most, because the conditioning of that little problem is set by the feature covariance, which I do not
control and which is generally far from isotropic: some feature directions are high-variance and
correlated across the 1600 inputs, others are nearly dead. I can put a number on how much this costs.
Gradient descent on a quadratic with condition number $\kappa$ contracts the error along the worst
direction by a factor $(\kappa-1)/(\kappa+1)$ per step. For a plausibly ill-conditioned head problem,
$\kappa=100$, that factor is $99/101 = 0.980$, and over the ten evaluation steps it compounds to
$0.980^{10} = 0.82$ — after all ten steps I have removed only about $18\%$ of the error along the
informative-but-stiff direction, because plain SGD spends those steps zig-zagging fast along the cheap
directions and crawling along the ones that carry the class signal. Ten steps from a single example per
class is not many steps to waste on a zig-zag. ANIL's own gain came from *removing* the body's
pointless inner loop; what remains is a head whose inner loop is still geometrically naive. So the next
move is not another axis of *what* to adapt — it is to fix *how* the surviving inner step moves:
precondition the gradient so the step points down the valley instead of across it.

What does "precondition" mean concretely, and where does the preconditioner come from? The classical
answer is second-order: Newton steps `θ − α H^{-1} ∇L`, rescaling the gradient by the inverse loss
Hessian; natural gradient steps `θ − α F^{-1} ∇L`, rescaling by the inverse Fisher. Both, on that
$\kappa=100$ problem, drive $\kappa$ toward $1$ and turn the ten-step crawl into essentially a single
straight shot. But I cannot lift them in unchanged, and the reason is the same few-shot pathology that
has shaped every step so far: Newton and natural gradient *compute* `H` or `F` from the current task's data —
and the current task is one or five examples. A curvature estimated from five points is the curvature of
*the support set*, and stepping "as far as possible" along it is stepping as far as possible to fit five
points, which is exactly how I overfit. I saw this already in another guise: Meta-SGD's extra capacity,
tuned to the support, meta-overfit at 5-shot. So I must *not* compute the preconditioner from the task's
loss. But the meta-learning frame gives me the escape it always has — the task *distribution*. If a
shared *initialization* transfers across tasks (it does — that is MAML), and a shared per-coordinate
*rate* transfers (it does — that is Meta-SGD), then a shared *preconditioner matrix* should transfer
too. So I learn the preconditioner `M` across tasks, jointly with θ, by the same meta-objective: not
"fit the support faster" but "transform the inner gradient so the *post-adaptation query* loss
generalizes." `M` becomes a meta-parameter, trained for held-out performance, not estimated from data.
That is the move that distinguishes this from naively dropping a Hessian into the head's inner loop —
which would estimate curvature from five points and overfit the same way Meta-SGD did.

Now the affordability problem, which is where this has to be made concrete against the harness. A full
preconditioner `M` is `dim(θ) × dim(θ)`. Even restricted to the head, that is `(5·1600)² ≈ 6.4×10^7`
numbers — and the harness budget caps the optimizer's learnable state at roughly the model size, about
$121\text{K}$, times $1.2$ against two copies, so $\sim291\text{K}$. A full matrix is out, for the same
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

Kronecker is not the only sub-quadratic structure I could have picked, so let me rule out the obvious
competitors by arithmetic before committing. A low-rank preconditioner `M ≈ UV^⊤` with `U, V` of shape
`dim(θ)×r` costs `2·dim(θ)·r`; on the head alone (`dim = 8000`) staying inside a sensible share of the
`291K` budget forces `r` down to single digits, and a rank-six transform simply cannot represent an
independent scaling of 1600 feature directions — the very thing the ill-conditioned head needs — so
low-rank is too poor on the dominant mode while spending its budget mixing coordinates across the tensor
structure at random. A per-output-channel block-diagonal `M` (a separate `C_in·d × C_in·d` block per
output channel) is the opposite failure: for the head that is `5·1600² = 12.8M` numbers, worse than the
full matrix restricted to what it captures. The Kronecker factorization sits between them precisely
because it *respects* the tensor's three modes instead of fighting them: it recombines within each mode
and takes their product, which is where the real dependencies live (output channel to output channel,
input channel to input channel) and nowhere they do not. That is why it is the structure I keep.

This is strictly richer than every method so far, and the special cases make that precise. If every `M`
is the *identity*, then each mode-product is a no-op, `MC(G) = G`, and the inner step is `θ − α ∇L` —
exactly MAML. If every `M` is *diagonal*, the transform is a per-coordinate rescaling — exactly
Meta-SGD. The full Kronecker-factored `M` keeps the *off-diagonal* coordinate dependencies a diagonal
throws away: it can combine the gradient on one input channel with another's, which is precisely the
off-valley correction plain SGD (and even Meta-SGD's diagonal) cannot make. Make it concrete on two
output channels: a dense `M_o = [[1, 0.5], [0.5, 1]]` sends the channel gradients `(g_1, g_2)` to
`(g_1 + 0.5 g_2,\; 0.5 g_1 + g_2)`, so a large gradient on channel 1 now nudges channel 2's update —
the cross-term `0.5 g_1` is exactly the entry no diagonal, however finely tuned, can produce, because a
diagonal's off-diagonal is zero by definition. If the two channels' gradients are correlated across the
task family (which whitening the feature covariance is all about), that off-diagonal is where the
conditioning fix lives. So initializing all factors to identity starts the method as MAML and lets
meta-training deform them away as it discovers which coordinate recombinations help the post-adaptation
query loss — the same "start from behavior I trust" principle that made me initialize Meta-SGD's rates
uniformly.

And I should expect this to *generalize* rather than just fit faster, for a reason I can read off the
meta-gradient. With inner step `θ^τ = θ − α M ∇L_train`, the gradient of the meta-loss with respect to a
flattened `M` is `−α ∇_θ L_val(θ^τ) · ∇_θ L_train(θ)^⊤`, an outer product. Check the shapes: `∇_θ L_val`
is a `dim(θ)`-vector, `∇_θ L_train` is a `dim(θ)`-vector, their outer product is `dim(θ)×dim(θ)`, exactly
the shape of the (flattened) `M` it updates — consistent, and rank-1 per task, so over the four-task
meta-batch and 60,000 iterations `M` accumulates a sum of many such rank-1 outer products, a Gram-like
object. It *resembles* the empirical Fisher but differs in three features that all favor generalization:
it uses *separate* train and validation sets (the validation set never enters a within-task Fisher), it
evaluates the validation gradient at the *post-adaptation* parameters `θ^τ`, and it does not force `M`
positive-definite. That last choice is deliberate and worth dwelling on. A positive-definite `M` (say
parameterized as `LL^⊤`) would guarantee the preconditioned step is a descent direction on the support
loss — but descending the support loss as efficiently as possible is *exactly the overfitting* I have
been fighting since MAML. So I want to keep the freedom to *ascend* the support loss on the coordinates
where doing so generalizes, which means letting `M` be indefinite; the identity-initialized factors carry
no sign constraint, so nothing stops an entry from going negative, and that is the same freedom I
deliberately left in Meta-SGD's unconstrained rates, now lifted from a diagonal to a structured
transform. I trade the descent guarantee (which I do not want) for the ability to reshape the gradient
however most helps the held-out points (which I do). And because the factors are static across the inner
steps — one `M` reused at every step, exactly as Meta-SGD's `α` was — the transform extends cleanly from
the five training steps to the ten evaluation steps, with no per-step schedule that would have nothing to
say beyond step five.

Here I have to confront a real tension, because the plan is to apply `M` across *all* layers, and ANIL
just showed me that adapting the body is at best useless and at worst the source of the 5-shot
meta-overfitting. Why re-introduce a body inner loop it took three methods to understand and then delete?
The honest answer is a capacity count, and it cuts the other way from intuition. Meta-SGD's diagonal put
one free rate on every body coordinate — on `conv2` alone that is `64·64·9 = 36864` unconstrained
numbers, and across the four conv weights `1728 + 3·36864 = 112320`. The Kronecker factorization on those
same conv weights costs `4186 + 3·8273 = 29005` — about *four times fewer* parameters than Meta-SGD's
diagonal, `36864` versus `8273` on `conv2` itself. So the structured preconditioner is not more capacity
on the body than the rule that overfit; it is *less*, while being strictly more expressive (it can
represent the identity, the diagonal, and off-diagonal channel mixing, whereas the diagonal cannot leave
its own class). The factorization shares statistics across coordinates — one `64×64` input factor governs
all `64·9` input-times-filter slots of every output channel — so it has far fewer independent knobs to
tune to the meta-training episodes. Combined with identity initialization (so an untouched body factor
stays exactly MAML, which freezing showed is a near-no-op on the body) and generalization-trained
factors, the body's re-included inner loop is a controlled bet, not a return to Meta-SGD's failure. It is
still a bet — meta-overfitting from the extra factors is the risk I will watch, the same failure mode
that bit Meta-SGD at 5-shot — but it is a bet made with fewer body parameters than the rule that lost, and
that is what lets me apply the preconditioner network-wide and let the outer-loop-learned features
themselves be shaped by a better-conditioned meta-objective.

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
the literal generalization of the earlier edits: where MAML used a scalar in `maml_update` and
Meta-SGD used a per-parameter vector in `update_module`, I use a per-parameter *transform* and the same
differentiable re-route.

The budget forces the one deviation from the textbook transform, and I should not hand-wave it — I should
add it up. On the conv side the factors are tiny because every conv layer has `C_in, C_out ≤ 64`: the four
conv weights cost `29005`, and even the many 1D parameters (four conv biases plus four BatchNorm
scale-and-shift pairs, a dozen 64-vectors in all) each carry a dense `64×64 = 4096` output factor and sum
to `49152` — that dozen of small vectors, not the conv weights, is actually the single largest slice of
the optimizer state, which surprised me until I saw the `64²` factor is the same size whether the
underlying parameter is a `64×64×9` weight or a bare `64`-vector. The single genuinely oversized factor is
the head's input mode: `C_in = 1600` makes a dense `M_i` a `1600×1600 = 2.56M`-parameter matrix by itself.
Add that and the total optimizer state would be `≈ 2.64M`, twenty-two times the whole `121K` model and
nine times over the `291K` budget — and it is *exactly the cost my factorization was meant to avoid*,
because splitting the input and filter modes was supposed to keep me from ever carrying a single large
`C_in·d` block. So for any mode larger than a threshold `MAX_DENSE = 256` I degrade its factor to a
*diagonal* — a per-element rate over that mode, Meta-SGD's footprint — while keeping the dense factors on
every affordable mode. Concretely the head keeps a dense `5×5` output-mode `M_o` (it can freely recombine
the five class-logit gradients, the cheap and meaningful part for a five-way head) and a diagonal
input-mode rate over the 1600 features, so the head weight costs `25 + 1600 + 1 = 1626` instead of `2.56M`.
With that, the optimizer state lands at `29005 + 49152 + 1626 + 25 = 79808 ≈ 80K` against a model of
`121K`, a total of `≈ 201K` comfortably inside the `291K` budget with about `90K` to spare — full
structured curvature where it is affordable, a diagonal fallback only on the one mode that cannot be. The
factors still initialize to identity (diagonal factors to ones), so the method still begins as MAML. The
full module is in the answer.

So the move takes the axis ANIL left untouched: ANIL fixed *which* parameters adapt and left the
surviving head's step as plain gradient descent; meta-curvature fixes *how* that step moves by
preconditioning the gradient with a learned, Kronecker-factored, generalization-trained curvature —
applied across all layers, not only the head, so the body's outer-loop-learned features are themselves
shaped by a better-conditioned meta-objective. The bar this must clear is ANIL's real numbers, and the
falsifiable claims are pointed at the regime where the *direction* of the step should matter most. At
**1-shot** — the hardest, most ill-conditioned setting, where ten steps from a single example per class
are most wasted on the zig-zag I just quantified — meta-curvature should clear ANIL's 0.4815: a
preconditioned step that points down the valley should extract more from those ten steps than plain SGD,
so I expect 1-shot to be the largest gain, and I would treat *failure* to beat 0.4815 at 1-shot as the
method's central claim falsified. At **5-shot** I expect to at least match ANIL's strong 0.6366
(miniImageNet) and 0.7138 (CIFAR-FS) and likely edge past — and here I would especially watch whether it
finally conquers the miniImageNet 5-shot column that ANIL only tied with MAML (0.6366 vs 0.6379), since
the structured curvature is a strict superset of both the identity (MAML) and diagonal (Meta-SGD) steps
and is meta-trained for generalization rather than support-fitting. The risk I will watch is the same
meta-overfitting that bit Meta-SGD at 5-shot, which the identity initialization, the four-times-smaller
body factor count, and the diagonal fallback on the oversized head mode are meant to contain. Having
exhausted *where to start*, *how far per coordinate*, and *which parameters*, the construction finally
learns *which direction* — the structured, generalization-oriented preconditioner of the inner gradient
that captures the coordinate dependencies every earlier diagonal-or-identity step could not.
