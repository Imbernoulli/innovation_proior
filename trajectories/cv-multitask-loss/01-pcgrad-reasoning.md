I have one shared backbone feeding two heads, a fine 100-way head and a coarse 20-way head, and the
only knob I am allowed is how the two cross-entropy losses combine into the one scalar SGD descends.
The scaffold default just adds them, `L = L_fine + L_coarse`, which means the optimizer steps along
`g = g_fine + g_coarse` on the shared trunk. I want to start the ladder from the place where I most
distrust that bare sum, because the auxiliary coarse task is supposed to *help* the fine objective,
and the failure mode I have seen again and again in hard-parameter-sharing setups is that joint
training lands *below* what each task reaches alone — the shared network underperforms exactly the
efficiency it was supposed to buy. So before I reach for any clever reweighting, I want to interrogate
the one operation everything skips over: the *combination* of the two gradients. The optimizer never
sees `g_fine` and `g_coarse` separately; it sees their sum and steps along it. If anything is broken,
it has to be visible in the geometry of that sum, and a fix that only rescales the losses can never
reach a problem that lives in directions.

That last sentence is a claim, and it fixes where I start, so it is worth settling. There are
really only two families of intervention available to me. One is the *weighting* family: replace the
sum by `L = w_fine L_fine + w_coarse L_coarse` with nonnegative weights — static, grid-searched,
learned, or set by some rule I have not written yet — which scales the two loss magnitudes before adding. The other
is the *geometry* family: leave the losses alone but modify the *directions* of the two gradients
before they are combined. The question is whether the weighting family can reach a directional
disease, and the answer falls out of one line of algebra. Under weighting the shared step becomes
`w_fine g_fine + w_coarse g_coarse`, and the inner product of the two weighted gradients is
`(w_fine g_fine) · (w_coarse g_coarse) = w_fine w_coarse (g_fine · g_coarse)`. Since both weights are
nonnegative, this has exactly the same sign as `g_fine · g_coarse`. Worse — better, for my argument —
the *cosine* between the two weighted gradients is
`(w_fine w_coarse g_fine · g_coarse)/(w_fine ‖g_fine‖ · w_coarse ‖g_coarse‖) = cos φ`, the weights
cancel entirely and the angle between the two gradients is left *unchanged*. Reweighting slides each
gradient along its own fixed ray; it never rotates one toward or away from the other. So if the two
gradients point partly against each other, no choice of nonnegative weights can stop them pointing
against each other; the overlap that cancels in the sum stays exactly as it was. That settles it: a
directional disease is *unreachable* by the entire weighting family, so the only way to learn whether
the disease is directional at all is to fix the direction and watch. That is why I start with geometry
rather than a weighting scheme — it operates one level deeper than weighting, so if it under-delivers on
the magnitude axis the diagnosis points cleanly at what kind of intervention the task actually rewards.

So let me stare at the geometry. Two gradient vectors `g_fine`, `g_coarse` in the same shared-trunk
parameter space; the step goes along `g_fine + g_coarse`. When are they fighting? When they point
partly against each other — negative inner product, `g_fine · g_coarse < 0`, equivalently negative
cosine `cos φ = (g_fine · g_coarse)/(‖g_fine‖‖g_coarse‖) < 0`. Call that *conflicting*. In that case
`‖g_fine + g_coarse‖² = ‖g_fine‖² + ‖g_coarse‖² + 2 g_fine · g_coarse` is strictly *less* than
`‖g_fine‖² + ‖g_coarse‖²`: the overlap cancels. But I should be careful — is conflict by itself fatal?
If two opposing gradients are of equal size and I step along their sum, the sum still points downhill
on `L = L_fine + L_coarse`; I still decrease the combined objective. Conflict alone is not the disease.
What makes it hurt is co-occurrence with two other things. First, *magnitude dominance*: suppose on
this task the coarse loss, being a 20-way problem, is easier and smaller, while the fine 100-way loss
is large — then `g = g_fine + g_coarse` is essentially `g_fine`, the coarse signal barely registers,
and because they conflict, that step actively moves *against* the coarse task while the fine task
drags the trunk. Worse for the symmetric story, the same asymmetry can run the other way at a
different stage of training, and whichever task dominates pushes the shared features in a direction
that *undoes* the other. Second, *high curvature*: a gradient step trusts a local linear model, and in
the narrow, steeply-curved valleys that deep nets live in, that model overestimates the gain on the
dominating task and underestimates the harm to the dominated one — the optimizer takes a trade it
thinks is good and that is actually bad. So the disease is the triad — conflicting gradients *and* one
dominating *and* high curvature — and when all three hit at once the summed gradient walks the
optimizer into a bad trade it cannot see, and joint training stalls below single-task. I can even put
a rough number on the first two-thirds of the triad: at init the fine head faces a 100-way problem so
its cross-entropy sits near `log 100 ≈ 4.6`, the coarse head a 20-way problem near `log 20 ≈ 3.0`, and
those two scales feed gradients of correspondingly different magnitude into the same trunk, so
dominance is not hypothetical on this benchmark — it is baked into the class counts.

So let me design the geometric intervention from scratch for these two gradients and see what is
forced. I want to modify `g_fine` to remove only its conflict with `g_coarse`, changing it as little
as possible otherwise. The conflict between them lives entirely in the component of `g_fine` that lies
*along* `g_coarse`, because the inner product only sees that component. Decompose `g_fine` into the
part parallel to `g_coarse` and the part orthogonal to it. The parallel part is
`((g_fine · g_coarse)/‖g_coarse‖²) g_coarse` — the standard projection of `g_fine` onto `g_coarse`.
When they conflict, `g_fine · g_coarse < 0`, so this parallel part points *opposite* to `g_coarse`:
it is exactly the piece of `g_fine` that is undoing the coarse task's progress. Strip it out,
`g_fine ← g_fine − ((g_fine · g_coarse)/‖g_coarse‖²) g_coarse`, and what is left is `g_fine` projected
onto the *normal plane* of `g_coarse`. The new `g_fine` now has zero inner product with `g_coarse`, so
it no longer conflicts, and I changed it by the minimum amount needed (I removed precisely the
offending parallel component and nothing else). No quadratic program, no line search — it is one inner
product and one axpy, a closed form. And crucially I do this *only* when they conflict: if
`cos φ ≥ 0` the gradients already cooperate, the parallel component is *helping*, pulling fine and
coarse in compatible directions, and ripping it out would destroy the positive transfer that is the
entire reason I am sharing a trunk. So the rule is conditional — project when `g_fine · g_coarse < 0`,
leave alone otherwise — which is the multi-task analogue of the conflict-detection-by-inner-product
idea from continual learning (project away a step that would raise a protected task's loss), but made
*symmetric*: both fine and coarse are first-class here, so I project `g_coarse` onto the normal plane
of `g_fine` as well, by the same formula. With only two tasks there is no loop and no task ordering to
worry about; the projection is a single symmetric operation, and I do not need the random task-order
shuffling that a many-task version would require to stay unbiased. This is *projecting conflicting
gradients* — PCGrad, in its two-task form.

The projection is easy to state and easy to get backwards, so I run it on numbers. Take a 2-D instance
with both a conflict and a magnitude imbalance, the interesting part of the triad: `g_fine = g_0 =
(2, 1)`, `g_coarse = g_1 = (−1, 1)`. Their dot is `−1 < 0`, so they conflict; the squared norms are `5`
and `2` and the cosine is `−1/√10 ≈ −0.316`. The plain sum `g_0 + g_1 = (1, 2)` has squared norm `5`
against an honest `‖g_0‖² + ‖g_1‖² = 7` — the `2(g_0·g_1) = −2` cross term has eaten two units of length.
Projecting, `g_0^{proj} = (2,1) − (−1/2)(−1,1) = (1.5, 1.5)` (dot with `g_1` now `0`) and `g_1^{proj} =
(−1,1) − (−1/5)(2,1) = (−0.6, 1.2)` (dot with `g_0` now `0`). Their sum `g^PC = (0.9, 2.7)` has squared
norm `8.1` — *larger* than the plain sum's `5`. The projection has not merely removed a cancellation, it
has recovered length the cancellation was hiding; the surgery is a strict escape from the flattening the
plain sum suffers under conflict, not a defensive trim.

The length going *up* rather than merely stopping its fall tells me the operation is more than
orthogonalization. Collecting terms, `g^PC = (1 − cos φ/R) g_0 + (1 − cos φ·R) g_1` with
`R = ‖g_0‖/‖g_1‖`; on the example `R ≈ 1.581` and `cos φ ≈ −0.316` give coefficients `1.2` on `g_0`
and `1.5` on `g_1`. Under conflict, where `cos φ < 0`, *both* coefficients exceed 1: PCGrad does not
merely cancel the conflict, it *amplifies* each task's own direction, and that amplification is where
the extra length comes from — the heavy-ball-flavoured behaviour that breaks the step out of the valley
the plain sum sits stuck in.

I do need to make sure the de-conflicted step is genuine descent and not a hack that could quietly raise
a loss. To first order the change in `L_fine` along direction `d` is `−t (g_fine · d)`, so `g_fine · d`
measures descent on fine. Under the plain sum `g_0·(g_0 + g_1) = 5 + (−1) = 4`; under PCGrad, `g_1^PC`
is orthogonal to `g_0`, so `g_0·g^PC = ‖g_0‖² − dot²/‖g_1‖² = 5 − 1/2 = 4.5` — strictly more. For coarse
the plain sum gives `2 + (−1) = 1` and PCGrad gives `2 − 1/5 = 1.8`, again more. So whenever the conflict
term is negative and the projection coefficients stay positive, the de-conflicted step delivers strictly
more first-order descent on *each* task than the plain sum. The one edge is the near-anti-parallel regime
where a coefficient could flip; there the smooth-loss quadratic bound still gives a decrease scaling like
`(1 − cos²φ)`, vanishing only at exact anti-alignment, which two stochastic minibatch gradients
essentially never hit. So PCGrad is a genuine descent method, and its single-step win over the plain sum
is exactly the triad regime — conflict, dominance, curvature.

None of this is free, and I want the cost on the table before I commit. The plain sum costs one forward
and one backward per step. PCGrad, as I am about to implement it, costs one forward, then two calls to
`torch.autograd.grad` (one per task, each roughly a backward pass over the shared graph), then the
backward on the surrogate — which is trivial, since the surrogate is linear in the parameters and its
backward does not re-traverse the trunk. So the real tax is the two extra graph traversals: roughly
three backward-equivalents per step where the plain sum needed one. With a backward costing on the
order of twice a forward, that pushes the per-step wall-clock to something like two to three times the
baseline. I will pay that only if the de-conflicting actually earns its keep — and here is the first
crack of doubt, which I hold onto rather than suppress: if the two gradients rarely conflict, I am
paying triple for a projection that almost never fires.

Now I have to land this in *this task's* edit surface, and here is where the implementation is forced
into a shape that differs from a generic PCGrad wrapper. The wrapper version assumes it owns the
optimizer: it sits between `backward` and `step`, reads each task's `.grad` off the parameters, does
the projection, writes the corrected gradient back, and lets the base optimizer step. But the scaffold
gives me none of that. The only thing I get to fill is `MultiTaskLoss.forward(fine_loss, coarse_loss,
epoch, total_epochs)`, and it must *return a scalar* on which the loop calls `.backward()`. I am handed
the two already-reduced scalar losses, not the shared parameters, not the optimizer, not the per-task
`.grad` fields. So I cannot run the wrapper's read-modify-write-then-step protocol. I have to do all of
PCGrad *inside* `forward`, before the loop's single `.backward()`. Two consequences follow, and both
are real departures from the wrapper form that I must reason out, not import.

First, I have to *recover the shared parameters myself*, because the interface never gives them to me.
Both `fine_loss` and `coarse_loss` are tensors with a `grad_fn`, so the autograd graph that produced
them is reachable from them. I can walk that graph backward from `fine_loss.grad_fn`, following
`next_functions`, and collect every leaf node that carries a `.variable` which `requires_grad` — those
leaves are exactly the model parameters that fed the loss. The fine loss's graph touches the shared
trunk plus the fine head; that set of leaves is what I treat as the parameters to de-conflict over,
which is the right set precisely because the trunk is the shared surface where coarse's gradient can
collide with fine's — the conflict I care about lives on exactly those leaves. One correctness detail
lets me cache: the network's graph *topology* is identical on every minibatch (same modules, same
connectivity, only the tensor values change), so the leaf set I collect on the first call is the leaf
set forever, and I can store it after that first walk and never re-traverse — turning an `O(graph)`
walk into a one-time cost rather than a per-step one.

Second, having the parameters, I compute the two per-task gradients *explicitly* with
`torch.autograd.grad(fine_loss, params, retain_graph=True)` and the same for `coarse_loss` — not by
calling `.backward()`, because the loop owns the one `.backward()` and I must not consume the graph or
populate `.grad` prematurely. `allow_unused=True` and a zero fill handle parameters a given task does
not touch (the fine head contributes no gradient to the coarse loss, and vice versa). I flatten both
into vectors `g0`, `g1`, run the single symmetric projection when `dot = g0·g1 < 0` — using the
*originals* of both for the two projections so the operation is symmetric (`g0_proj = g0 − (dot/‖g1‖²)
g1`, `g1_proj = g1 − (dot/‖g0‖²) g0`) — and form `g_pcgrad = g0 + g1`. Small `+1e-12` floors on the
squared norms keep the division safe, which matters more than it looks: near a minimum the gradients
shrink toward zero and the raw `1/‖g‖²` would blow up exactly when the surgery should be doing nothing.

The last piece is the trick that lets a projected gradient survive the interface's "return a scalar"
contract, and I want to be sure it actually computes what I think. I have a desired gradient
`g_pcgrad`, but the loop is going to call `.backward()` on whatever scalar I return and then `step()`.
So I construct a *surrogate loss* whose gradient with respect to each parameter is exactly the
corresponding chunk of `g_pcgrad`: take the de-conflicted gradient chunk for parameter `p`, **detach**
it, and add `(chunk · p).sum()` to the surrogate. Differentiate: `∂/∂p Σ_k chunk_k p_k = chunk`, since
`chunk` is now a constant, so backpropagating the returned surrogate deposits `g_pcgrad` into every
`p.grad` and the optimizer steps along the de-conflicted direction — exactly as if a wrapper had
written the corrected gradient back. The `detach` is not cosmetic and I should say why: `chunk` is a
*function of* `p`, because it came out of `g_fine`/`g_coarse`, which came out of the network's forward
pass on `p`. If I left it attached, `∂/∂p (chunk(p) · p)` would pick up a spurious `(∂chunk/∂p)·p` term
and install a gradient that is not `g_pcgrad` at all — the second-order leakage would quietly corrupt
the very direction I worked to compute. Detaching freezes `chunk` into the constant target it is meant
to be. This surrogate-loss device is the bridge from "I can only return a scalar" to "I need to impose
a specific gradient," and it is the defining feature of how PCGrad has to be expressed *here* rather
than as the external wrapper. (The full scaffold module is in the answer.)

So the delta from the equal-weighting default: recover the shared parameters by walking the loss graph,
project away the two gradients' conflict only when `cos φ < 0`, and return a surrogate scalar whose
backward installs the de-conflicted gradient. What do I actually expect, given the two extra backward
passes I am paying for? Two worries are specific to this task. First, coarse is *not* an arbitrary
second task — it is a deterministic *coarsening* of fine, a fixed many-to-one map of the fine label, so
a feature direction that lowers the fine loss almost always lowers the coarse loss too. That is
correlated gradients, `cos φ ≥ 0`, and on every such step PCGrad detects no conflict, does nothing, and
I am back to the plain sum having paid triple. Second, even when it fires, PCGrad fixes the *direction*
of the conflict and does nothing about the *magnitude imbalance* between a 100-way and a 20-way
cross-entropy — it never asks how much each task should count, only whether they fight. If the real
bottleneck on fine accuracy is that the coarse task is mis-weighted, starving the fine head of a scarce
trunk's capacity, a direction-only fix leaves that on the table. So my falsifiable expectation is that
this lands respectably but *not* at the top, and I can predict the shape of the disappointment: the
smaller the backbone, the more a scarce trunk's capacity is contested, so magnitude blindness should
cost most on the capacity-scarce ResNet-20 and least on the deeper ResNet-56 and larger-headed
VGG-16-BN, which have capacity to spare. If that is the pattern — a direction fix that is fine where
capacity is abundant and soft where it is scarce — the next move is forced: stop projecting directions
and go straight at the fine/coarse *magnitude* balance this attempt deliberately refused to touch.
