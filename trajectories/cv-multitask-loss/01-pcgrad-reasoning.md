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

But that last sentence is a claim, and before I commit the whole first rung to it I should prove it,
because the entire choice of where to start the ladder turns on whether it is actually true. There are
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
directional disease is *unreachable* by the entire weighting family, and the only way to find out
whether the disease is directional is to fix the direction and see what happens. That is precisely why
I start the ladder here rather than with a weighting scheme — I deliberately pick the rung that
operates one level deeper than weighting, so that if it under-delivers on the magnitude axis (and I
have a specific reason to suspect it will, which I will get to) the diagnosis points cleanly at *what
kind* of intervention the task actually rewards, and the next rung is forced.

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

Before I trust any of that verbally I want to run it on actual numbers, because the projection is easy
to state and easy to get backwards. Take a clean 2-D instance with both a conflict and a magnitude
imbalance, so I am in the interesting part of the triad: `g_fine = g_0 = (2, 1)`,
`g_coarse = g_1 = (−1, 1)`. Their dot product is `2·(−1) + 1·1 = −1 < 0`, so they conflict, as
intended. The squared norms are `‖g_0‖² = 5` and `‖g_1‖² = 2`, and the cosine is
`−1/√(5·2) = −1/√10 ≈ −0.316`. The plain sum is `g_0 + g_1 = (1, 2)` with squared norm `5`, and set
against `‖g_0‖² + ‖g_1‖² = 7` I can see the cancellation directly: the `2·(g_0·g_1) = −2` cross term
has eaten two units of squared length off the honest total. Now project. The de-conflicted fine
gradient is `g_0 − (dot/‖g_1‖²) g_1 = (2,1) − (−1/2)(−1,1) = (2,1) − (1/2, −1/2) = (1.5, 1.5)`, and its
dot with `g_1` is `1.5·(−1) + 1.5·1 = 0` — orthogonal, conflict gone, as the derivation promised. The
de-conflicted coarse gradient is `g_1 − (dot/‖g_0‖²) g_0 = (−1,1) − (−1/5)(2,1) = (−0.6, 1.2)`, and its
dot with `g_0` is `−0.6·2 + 1.2·1 = 0` — orthogonal too. Summing the two projected gradients gives
`g^PC = (1.5 − 0.6, 1.5 + 1.2) = (0.9, 2.7)`, whose squared norm is `0.81 + 7.29 = 8.1`. That number is
the whole point of the exercise: the plain sum had squared length `5`, and the de-conflicted step has
squared length `8.1`, *larger* — the projection has not merely removed a cancellation, it has recovered
length the cancellation was hiding. So the surgery is not a defensive trim; it is a strict escape from
the flattening the plain sum suffers under conflict.

That the length went *up* rather than merely stopped going down is worth understanding, because it
tells me the operation is doing something more than orthogonalization. Write `g^PC = g_0^PC + g_1^PC`
and collect terms: `g^PC = (1 − cos φ/R) g_0 + (1 − cos φ·R) g_1` with `R = ‖g_0‖/‖g_1‖`. Plug in the
numbers to check I have the coefficients right: `R = √5/√2 = √2.5 ≈ 1.581`, so
`cos φ/R = −0.316/1.581 = −0.2` and the coefficient on `g_0` is `1 − (−0.2) = 1.2`; and
`cos φ·R = −0.316·1.581 = −0.5` so the coefficient on `g_1` is `1 − (−0.5) = 1.5`. Reassembling,
`1.2·(2,1) + 1.5·(−1,1) = (2.4, 1.2) + (−1.5, 1.5) = (0.9, 2.7)` — exactly the `g^PC` I got the other
way. So under conflict, where `cos φ < 0`, *both* coefficients exceed 1: PCGrad does not merely cancel
the conflict, it *amplifies* each task's own direction, and that amplification is where the extra
length comes from. It is the heavy-ball-flavoured behaviour that lets the step break out of the valley
the plain sum sits stuck in.

I still owe myself a check that this is a legitimate descent direction and not a plausible-looking hack
that could quietly raise a loss. The cleanest thing I can actually compute is the *first-order* change
each task's loss undergoes under the two candidate steps, because to first order the change in `L_fine`
from a step along direction `d` is `−t (g_fine · d)`, so bigger `g_fine · d` means more descent on
fine. Under the plain sum, `g_fine · (g_0 + g_1) = ‖g_0‖² + dot = 5 + (−1) = 4`. Under PCGrad,
`g_fine · g^PC = g_0 · (g_0^PC + g_1^PC)`, and since `g_1^PC` was built orthogonal to `g_0` the second
term vanishes, leaving `g_0 · g_0^PC = ‖g_0‖² − dot²/‖g_1‖² = 5 − 1/2 = 4.5`. So PCGrad buys `4.5`
units of first-order descent on fine against the plain sum's `4` — strictly more. Repeat for coarse:
plain sum gives `g_coarse · (g_0 + g_1) = ‖g_1‖² + dot = 2 + (−1) = 1`, and PCGrad gives
`g_1 · g_1^PC = ‖g_1‖² − dot²/‖g_0‖² = 2 − 1/5 = 1.8`, again strictly more. This is the real content of
the descent guarantee, and I have now verified it on numbers rather than asserted it: whenever the
conflict term is negative and the projection coefficients stay positive, the de-conflicted step delivers
strictly more first-order descent on *each* task than the plain sum does. The general boundary is that
the improvement on fine is `−dot·(1 + cos φ·R)` and on coarse `−dot·(1 + cos φ/R)`, both positive so
long as the conflict is not so severe relative to the magnitude imbalance that a coefficient flips
negative — which is the near-anti-parallel regime, and there the smooth-loss quadratic upper bound is
the honest backstop: it gives a strict decrease scaling like `(1 − cos²φ)`, vanishing only when the two
gradients are *exactly* anti-parallel, and exact anti-alignment of two stochastic minibatch gradients
essentially never happens. So PCGrad is a descent method; it does not *break* anything, and the
single-step win over the plain sum is exactly the triad regime — conflict, magnitude dominance, high
curvature — recovered from the algebra rather than hoped for.

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

So the delta from the equal-weighting default is: where the default returned `fine_loss + coarse_loss`
and let SGD step along the raw sum, this rung recovers the shared parameters by walking the loss graph,
computes the two per-task gradients, projects away their conflict only when `cos φ < 0`, and returns a
surrogate scalar whose backward pass installs the de-conflicted gradient. Now let me reason about what
I actually expect, because that sets up the next rung, and the two extra backward passes I am paying
for sharpen the stakes. The honest worry is twofold and specific to this task. One: fine and coarse
here are *not* an arbitrary, semantically unrelated pair — coarse is a deterministic *coarsening* of
fine, its label is a fixed many-to-one map of the fine label, so a feature direction that lowers the
fine loss almost always lowers the coarse loss too. That is the definition of correlated gradients,
`cos φ ≥ 0`, and on every such step PCGrad detects no conflict, does nothing, and I am back to the
plain sum having paid triple for the privilege. Two: even on the minority of steps where it fires,
PCGrad fixes the *direction* of the conflict and does nothing about the *magnitude imbalance* between a
100-way and a 20-way cross-entropy — I showed above the two losses sit at genuinely different scales,
and PCGrad never asks how much each task should count, only whether they fight. If the real bottleneck
on fine accuracy is that the coarse task is mis-weighted — too loud early, or starving the fine head of
a scarce trunk's capacity — a direction-only fix leaves that entirely on the table. So my falsifiable
expectation is that this rung lands respectably but *not* at the top, and I can even predict the
*shape* of where it disappoints: the smaller the backbone, the more a scarce trunk's capacity is
contested, and the more a magnitude blindness should cost, so I expect the capacity-scarce ResNet-20 to
be the weakest of the three backbones relative to what a good *weighting* rule would reach there, with
the deeper ResNet-56 and the larger-headed VGG-16-BN less hurt because they have capacity to spare. If
that is what the numbers say — a direction fix that is fine where capacity is abundant and soft exactly
where it is scarce — the next rung writes itself: stop projecting directions and go straight at the
fine/coarse *magnitude* balance this rung deliberately refused to touch.
