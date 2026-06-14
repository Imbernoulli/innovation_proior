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
optimizer into a bad trade it cannot see, and joint training stalls below single-task.

Now what do the magnitude-only fixes do, and why don't they reach this? Everything in the
loss-weighting family — learning a per-task variance, balancing the per-task loss magnitudes, any
`L = Σ w_i L_i` with nonnegative `w_i` — is a *scaling* operation. If `g_fine` and `g_coarse`
conflict, then `w_fine g_fine + w_coarse g_coarse` is still two vectors pointing partly against each
other, still cancelling in the overlap. No choice of nonnegative weights can make a conflicting
*direction* stop conflicting; reweighting changes how much each fighter weighs, it does not stop the
fight. That is precisely the gap I want this first rung to attack: the cancellation is geometric, so
the fix has to touch the geometry, not the magnitudes. This is also why I start the ladder here rather
than with a weighting scheme — I want the rung that operates one level deeper than weighting, so that
when it underperforms (and I suspect it will, for reasons I will get to) the diagnosis points cleanly
at *what kind* of intervention the task actually rewards.

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

The combined step is then `Δθ = g_fine^PC + g_coarse^PC`. It is worth confirming this is a legitimate
descent direction and not a plausible hack. In the clean conflicting case, write `g_0 = g_fine`,
`g_1 = g_coarse`, the de-conflicted update is `g^PC = g_0 + g_1 − ((g_0·g_1)/‖g_1‖²) g_1 −
((g_0·g_1)/‖g_0‖²) g_0`. Using the heavy-ball-flavoured rewrite `g^PC = (1 − cos φ/R) g_0 +
(1 − cos φ·R) g_1` with `R = ‖g_0‖/‖g_1‖`, and noting that under conflict `cos φ < 0` makes both
coefficients exceed 1, PCGrad does not merely cancel the conflict — it *amplifies* each task's own
direction, which is exactly what lets the step escape the cancellation that flattens the plain sum.
With a smooth loss (Lipschitz gradient `L`, step `t ≤ 1/L`) the standard quadratic upper bound, after
the algebra collapses, gives `L(θ⁺) ≤ L(θ) − ½ t (1 − cos²φ) ‖g‖²`: as long as `cos φ > −1` the
factor `(1 − cos²φ)` is strictly positive, so the objective strictly decreases unless `g = 0` or the
two gradients are *exactly* anti-parallel — and exact anti-alignment of two stochastic minibatch
gradients essentially never happens. So PCGrad is a descent method; it does not *break* anything. The
single-step win over the plain sum is exactly the triad regime: it provably reaches a lower loss than
the summed-gradient step when the gradients conflict enough (`cos φ ≤ −Φ`, with the magnitude
similarity `Φ = 2‖g_0‖‖g_1‖/(‖g_0‖²+‖g_1‖²)` fusing "they conflict" with "their magnitudes differ"),
the curvature is high enough, and the step is large enough that the curvature mis-estimation actually
bites. That is the precise condition I want a geometry fix to win under, recovered from the algebra.

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
leaves are exactly the model parameters (and the two heads) that fed the loss. The fine loss's graph
touches the shared trunk plus the fine head; that set of leaves is what I treat as the parameters to
de-conflict over. I cache this list after the first call so I am not re-walking the graph every step.

Second, having the parameters, I compute the two per-task gradients *explicitly* with
`torch.autograd.grad(fine_loss, params, retain_graph=True)` and the same for `coarse_loss` — not by
calling `.backward()`, because the loop owns the one `.backward()` and I must not consume the graph or
populate `.grad` prematurely. `allow_unused=True` and a zero fill handle parameters a given task does
not touch (the fine head contributes no gradient to the coarse loss, and vice versa). I flatten both
into vectors `g0`, `g1`, run the single symmetric projection when `dot = g0·g1 < 0` — using the
*originals* of both for the two projections so the operation is symmetric (`g0_proj = g0 − (dot/‖g1‖²)
g1`, `g1_proj = g1 − (dot/‖g0‖²) g0`) — and form `g_pcgrad = g0 + g1`. Small `+1e-12` floors on the
squared norms keep the division safe.

The last piece is the trick that lets a projected gradient survive the interface's "return a scalar"
contract. I have a desired gradient `g_pcgrad`, but the loop is going to call `.backward()` on whatever
scalar I return and then `step()`. So I construct a *surrogate loss* whose gradient with respect to
each parameter is exactly the corresponding chunk of `g_pcgrad`: take the de-conflicted gradient chunk
for parameter `p`, **detach** it (it is now a constant target, not part of the graph), and add
`(chunk · p).sum()` to the surrogate. Then `∂(chunk·p)/∂p = chunk`, so when the loop backpropagates
the returned surrogate it deposits `g_pcgrad` into every `p.grad`, and the optimizer steps along the
de-conflicted direction — exactly as if a wrapper had written the corrected gradient back. This
surrogate-loss device is the bridge from "I can only return a scalar" to "I need to impose a specific
gradient," and it is the defining feature of how PCGrad has to be expressed *here* rather than as the
external wrapper. (The full scaffold module is in the answer.)

So the delta from the equal-weighting default is: where the default returned `fine_loss + coarse_loss`
and let SGD step along the raw sum, this rung recovers the shared parameters by walking the loss graph,
computes the two per-task gradients, projects away their conflict only when `cos φ < 0`, and returns a
surrogate scalar whose backward pass installs the de-conflicted gradient. Now let me reason about what
I actually expect, because that sets up the next rung. The honest worry is twofold and specific to
this task. One: fine and coarse here are *not* an arbitrary, semantically unrelated pair — coarse is a
deterministic *coarsening* of fine, so their gradients are likely *correlated*, not conflicting, much
of the time. If `cos φ ≥ 0` on most steps, PCGrad does nothing and I am back to the plain sum, paying
two extra backward passes per step for no change. Two: PCGrad fixes the *direction* of the conflict but
does nothing about the *magnitude imbalance* between a 100-way and a 20-way cross-entropy — it never
asks how much each task should count, only whether they fight. If the real bottleneck on fine accuracy
is that the coarse task is mis-weighted (too loud early, or starving the fine head of capacity), a
direction-only fix leaves that on the table. So my falsifiable expectation is that this rung lands
respectably but *not* at the top — that across ResNet-20, ResNet-56, and VGG-16-BN it does not clearly
beat what a good *weighting* rule would do, precisely because on a hierarchical fine/coarse pair the
gradients rarely conflict and the lever that matters is the relative weight, which PCGrad never
touches. If that is what the numbers say, the next rung writes itself: stop projecting directions and
start *learning the weights*, with a principled penalty that keeps them from collapsing.
