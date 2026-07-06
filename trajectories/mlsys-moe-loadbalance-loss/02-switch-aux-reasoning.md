The control did exactly what I feared, and the two numbers it handed back are worth reading together
before I do anything, because they define the shape of the problem. The imbalance came in at
`L_imb = 0.1286` on a perplexity of `41.594`, i.e. a cross-entropy of `3.7280` — and `exp(3.7280)`
is indeed `41.60`, so the two readings are consistent and I am reading a genuinely trained model, not
an artifact. Now place `0.1286` on the scale I worked out for this metric. The reachable ceiling
under top-two routing is `0.75` (all traffic on a single pair of experts), so the control sits at
about a sixth of the way to full collapse; against my calibration anchor — a conspicuous two-fold
overload of a pair of experts reads `0.25` — it is roughly half of that, a mild-to-moderate skew.
And it sits on a cross-entropy that is perfectly ordinary: a branching factor in the low forties, a
competent model. So the control confirmed the diagnosis in exactly the joint shape I predicted — a
fine cross-entropy over a clearly skewed histogram — and it did it above any plausible sampling
floor, so the skew is real and not finite-batch noise. The disease is real, the cross-entropy is
structurally blind to it, and the cure cannot come from the cross-entropy. I have to add a term, by
hand, that looks at the routing distribution itself and pushes it toward uniform. The question is
what that term should be, and it is not obvious, because the most natural thing to penalize is
exactly the thing I cannot differentiate.

Let me see the obstacle cleanly first, because it eliminates most of the candidates before I even
list them. What I actually want to control is the fraction of tokens each expert receives, `f_i`. If
I could just penalize how far the vector of `f_i` sits from uniform — say `Σ_i (f_i − 1/N)^2`, or the
`L_imb` I already measure — I would be done in one line. But `f_i` is a hard count: it comes from the
top-K selection, an `argmax`-like operation, and counts are flat with jumps. Nudge a router weight
and, for a while, every token keeps its same top-two experts, so `f` does not move and its gradient
is exactly zero; then one token's ranking flips and a count jumps by one. A penalty written purely on
`f` is piecewise-constant, so it hands the router no gradient to follow — it is either silent or
discontinuous, never a slope. So the entire family of "just penalize the counts" losses is dead on
arrival, however natural it looks. Whatever I write has to *correlate* with the count imbalance but
carry its gradient through something differentiable, and the only differentiable thing the router
exposes is its softmax probability vector `P` — the continuous mass `P_i` it puts on each expert
before the hard selection collapses it into a choice.

That leaves a small design space of differentiable balancers, and I want to walk it rather than jump.
One option is to forget the counts entirely and penalize only the probabilities: something like
`Σ_i (P_i − 1/N)^2`, or equivalently pushing every `P_i` toward `1/N`. This is differentiable and its
minimum is uniform, but it aims at the wrong target. It flattens the router's probability *per token*,
which is precisely the sharp, content-dependent routing I argued I must protect — it manufactures
balance by destroying specialization, the exact conjunction I do not want to break. A second option
is the "importance" penalty `Σ_i P_i^2` on its own: it is smooth, minimized at uniform, and it does
push mass off dominant experts. But it is ungrounded in what the router actually *did* — it can be
satisfied by a router that spreads its probability while its hard top-two still lands on a favored
few, because `P` and the realized counts can diverge. It penalizes intended imbalance but not
necessarily realized imbalance. What I want is a term that is anchored to the real usage `f` — so it
fires when experts are actually overloaded — yet flows its gradient through `P` so it is trainable.
The construction that does both is the product, summed over experts: form `f_i · P_i` for each
expert, the fraction it actually received times the average probability the router assigned it, and
sum. Treat the `f_i` as fixed weights — detach them, their own gradient is useless anyway — and let
the gradient flow only through `P_i`. The counts steer *where* the pressure points; the probability
is *how* the pressure is applied.

The reason I insist on anchoring to `f` rather than trusting `P` alone is worth making concrete,
because it is what kills the importance-only option. Imagine a router whose probability vector is
fairly flat — `P` close to uniform — but whose top-two selection, because two experts sit a
persistent hair above the rest, still lands almost every token on the same pair. Then `Σ_i P_i^2` is
near its floor and the importance penalty cheerfully reports "balanced," while the realized counts
`f` are badly skewed and the true `L_imb` is large. A penalty built on `P` alone would be satisfied
by a router that is, in the only sense that matters for load, collapsed. Multiplying by the realized
`f` closes exactly that loophole: `f_i P_i` can only be small where the router *both* assigns little
probability *and* sends few tokens, so the term stays large until the experts the router actually
overloads are the same ones it is pulling probability off. `f` grounds the penalty in what happened;
`P` makes it trainable; neither alone is enough, and that is the whole reason the object is a product
and not either factor by itself.

I want to verify that this actually applies the pressure I claim, because a surrogate that pushes the
wrong way would be worse than none. So let me differentiate. With `P = softmax(z)` the Jacobian is
`∂P_i/∂z_j = P_i(δ_ij − P_j)`, so for the single-layer term `T = N Σ_i f_i P_i` with `f` detached,
`∂T/∂z_j = N Σ_i f_i P_i(δ_ij − P_j) = N[f_j P_j − P_j Σ_i f_i P_i] = N P_j (f_j − m)`, where I write
`m = Σ_i f_i P_i` for the `P`-weighted average of the counts. Read that gradient. Descending it moves
the logit `z_j` *down* when `f_j > m` — the experts whose realized load is above the weighted mean,
the hot ones — and *up* when `f_j < m` — the cold, underused ones. That is exactly the corrective
pressure I wanted: the router is pulled to move probability off the experts it is overloading and
onto the ones it is starving, and the only experts it leaves alone are those already near the mean.
The construction is doing the right thing, and I did not have to assume it — the sign of `f_j − m`
guarantees it.

But the same formula carries a warning I should note now, because it will matter two rungs from now.
The magnitude of the push on expert `j` scales with `P_j`. An expert that is nearly dead — one the
router has already driven to `P_j ≈ 0` — gets a gradient of nearly zero *even though* it is the one
most in need of rescue, because the `P_j` prefactor and its being cold both shrink the term. So this
surrogate resurrects mildly-underused experts eagerly and severely-starved experts feebly; its grip
weakens exactly in the tail where collapse is worst. I will not do anything about that here — this
rung is about establishing that the `f·P` surrogate works at all — but I am flagging it as a
structural property of the penalty, not a tuning accident.

To make sure that warning is real and not just algebra, let me put numbers through the formula. Take
a moderately skewed router where usage tracks probability, `f = P = (0.30, 0.25, 0.15, 0.10, 0.08,
0.06, 0.04, 0.02)`. Then `m = Σ_i f_i P_i = Σ_i P_i^2 = 0.197`, and the per-expert push `N P_j(f_j −
m)` comes out to `+0.247` and `+0.106` on the two hot experts — positive gradient, so descent
suppresses them — and negative, i.e. lifting, on all the rest. The instructive part is the size of
the lift across the cold tail. The middling-cold expert at `f = 0.06` receives a lift of magnitude
`0.066`; the coldest expert at `f = 0.02`, the one nearest death, receives only `0.028` — a factor
of `2.3` *weaker* pull on the expert that needs it most. That inversion is the `P_j` prefactor at
work: the coldest expert has the largest deficit `f_j − m = −0.177` but the smallest probability to
multiply it by, and the product loses to the warmer expert whose deficit is smaller but whose `P_j`
is three times larger. So the surrogate is real and it does steer toward uniform, but numerically it
rescues the recoverable experts far harder than the nearly-dead ones. I am not going to fix that here;
I am recording that the number, not just the intuition, says the grip is weakest in the tail.

There is a subtlety in detaching `f` that I should convince myself about, because freezing the counts
each step looks at first as if it could aim the penalty at the wrong place. Within a single step, with
`f` held fixed, the gradient of `Σ_i f_i P_i` in `P` alone is minimized not by going uniform but by
dumping all probability onto whichever expert currently has the smallest `f_i` — the objective for one
frozen step points at the coldest corner, not the center. If `f` never updated, the router would
overshoot. But `f` is recomputed from the routing every step, so the real process is an alternation:
measure the counts, nudge `P` toward the currently-cold experts, let the realized counts shift toward
those newly-warmed experts, measure again. That is coordinate descent between the count vector and the
probability vector, and its only rest point is where nudging `P` stops changing the objective — which
by the gradient formula `N P_j(f_j − m)` is precisely where `f_j = m` for every `j`, i.e. where usage
is uniform. So the detached-`f` scheme converges to uniform *usage*, not merely uniform probability,
even though no single step's frozen objective points there. Detaching is therefore not an
approximation I am grudgingly tolerating; it is the right thing to do, because `f`'s own gradient is
the useless zero-or-jump of the count, and letting it flow would inject nothing but noise at the
boundary crossings while adding no information the alternation does not already supply.

Two scaling choices make the surrogate well-behaved, and I want to check both by hand rather than
inherit them. First, is uniform routing really the optimum, or have I built a penalty whose minimum
sits somewhere useless? The detached-`f` view says: at a uniform allocation `f_i = 1/N` for all `i`,
the weighted mean `m = Σ (1/N) P_i = 1/N`, so `f_j − m = 0` for every `j` and the gradient vanishes —
the penalty stops pushing exactly at balance, which is what I need. The coupled view is sharper.
Realized usage tracks probability — an expert the router prefers gets routed to more — so to first
order `f_i ≈ P_i`, and then `Σ_i f_i P_i ≈ Σ_i P_i^2`. By Cauchy–Schwarz (or just the fact that the
mean of squares is at least the square of the mean), `Σ_i P_i^2 ≥ (Σ_i P_i)^2 / N = 1/N`, with
equality if and only if `P` is uniform. So the penalty really is minimized at uniform and nowhere
else. Second, the value at that minimum is `1/N`, which drifts with the number of experts; to make
the balanced optimum scale-free I multiply the whole thing by `N`, so `N Σ_i P_i^2` reads `1` at
uniform regardless of `N`. It is worth seeing the range this normalized penalty spans: at uniform,
`N Σ P_i^2 = 8·(1/8) = 1`; at a two-expert collapse `P = (0.5, 0.5, 0, …)`, `N Σ P_i^2 = 8·0.5 = 4`;
at the (unreachable-under-top-2) one-expert limit it would be `8`. So the penalty ranges from `1` to
`4` across the physically reachable configurations, a factor of four of headroom for the gradient to
work with.

That headroom then gets multiplied by the coefficient `α`, and I want the budget arithmetic explicit
so I know the penalty breaks collapse without dictating predictions. The literature value is around a
hundredth, and I can see why from the numbers just computed. With `α = 10^{-2}`, the penalty
contributes between `α·1 = 0.01` (at balance) and `α·4 = 0.04` (at a two-expert collapse) to the
total loss. Against a cross-entropy of order a few nats — the control's was `3.7280` — that is at
most a roughly one-percent addition to the objective. So the balancing term is strong enough to
register a real gradient against the router (the `0.04` at collapse is not negligible next to the
gradients the router already feels) but far too weak to start overriding the cross-entropy and
distorting the predictions. Push `α` an order of magnitude higher and it would begin to dictate the
router's choices and drag CE up; drop it an order lower and it would be lost in the noise of the LM
gradient. `α = 10^{-2}` is the value that fits between those failures, and the numbers above are why.
The penalty is averaged over the two MoE layers so a deeper stack does not silently scale the weight.
That averaging is not just a normalization convenience: each of the two MoE layers has its own router
and its own histogram, and there is no reason their skews are aligned — the first layer might collapse
onto experts the second leaves cold. Computing `f·P` per layer and averaging keeps each layer
answerable for its own balance, whereas pooling the counts across layers before penalizing would blur
two independent balance problems into one and could let a well-spread layer statistically mask a
collapsed one. So the mean-over-layers is the version that penalizes each router for the histogram it
actually produced.

There is one more decision, and it is the one I expect to come back and bite, so let me name it now
rather than discover it later: over what set of tokens do I compute `f_i`? The obvious answer, and
the one the original Switch and GShard losses take, is the micro-batch — the handful of tokens
processed together in one forward pass on one device. It is the cheapest and most local choice and it
is what the classical loss does. But I can already feel the problem. A micro-batch is a small, noisy
sample of the corpus, and it may be genuinely lopsided in content — a slice that happens to be all
one latent topic — for which a *skewed* expert usage is the correct, specialized behavior. Forcing
the `f_i` of that slice toward uniform punishes the router for doing the right thing on that slice.
The micro-batch penalty cannot tell the difference between collapse, which I want to stop, and
legitimate per-slice specialization, which I want to keep; it enforces the right constraint at the
wrong scope. In this single-process reproduction I emulate the micro-batch by splitting each training
batch into four micro-splits, computing the penalty on each, and averaging — which is quantitatively
stricter than one global constraint, since I am now demanding uniformity of four separate quarters
rather than of the whole, and those four demands can conflict with one another when the quarters
differ in content.

I can be concrete about how that over-constraint turns into a cross-entropy cost, because it is a
competition between two gradients on the same slice. Suppose one micro-split happens to be dominated
by a single latent topic. The cross-entropy gradient on that split wants the router to send those
tokens sharply to the two experts that have specialized on that topic — that is what lowers the loss
on those tokens — so it drives `f` on the split *away* from uniform. The micro-batch balancing
gradient on the same split does the opposite: by the formula above it reads the topic-experts as hot,
`f_j > m`, and pushes probability off them and onto the idle experts, back toward uniform. The two
gradients partially cancel on that split, and the router settles at a compromise that is neither as
specialized as the cross-entropy alone would make it nor as flat as the penalty alone would demand.
The residual mis-specialization is a small cross-entropy tax levied on every slice that is internally
lopsided, summed over four splits a batch and fifteen hundred steps. So I suspect this loss will
balance the load — it will clearly beat the control on imbalance — but partly by flattening
specialization, and that flattening should show up as a cross-entropy that is not quite as low as it
could be. Whether the tax is large enough to *see* at this scale is genuinely uncertain: the slices
here are quarters of a small batch and may be too alike in content for the effect to register in the
CE number at all. But the sign of the effect is not in doubt, and that is exactly what makes the
granularity the one thing worth changing next.

There is one more mismatch worth naming, because it colors how I read the eval number against the
training penalty. The penalty trains on micro-batch counts, but the `L_imb` I will be scored on is
measured over held-out evaluation batches at a larger, more global scope. Those two are not the same
constraint: a router driven uniform on every quarter of a batch is automatically uniform on the whole
batch — the whole-batch counts are just the sum of the quarters' — but the converse does not hold, so
the micro-batch penalty is strictly *stronger* than the whole-batch uniformity the metric actually
rewards. That mismatch cuts two ways at once, and seeing both is the point. It flatters the `L_imb`
number, because I am over-solving the very quantity being measured; and it is exactly the source of
the specialization tax on `L_CE`, because the extra strictness buys me nothing the metric wants while
costing me the per-slice sharpness I need. So the micro-batch scope is not merely a different choice —
it is stricter than either the goal or the metric requires, which is the cleanest possible statement
of why it is the wrong knob and why the fix is to relax the scope, not the form.

Before I commit, one arithmetic on the fitness, so I know what a success looks like in `r` and not
just in `L_imb`. The control gave `r = −3.8566`, which is exactly `−(3.7280 + 0.1286)`. If this rung
does what I expect — cuts the imbalance to somewhere near half, into the low `0.06`s, while the
cross-entropy stays pinned near the control's — then by the one-to-one exchange rate `r` should climb
by roughly `+0.06` to `+0.07`, landing near `−3.79`, and every bit of that gain should live in the
imbalance term while the cross-entropy term stands still. That decomposition is the real test, sharper
than the headline `r`. If `r` improves but the split shows the cross-entropy has crept upward to
finance it, that is the micro-batch over-constraint tax surfacing already, and it would tell me the
granularity is costing me even at this small scale. If `r` improves with the cross-entropy flat, the
`f·P` surrogate has done its job cleanly and the tax is below the resolution of this run — in which
case the case for changing the scope rests on principle rather than on a visible CE gap, and I should
say so honestly rather than pretend the small model showed me a cost it did not.

So the rung is: add `α · N · Σ_i f_i P_i` to the cross-entropy, with `α = 10^{-2}`, `f_i` the
detached micro-batch count and `P_i` the differentiable mean router probability, averaged over the
layers and the four micro-splits. My falsifiable expectations are two. The imbalance should drop
sharply relative to the control's `0.1286` — this is the established fix and the gradient derivation
above says it must steer toward uniform — so I would expect it roughly halved or better into the low
tens of thousandths; if it barely moved, my gradient analysis would be wrong. And the cross-entropy
should stay close to the control's, held perhaps a hair higher by the over-constraint, though at this
small scale, where four quarters of a batch are still fairly representative of one another, the
specialization cost may be too small to surface as a visible CE penalty. That last uncertainty is
itself the wall this rung walks into: the fix is correct in *form* but applied at the wrong
granularity, and if the only defect is that I am demanding uniformity of every tiny slice rather than
of the corpus as a whole, then the next rung is to change nothing about the penalty — same `f·P`,
same `N`, same `α` — except the set of tokens the counts are measured over.
