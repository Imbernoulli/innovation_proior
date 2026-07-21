The control did exactly what I feared, and the two numbers are worth reading together before I do
anything. Imbalance came in at `L_imb = 0.1286` on a perplexity of `41.594`, a cross-entropy of
`3.7280` — and `exp(3.7280) = 41.60`, so the two readings are consistent and I am reading a genuinely
trained model, not an artifact. Placing `0.1286` on the scale I worked out: the reachable ceiling under
top-two routing is `0.75`, so the control sits about a sixth of the way to full collapse, and against
my calibration anchor of `0.25` for a conspicuous two-fold overload it is roughly half of that — a
mild-to-moderate skew, and comfortably above any plausible sampling floor. All of it sitting on a
perfectly ordinary cross-entropy, a branching factor in the low forties. So the control confirmed the
diagnosis in exactly the joint shape I predicted: a fine cross-entropy over a clearly skewed histogram.
The disease is real, the cross-entropy is structurally blind to it, and the cure has to be a term I add
by hand that looks at the routing distribution and pushes it toward uniform. The question is what that
term should be, and it is not obvious, because the most natural thing to penalize is exactly the thing
I cannot differentiate.

The obstacle eliminates most candidates before I list them. What I want to control is the fraction of
tokens each expert receives, `f_i`; if I could penalize how far the vector of `f_i` sits from uniform I
would be done in one line. But `f_i` is a hard count from the top-K selection — piecewise-constant with
jumps, gradient exactly zero until one token's ranking flips and a count jumps by one. So the entire
family of "just penalize the counts" losses is dead on arrival. Whatever I write has to *correlate* with
the count imbalance but carry its gradient through something differentiable, and the only differentiable
thing the router exposes is its softmax probability vector `P` — the continuous mass `P_i` it puts on
each expert before the hard selection collapses it into a choice.

That leaves a small design space. One option is to forget the counts and penalize only the
probabilities, `Σ_i (P_i − 1/N)^2` — differentiable, minimized at uniform, but it flattens the router's
probability *per token*, which is precisely the sharp content-dependent routing I must protect. It
manufactures balance by destroying specialization. A second option is the importance penalty `Σ_i P_i^2`
alone: smooth and minimized at uniform, but ungrounded in what the router actually *did*. A router whose
`P` is fairly flat but whose hard top-two still lands almost every token on the same pair — because two
experts sit a persistent hair above the rest — satisfies `Σ_i P_i^2` near its floor while the realized
counts `f` are badly skewed. It penalizes intended imbalance, not realized imbalance. What I want is a
term anchored to the real usage `f`, so it fires when experts are actually overloaded, yet flowing its
gradient through `P` so it is trainable. The construction that does both is the product summed over
experts: `Σ_i f_i · P_i`, the fraction actually received times the average probability assigned, with
`f_i` treated as fixed weights — detached, since their own gradient is useless — and the gradient
flowing only through `P_i`. The counts steer *where* the pressure points; the probability is *how* it
is applied. Multiplying by `f` closes exactly the importance-only loophole: `f_i P_i` stays large until
the experts the router overloads are the same ones it is pulling probability off.

Let me differentiate it, since a surrogate pushing the wrong way would be worse than none. With
`P = softmax(z)` the Jacobian is `∂P_i/∂z_j = P_i(δ_ij − P_j)`, so for the single-layer term
`T = N Σ_i f_i P_i` with `f` detached, `∂T/∂z_j = N P_j(f_j − m)`, where `m = Σ_i f_i P_i` is the
`P`-weighted average of the counts. Descending it moves the logit `z_j` *down* when `f_j > m` — the
hot experts — and *up* when `f_j < m` — the cold ones — leaving alone only those near the mean. That is
exactly the corrective pressure I wanted, and the sign of `f_j − m` guarantees it without my having to
assume it.

But the same formula carries a warning that will matter two rungs from now: the push on expert `j`
scales with `P_j`. An expert the router has already driven to `P_j ≈ 0` gets a gradient of nearly zero
*even though* it is the one most in need of rescue, because the `P_j` prefactor and its being cold both
shrink the term. Put numbers through it. Take `f = P = (0.30, 0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02)`;
then `m = Σ P_i^2 = 0.197`, and the per-expert push `N P_j(f_j − m)` suppresses the two hot experts
(`+0.247`, `+0.106`) and lifts the rest. The middling-cold expert at `f = 0.06` receives a lift of
`0.066`; the coldest at `f = 0.02`, nearest death, receives only `0.028` — a factor of `2.3` *weaker*
pull on the expert that needs it most, because it has the largest deficit but the smallest probability
to multiply it by. So the surrogate steers toward uniform, but its grip weakens exactly in the tail
where collapse is worst. I will not fix that here — this rung establishes that `f·P` works at all — but
it is a structural property of the penalty, not a tuning accident.

There is a subtlety in detaching `f` worth settling, because within a single step, with `f` held fixed,
the gradient of `Σ_i f_i P_i` in `P` is minimized not by going uniform but by dumping all probability
onto whichever expert currently has the smallest `f_i` — the frozen objective points at the coldest
corner, not the center. If `f` never updated, the router would overshoot. But `f` is recomputed every
step, so the real process is coordinate descent: measure the counts, nudge `P` toward the currently-cold
experts, let realized counts shift, measure again. Its only rest point is where `N P_j(f_j − m) = 0` for
every `j`, i.e. `f_j = m`, uniform *usage*. So detaching is not an approximation I tolerate — it is the
right thing, since `f`'s own gradient is the useless zero-or-jump of the count and letting it flow would
inject only boundary-crossing noise.

Two scaling choices make the penalty well-behaved. First, is uniform really the optimum? Realized usage
tracks probability, so to first order `f_i ≈ P_i` and `Σ_i f_i P_i ≈ Σ_i P_i^2 ≥ (Σ_i P_i)^2/N = 1/N`,
with equality iff `P` is uniform — so the penalty is minimized at uniform and nowhere else. Second, that
minimum value `1/N` drifts with the number of experts, so I multiply by `N` to make the balanced optimum
scale-free at `1`. The range is then instructive: `N Σ P_i^2 = 1` at uniform, `8·0.5 = 4` at a two-expert
collapse — a factor of four of headroom for the gradient to work with. That headroom times the
coefficient `α` sets the budget. With `α = 10^{-2}` the penalty contributes between `0.01` (at balance)
and `0.04` (at collapse) against a cross-entropy of order `3.7` — at most a roughly one-percent addition,
strong enough to register a real gradient against the router but far too weak to override the
cross-entropy and distort predictions. An order of magnitude higher would dictate the router's choices
and drag CE up; an order lower would be lost in the LM gradient. The penalty is averaged over the two
MoE layers, and that is not just normalization: each layer has its own router and its own histogram, and
the first might collapse onto experts the second leaves cold. Computing `f·P` per layer keeps each layer
answerable for its own balance; pooling counts across layers could let a well-spread layer statistically
mask a collapsed one.

There is one more decision, and it is the one I expect to come back and bite, so let me name it now:
over what set of tokens do I compute `f_i`? The classical Switch/GShard choice is the micro-batch — the
handful of tokens in one forward pass on one device. It is the cheapest, most local choice, but a
micro-batch is a small, noisy sample and may be genuinely lopsided in content — a slice that happens to
be all one latent topic, for which a *skewed* expert usage is the correct, specialized behavior. Forcing
that slice's `f_i` toward uniform punishes the router for doing the right thing. The micro-batch penalty
cannot tell collapse, which I want to stop, from legitimate per-slice specialization, which I want to
keep; it enforces the right constraint at the wrong scope. In this single-process reproduction I emulate
the micro-batch by splitting each training batch into four micro-splits, penalizing each and averaging —
which is quantitatively stricter than one global constraint, since I now demand uniformity of four
separate quarters, and those demands can conflict when the quarters differ in content.

That over-constraint turns into a cross-entropy cost as a competition between two gradients on the same
slice. On a split dominated by one latent topic, the cross-entropy gradient wants the router to send
those tokens sharply to the two topic-experts — driving `f` on the split *away* from uniform — while the
balancing gradient reads those experts as hot and pushes probability off them, back toward uniform. The
two partially cancel, and the router settles at a compromise less specialized than cross-entropy alone
would make it. That residual mis-specialization is a small cross-entropy tax on every internally-lopsided
slice. So I suspect this loss will balance the load but partly by flattening specialization, and that
flattening should show up as a cross-entropy not quite as low as it could be. Whether the tax is large
enough to *see* at this scale is genuinely uncertain — the slices here are quarters of a small batch and
may be too alike in content for the effect to register — but the sign is not in doubt, which is exactly
what makes granularity the one thing worth changing next. And there is a scope mismatch that colors how I
read the eval: the penalty trains on micro-batch counts, but the `L_imb` I am scored on is measured at a
larger, global scope. A router driven uniform on every quarter is automatically uniform on the whole, but
not conversely, so the micro-batch penalty is strictly *stronger* than the whole-batch uniformity the
metric rewards. That flatters the `L_imb` number while being the exact source of the specialization tax —
the extra strictness buys nothing the metric wants and costs the per-slice sharpness I need.

So the rung is: add `α · N · Σ_i f_i P_i` with `α = 10^{-2}`, `f_i` the detached micro-batch count and
`P_i` the differentiable mean router probability, averaged over the layers and four micro-splits. My
falsifiable expectations are two. The imbalance should drop sharply from the control's `0.1286` — the
gradient derivation says it must steer toward uniform — so I expect it roughly halved or better; if it
barely moved, my gradient analysis would be wrong. And the cross-entropy should stay close to the
control's, held perhaps a hair higher by the over-constraint, though at this small scale the
specialization cost may be too small to surface. If the imbalance roughly halves at flat cross-entropy,
`r` climbs by that much, near `−3.79`, and the honest test is that essentially all of the gain lives in
`L_imb` while `L_CE` stands still; if the cross-entropy has crept upward to finance the gain, that is the
micro-batch over-constraint tax surfacing already. Either way, if the only defect is that I am demanding
uniformity of every tiny slice rather than of the corpus, the next rung changes nothing about the penalty
— same `f·P`, same `N`, same `α` — except the set of tokens the counts are measured over.
