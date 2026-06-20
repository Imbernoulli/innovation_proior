The control did exactly what I feared: a tolerable cross-entropy sitting on top of a clearly skewed
token allocation, the imbalance well above zero. The disease is real, and the diagnosis from the
last rung was sharp — the cross-entropy is structurally blind to balance, so the cure cannot come
from the cross-entropy. I have to add a term, by hand, that looks at the routing distribution itself
and pushes it toward uniform. The question is what that term should be, and it is not obvious,
because the most natural thing to penalize is something I cannot differentiate.

Let me see the obstacle clearly first. What I actually want to control is the fraction of tokens
each expert receives — call it `f_i` for expert `i`. If I could just penalize how far the vector of
`f_i` sits from uniform, I would be done. But `f_i` is a count: it comes from the hard top-K
decision, from `argmax`-like selection of which experts each token goes to. Counts have zero
gradient almost everywhere — nudging the router's weights a hair does not change which expert is
top-two until some token crosses a boundary, and then the count jumps discontinuously. So a penalty
written purely on `f_i` gives the router no gradient to follow. I need a penalty that *correlates*
with the imbalance but flows through something differentiable, and the only differentiable thing the
router exposes is its softmax probability vector — the continuous mass `P_i` it puts on each expert
before the hard selection.

So the trick has to be to pair the non-differentiable count with the differentiable probability. Here
is the construction I keep coming back to. For each expert, form the product `f_i · P_i` — the
fraction of tokens it actually got, times the average probability mass the router assigned it — and
sum over experts. Treat the `f_i` as a fixed weight (detach it; its gradient is useless anyway) and
let the gradient flow only through `P_i`. Now ask what minimizing `Σ f_i P_i` does. The `f_i` are a
fixed set of weights that are large for the over-used experts and small for the under-used ones. To
shrink a weighted sum of the `P_i` under those weights, the router should move probability mass
*off* the experts with large weight — exactly the over-used ones — and onto the experts with small
weight. That is precisely the corrective pressure I want: the experts that are hot get their
probability pulled down, the cold ones get it pushed up, and the only experts safe from the penalty
are the ones already at their fair share. The non-differentiable count steers *where* the pressure
points; the differentiable probability is *how* the pressure is applied.

I want to sanity-check that the uniform routing is actually the optimum of this surrogate, because a
penalty whose minimum is somewhere other than uniform would be worse than useless. If every expert
gets the uniform share, then every `f_i = 1/N` and, at the balanced fixed point, every `P_i ≈ 1/N`,
so `Σ f_i P_i ≈ N · (1/N)(1/N) = 1/N`. To make this clean and scale-free I should multiply the whole
thing by `N`, so the balanced value is a constant near one regardless of how many experts I have;
then a single coefficient `α` in front sets how hard the penalty pushes relative to the
cross-entropy. The literature value for that coefficient is around a hundredth — strong enough to
break the collapse, weak enough that it does not start dictating the router's predictions and drag
the cross-entropy up. I will take `α = 1e-2`.

There is one more decision, and it is the one I expect to come back and bite me, so let me be
explicit about it now rather than discover it later. Over what set of tokens do I compute `f_i`? The
obvious answer, and the one the original Switch and GShard losses take, is: over the micro-batch —
the handful of tokens processed together in one forward pass on one device. It is the cheapest and
most local choice, and it is what the classical loss does. But I can already feel the problem with
it. A micro-batch is a small, noisy sample of the corpus, and it may be genuinely lopsided in
content — a slice that happens to be all code, or all one topic — for which a *skewed* expert usage
is the correct, specialized behavior. Forcing the `f_i` of that slice toward uniform punishes the
router for doing the right thing on that slice. The micro-batch penalty cannot tell the difference
between collapse, which I want to stop, and legitimate per-slice specialization, which I want to
keep. So I suspect this loss will balance the load — it will clearly beat the control on imbalance —
but it will do so partly by flattening specialization, and that flattening will show up as
cross-entropy that is not as low as it could be.

So the rung is: add `α · N · Σ_i f_i P_i` to the cross-entropy, with `f_i` the detached micro-batch
count and `P_i` the differentiable mean router probability, averaged over the layers; here I emulate
the micro-batch by splitting each training batch into a few micro-splits, computing the penalty on
each, and averaging. I expect a large drop in imbalance relative to the control — this is the
established fix and it works — at the price of a cross-entropy held slightly higher than it needs to
be, because the per-micro-batch uniformity pressure is over-constraining. That over-constraint is
the wall: the fix is correct in form but applied at the wrong granularity. If the only problem is
that I am demanding uniformity of every tiny slice rather than of the corpus as a whole, then the
next rung is to change nothing about the penalty except the set of tokens the counts are measured
over.
