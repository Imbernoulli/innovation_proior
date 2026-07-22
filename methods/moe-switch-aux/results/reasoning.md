The control did exactly what I feared: a tolerable cross-entropy sitting on top of a clearly
skewed token allocation, the imbalance well above zero. The disease is real, and the
diagnosis from the control was sharp — the cross-entropy is structurally blind to balance,
so the cure cannot come from the cross-entropy. I have to add a term, by hand, that looks at
the routing distribution itself and pushes it toward uniform. The question is what that term
should be, and it is not obvious, because the most natural thing to penalize is something I
cannot differentiate.

What I actually want to control is the fraction of tokens each expert receives — call it
`f_i` for expert `i`. If I could just penalize how far the vector of `f_i` sits from
uniform, I would be done. But `f_i` is a count: it comes from the hard top-K decision, from
`argmax`-like selection of which experts each token goes to. Counts have zero gradient
almost everywhere — nudging the router's weights a hair does not change which expert is
top-two until some token crosses a boundary, and then the count jumps discontinuously. So a
penalty written purely on `f_i` gives the router no gradient to follow. I need a penalty
that *correlates* with the imbalance but flows through something differentiable. The only
quantity the router exposes that is smooth in its weights is the softmax probability vector
— the continuous mass `P_i` it puts on each expert before the hard selection. So whatever I
write, the gradient has to enter through `P`, and `f` can at most enter as a fixed weight
that I detach.

Here is the simplest thing that uses both. For each expert form the product `f_i · P_i` —
the fraction of tokens it actually got, times the average probability mass the router
assigned it — and sum over experts: `S = Σ_i f_i P_i`, with `f_i` detached. The story I tell
myself is that the `f_i` act as weights that are large on the over-used experts and small on
the under-used ones, so shrinking this weighted sum of probabilities should pull mass off
the hot experts. That story is plausible, but a penalty's gradient can easily point
somewhere other than the cartoon, so I differentiate `S` directly and check where it points.

With `f` detached, `S` is a weighted sum of softmax outputs, and `dP_i/dz_j = P_i(δ_ij −
P_j)` for router logits `z`. So

`dS/dz_j = Σ_i f_i P_i(δ_ij − P_j) = P_j f_j − P_j Σ_i f_i P_i = P_j (f_j − f̄)`, where `f̄
= Σ_i f_i P_i`.

That is a clean form, and it is worth reading carefully. The sign of the gradient on logit
`j` is the sign of `f_j − f̄`: positive when expert `j` carries more than the
probability-weighted average count, negative when it carries less. Gradient descent
subtracts this, so it pushes the logit *down* on any expert whose count is above average and
*up* on any expert whose count is below average. Putting numbers on it: take `N=4` and a
skewed count `f = (0.5, 0.3, 0.15, 0.05)` with the router currently uniform, `P =
(¼,¼,¼,¼)`. Then `f̄ = Σ f_i/4 = 0.25` and the gradient is `P_j(f_j − f̄) = ¼·(f_j − 0.25) =
(0.0625, 0.0125, −0.025, −0.05)`. So the hot expert (count 0.5) gets a positive gradient —
its logit is driven down — and the cold expert (count 0.05) gets a negative gradient — its
logit is driven up. I take one gradient-descent step from uniform with this `f` and the
routing moves to roughly `(0.19, 0.23, 0.27, 0.30)`: mass has flowed off the hot expert and
onto the cold one, exactly as the gradient's sign said it would.

But that very calculation makes me nervous about a claim I was about to wave through: that
uniform routing is the *optimum* of this surrogate. If it were not, I would be installing a
penalty whose minimizer is somewhere I do not want, so I check directly — the naive version
of the check is misleading. The naive check is: plug in the balanced state, every `f_i =
1/N` and every `P_i = 1/N`, and read off `S = Σ (1/N)(1/N) = 1/N`. That tells me the *value*
at uniform; it tells me nothing about whether uniform is a minimum. And here is the trap:
with `f` *detached and held fixed*, `S` is **linear** in `P`. A linear function on the
probability simplex is minimized at a vertex, not in the interior. With the skewed `f` above
frozen in place, the minimizer of `S` is the vertex that dumps all probability on the single
smallest-`f` expert — I take many gradient steps with `f` frozen and the routing runs to
`(0.002, 0.003, 0.007, 0.988)`, all mass on expert 3, the coldest one. That is the
*opposite* of balanced. So "uniform is the optimum of `S`" is simply false if I read `S` as
a static objective with frozen counts.

What rescues the construction is that `f` is not frozen. During training, `f` is recomputed
every step from the routing the network currently produces — it is the actual count of where
tokens went under the current `P`. So the penalty is not a fixed potential I am descending;
it is a *coupled* pressure where the weights chase the routing. The right question is
therefore not "what is the minimizer of `S`" but "where does the coupled system have no
force left" — its fixed point. The force on logit `j` vanishes when `f_j = f̄` for every
`j`, i.e. when all the counts are equal, i.e. at `f_i = 1/N`. I check the balanced point
directly: set `P` uniform and `f` uniform, and the gradient `P_j(f_j − f̄)` is `(0,0,0,0)`
exactly — uniform is a genuine stationary point. And I check that a self-consistent *skewed*
state is not one: take `P = f = (0.5,0.3,0.15,0.05)` (counts matching the skewed routing)
and the gradient is `(0.0675, −0.0195, −0.0323, −0.0158)` — nonzero, positive on the hot
expert and negative on the cold ones, a restoring force pointing back toward uniform. So
uniform is the fixed point and skew is not; the coupling, not the static shape of `S`, is
what makes uniform the target. That is the real reason the surrogate works, and it is more
subtle than the value-at-uniform computation I almost stopped at.

Now the scaling and the weight. The balanced value of `S` is `1/N`, which drifts with the
number of experts; if I want one coefficient that behaves the same across model sizes I
should multiply by `N` so the balanced value is a constant near one. Then a single
coefficient `α` in front sets how hard the penalty pushes relative to the cross-entropy. I
want `α` large enough to overcome the collapse the control showed but small enough that the
penalty does not start dictating the router's predictions and drag the cross-entropy up; the
established setting for exactly this loss is around a hundredth, and I have no reason here
to deviate, so `α = 1e-2`.

One more decision remains: over what set of tokens do I compute `f_i`? The obvious answer,
and the one the original Switch and GShard losses take, is: over the micro-batch — the
handful of tokens processed together in one forward pass on one device. It is the cheapest
and most local choice. But I can already feel the problem with it, and the linearity I found
above sharpens it. A micro-batch is a small, noisy sample of the corpus, and it may be
genuinely lopsided in content — a slice that happens to be all code, or all one topic — for
which a *skewed* expert usage is the correct, specialized behavior. On that slice the fixed
point of the coupled pressure is still flat usage, because the force only vanishes at `f_i =
1/N` regardless of what the slice is about. So the micro-batch penalty cannot tell the
difference between collapse, which I want to stop, and legitimate per-slice specialization,
which I want to keep — it pushes both toward uniform with the same restoring force I just
computed. So I expect this loss will balance the load — it will clearly beat the control on
imbalance, because the restoring force is real and points the right way — but it will do so
partly by flattening specialization, and that flattening should show up as cross-entropy
that is not as low as it could be.

So the fix is: add `α · N · Σ_i f_i P_i` to the cross-entropy, with `f_i` the detached
micro-batch count and `P_i` the differentiable mean router probability, averaged over the
two MoE layers; here I emulate the micro-batch by splitting each training batch into 4
micro-splits, computing the penalty on each, and averaging. The open question is only how
much of the imbalance gain the flattening tax above eats out of the cross-entropy. That
over-constraint is the wall: the fix is correct in form but applied at the wrong
granularity. If the only problem is that I am demanding uniformity of every tiny slice
rather than of the corpus as a whole, then the natural next step is to change nothing about
the penalty except the set of tokens the counts are measured over.