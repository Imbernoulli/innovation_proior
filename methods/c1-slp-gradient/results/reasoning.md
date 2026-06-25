The coarse annealing gave me a `50`-piece profile at `1.5371`, and the feedback was clear about why it stops
there: `50` pieces is too coarse a grid to render an autoconvolution whose peak can be pushed lower, and the cap
on how low that peak can go is the cap on `R`. The optimized shape itself looks right — asymmetric, mass heavier
toward the ends, several heights pinned at zero — so I do not want to throw it away and search a long vector from
scratch. I want more resolution, so the autoconvolution can develop the fine structure a coarse vector cannot
represent. And I want a stronger optimizer than the polish I used before — gradient ascent on a softmax surrogate —
because it stalled, and before I commit to a heavier machine I want to understand *why* it stalled, since the
reason should dictate what replaces it.

So let me first pin down the mechanism, because my suspicion is about the shape of the objective, not the
step size. When I push the peak of the autoconvolution down I am not moving a single node; I suspect I am up
against a whole *plateau* of near-equal peak nodes, so that lowering one lets another rise. That is a checkable
claim, so before I trust it I should look at an actual warm-started profile and count. I run the softmax-Adam pass
on a small grid (`N=80`) to get a representative optimized vector and tabulate, on its autoconvolution, how many
nodes sit within a given relative margin of the peak. The warm start lands at `R = 1.5465`, and the count comes
back:

```
within 1e-6 of peak:   1 / 159 nodes
within 1e-3 of peak:   3 / 159 nodes
within 1e-2 of peak:  60 / 159 nodes
```

That settles it. The argmax is unique — only one node is truly at the maximum — but `60` of the `159`
autoconvolution nodes are within one percent of it. The peak is not a spike sitting above slack; it is the top of
a broad, nearly flat ridge. So any move that lowers the single argmax node by a hair immediately promotes one of
those `60` to be the new maximum, and `R` barely changes. That is exactly the failure mode I should expect from a
gradient that only ever sees one node at a time.

And I can confirm the gradient really is stuck, rather than just under-tuned, by taking one step from that warm
point and measuring. The softmax-max surrogate gradient at the warm vector, stepped at a range of learning rates,
gives:

```
warm R = 1.547746
 lr 1e-4 -> R = 1.547729
 lr 1e-3 -> R = 1.547584
 lr 1e-2 -> R = 1.546535
 lr 1e-1 -> R = 1.552178
```

The best of these buys about `0.0012` in `R`, and pushing the step larger (`lr 1e-1`) overshoots and makes `R`
*worse* — it has flattened the argmax but raised a neighbor. So the surrogate is not failing for want of a bigger
step; the descent direction itself is myopic. It attacks one node while the objective is the maximum over the
whole ridge. That is the real wall, and it tells me what the replacement has to do: lower *all* of the near-tight
nodes at once, not one at a time.

That reframing names the problem precisely. The true objective is a *minimax* — minimize the largest of many
smooth functions (the nodes) subject to the mass being fixed. Minimizing a maximum of functions under linear
constraints is a classic problem, and the standard handle on it is the epigraph form solved by *linear
programming*: introduce an auxiliary variable `z` standing for the peak, minimize `z` subject to every
autoconvolution node being `≤ z`, and hold the mass fixed. The obstruction is that each node `b_k = (a*a)_k` is
*quadratic* in the heights, so those constraints are not linear. The natural fix is to linearize them around the
current heights: `b_k(a+d) ≈ b_k(a) + (∇b_k)·d`. For that I need the gradient of a self-convolution node, and I
want to be sure I have it right before I build an LP on top of it, so I check it numerically. The claim is
`∂b_k/∂a_j = 2 a_{k−j}` — a shifted copy of the heights. On a random `5`-vector, finite-differencing node `k=4`
against that formula gives a maximum absolute discrepancy of `1.9e-10`, machine-precision agreement. So the
Jacobian is `J[k,j] = 2 a_{k−j}`, a shifted copy of the heights, and every constraint is linear in the step `d`.

A single LP solve then gives the best peak-lowering move under the linearized model — and because every near-tight
constraint is in the LP simultaneously, the move it returns lowers all of those `60`-odd ridge nodes together,
which is the precise thing the gradient could not do. I keep the move inside a small trust region so the
linearization stays accurate (the neglected quadratic term is second order in `d`), accept the step only if the
*true* `R` drops, grow the trust region on success and shrink it on rejection, and iterate. This is sequential
linear programming, and it matches the recipe the record constructions describe: focus the optimization on the
constraints that are close to tight — the positions where the convolution is largest — and drive them down
together.

There is a real efficiency question, because the autoconvolution at `N = 600` pieces has about `1200` nodes, and
writing a constraint for every one of them every round is wasteful: the vast majority are slack and play no role.
So I focus the LP on the *top-K* largest nodes — the near-tight set, which the count above shows is the only part
that matters. With a small trust region the peak cannot jump to a node far outside this set in one step, so the
restricted LP is a faithful local model, and it is an order of magnitude cheaper to solve. I interleave the
occasional full pass over all nodes as a check, so I never let a node sneak above the peak unseen, but the bulk of
the work is the cheap top-K solves. This focusing is what lets the method grind for thousands of rounds in the time
budget.

I still want a warm start, because sequential LP is a local method and the basin it lands in depends on where it
begins. The cleanest warm start is the softmax-Adam refinement itself: even though it plateaus, it gets me from the
flat ceiling down to around `1.52` cheaply and lands in a sensible asymmetric basin, and from there the SLP takes
over. (The ceiling is worth a sanity number: the flat indicator `a = 1` has autoconvolution peak `N` and mass `N`,
so `R = 2N·N/N² = 2.0` exactly — I checked it at `N = 1,5,50,600` and got `2.000000` each time, which is the right
top of the range.) So the rung is a two-stage pipeline: a short `β`-annealed Adam pass to break symmetry and reach
`~1.52`, then a long run of trust-region top-K sequential LP to grind the peak down, checkpointing the best vector
throughout so nothing is lost.

Before scaling to `N = 600` I run the whole pipeline once on the small `N=80` grid to see whether the SLP actually
moves where the gradient stalled. Starting from the warm vector at `1.5465`, blocks of `25` LP rounds give:

```
warm:           R = 1.54775
after block 0:  R = 1.54204   (25/25 steps accepted)
after block 1:  R = 1.54094   (25/25)
after block 2:  R = 1.54010   (25/25)
after block 3:  R = 1.53939   (17/25)
after block 4:  R = 1.53936   ( 3/25)
after block 5:  R = 1.53931   (11/25)
```

So the SLP does descend — from `1.5477` to `1.5393`, a real `0.0084` drop, far more than the gradient's `0.0012` —
and the accept-rate signature is exactly what I would expect: nearly every step is accepted early when the
linearized model has slack to exploit, then acceptance falls off (`17/25`, then `3/25`) as the ridge flattens and
the quadratic error starts rejecting steps and shrinking the trust region. The returned `N=80` profile is sparse
(`17.5%` of heights near zero) and its tallest weight sits at the very last index (`79/80`), the asymmetric
boundary-spike shape I was hoping the higher resolution would sharpen. This is a real confirmation of the
mechanism, and it is also a measured limit: at `N=80` the descent tapers around `1.539`, well above the AlphaEvolve
band, because the grid is still too coarse — which is precisely the resolution argument that motivates lifting `N`.

So at `N = 600` I expect the same two-stage behavior with more headroom. The Adam warm start should again land near
`1.52`; the SLP is the engine, and I expect each LP solve to lower the whole near-tight plateau a little, pushing
from `1.52` down into the `1.51` range and, with enough rounds, toward the `1.5053` band of the `600`-piece
AlphaEvolve construction — the same resolution AlphaEvolve used to reach `1.5053`, which is why `N = 600` with a
long SLP run is the right target for this rung. I do not expect to reach the `1.50286` record here: that record is
a `30000`-piece construction found by an agentic search spending tens of hours, and the gain from `1.5053` to
`1.50286` lives in the fourth decimal and demands both far finer resolution and far more compute than a single
`600`-piece run. I expect this rung to taper inside the AlphaEvolve `600`-piece band, the way the `N=80` run
tapered inside its own resolution limit — and the taper is the opening for the endpoint: if more pieces and a
longer SLP grind keep paying, the move is to lift to several thousand pieces and spend a long SLP run there,
pushing toward the record.
