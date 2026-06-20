The flat floor told me the only lever is the *shape* of the heights, and that I have to break the
perfect self-alignment at zero shift to get the worst overlap below `1/2`. So now I actually have to
optimize. The first decision is the piece count, and I deliberately keep it small — a couple dozen cells —
because I want to *understand* what an optimized profile looks like before I commit to a long vector. With
two dozen heights I can run many restarts cheaply, watch where the optimizer parks, and read the shape off
the result. The published `~600`-piece records are the destination, but I should not start there; I should
start where the search is fast and the structure is legible.

Now, what am I actually minimizing? The score is `max_k c_k` over integer shifts, rescaled — a *minimax*.
That is the crux and the difficulty. The objective is the maximum of many smooth functions of the heights
(one `c_k` per shift), so it is piecewise-smooth with kinks exactly where two shifts tie for the worst
overlap. A plain gradient method would chatter at those kinks, and a generic minimizer that only sees the
hard `max` gets no useful gradient information about the *other* near-worst shifts that are about to become
binding. I have hit this wall before in spirit: optimizing a `max` directly is brittle. The fix I reach
for is to replace the hard `max` with a smooth surrogate — a log-sum-exp / softmax over the shifts with a
sharpness parameter `β`. For moderate `β` the surrogate is a smooth, differentiable stand-in that *feels*
all the near-worst shifts at once and pushes them all down together; as `β → ∞` it converges to the true
`max`. So I will minimize the soft-max overlap, not the hard one, and anneal `β` upward so that by the end
the surrogate genuinely tracks the constant I am reporting.

The second issue is the constraints, and they are not optional decorations — they are what makes this the
Erdős problem rather than a trivial "set everything to zero" problem. The heights must stay in `[0,1]`
(box constraints) and must sum to exactly `n/2` (a linear equality). Without the sum constraint the
optimizer would just drive every height to `0`, killing the overlap and certifying a meaningless bound;
the `Σ v = n/2` rule is precisely the balance between `A` and `B` that forbids that. So I need a
constrained optimizer that respects a box and a linear equality. The natural choice is **SLSQP** —
sequential least-squares quadratic programming — which handles exactly box bounds plus equality
constraints, and which the agentic-search literature on this very problem (AutoEvolver) reports using.
I will hand it the smooth soft-max objective, the box, and the equality, and let it find a constrained
stationary point.

There is a subtlety I want to get right: SLSQP minimizes the *surrogate*, but I will *report* the true
hard-max overlap of whatever vector it returns. Those two numbers diverge slightly — the surrogate sits a
hair below the true max for finite `β`. So my recipe is a short ladder of SLSQP solves at increasing `β`:
start soft so the optimizer can move the whole profile around without getting stuck on a single binding
shift, then sharpen `β` so the surrogate hugs the true max and the final vector is genuinely good under
the real evaluator. After each solve I re-project onto the constraint set — clip to `[0,1]` and shift the
heights to restore `Σ = n/2` — so that the vector I score is exactly feasible, not approximately.

The third issue is local minima. This minimax landscape is non-convex and riddled with them — the
overlap envelope can be flattened in many qualitatively different ways, and SLSQP from one start finds one
local basin. With only two dozen heights the obvious remedy is *multi-start*: run the annealed-SLSQP
ladder from many random feasible initializations and keep the best true-overlap vector. Each random start
is a different random feasible profile (random heights, projected to sum `n/2`), and the basin SLSQP
settles into depends sharply on where it begins. A dozen starts at this small `n` is cheap and gives me a
reliable read on the best the coarse resolution can reach.

What do I expect? Breaking the flat symmetry should be worth a lot — the literature's very first
optimized step functions already crossed from `0.5` down to `0.4` and then into the `0.38`s. With only
`~24` pieces I will not reach the `0.3809` frontier; that needs hundreds of cells to resolve the fine
structure of a near-optimal profile. But I expect a large drop off the floor — into the low `0.38`s,
somewhere a touch above the `0.380927` Haugland landmark — because even a coarse near-binary profile
captures most of the gross structure that suppresses the worst overlap. The shape I expect to see is the
one the reasoning predicted: heights pushed toward `0` and `1` rather than `1/2`, asymmetric, arranged so
that no single shift accumulates a large product. The honest number I report is the true `max_k`-overlap of
the best returned vector under the frozen evaluator.

The limitation this rung will expose is resolution. Two dozen cells can only approximate the optimal
profile coarsely — the binding shifts are few and the steps are wide, so the worst overlap cannot be
shaved as finely as a long vector allows. The next rung has to lift the optimized coarse profile to many
more pieces and refine it there, where the extra degrees of freedom let the optimizer carve the finer
structure that brings the bound down toward the published step-function frontier.
