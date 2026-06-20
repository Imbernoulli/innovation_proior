The coarse annealing gave me a `50`-piece profile at `1.5371`, and the feedback was clear about why it stops
there: `50` pieces is too coarse a grid to render an autoconvolution whose peak can be pushed lower, and the cap
on how low that peak can go is the cap on `R`. The optimized shape itself looks right — asymmetric, mass heavier
toward the ends, several heights pinned at zero — so I do not want to throw it away and search a long vector from
scratch. I want more resolution, so the autoconvolution can develop the fine structure a coarse vector cannot
represent. And I want a fundamentally stronger optimizer, because the polish I used before — gradient ascent on a
softmax surrogate — has a specific weakness on this problem that I now have to confront head-on.

Let me name that weakness, because it dictates the whole design of this rung. When I push the peak of the
autoconvolution down, I do not move a single node; I flatten a whole *plateau* of near-equal peak nodes. At the
coarse `50`-piece optimum the autoconvolution already has many nodes within a hair of the maximum. A gradient on
the softmax surrogate, or a subgradient at the single argmax, can lower *one* peak node, but the moment it does,
another node in the plateau becomes the new maximum and `R` does not actually drop — the true objective is the
maximum over the *whole* set of near-tight nodes, and lowering one at a time is like pressing down one spot of a
balloon. I watched exactly this happen when I first tried to refine on a finer grid: the surrogate gradient
plateaued around `1.52` and would not go lower, because it was never attacking the right object. The right object
is the *minimax*: minimize the largest of all the near-peak nodes simultaneously, subject to the mass being fixed.

That reframing tells me what to do. Minimizing a maximum of smooth functions subject to linear constraints is a
classic problem, and the standard tool is the epigraph form solved by *linear programming*. I introduce an
auxiliary variable `z` standing for the peak, ask to minimize `z` subject to every autoconvolution node being `≤
z`, and hold the mass fixed. The catch is that each node `b_k = (a*a)_k` is *quadratic* in the heights, so the
constraints are not linear. I linearize them: around the current heights, `b_k(a+d) ≈ b_k(a) + (∇b_k)·d`, where
the gradient of a self-convolution node is itself just a shifted copy of the heights (`∂b_k/∂a_j = 2 a_{k−j}`).
That makes every constraint linear in the step `d`, so a single LP solve gives the best peak-lowering move under
the linearized model — and crucially it lowers *all* the near-tight nodes at once, which is exactly what the
gradient could not do. I keep the move inside a small trust region so the linearization stays accurate (the
neglected quadratic term is second order in `d`), accept the step only if the *true* `R` drops, grow the trust
region on success and shrink it on rejection, and iterate. This is sequential linear programming, and it is the
same idea the published record constructions describe: focus the optimization on the constraints that are close to
tight — the positions where the convolution is largest — and drive them down together.

There is a real efficiency question here, because the autoconvolution at `N = 600` pieces has about `1200` nodes,
and writing a constraint for every one of them every round is wasteful: the vast majority are slack and play no
role. So I focus the LP on the *top-K* largest nodes — the near-tight set — which is the active part of the
minimax. With a small trust region the peak cannot jump to a node far outside this set in one step, so the
restricted LP is a faithful local model, and it is an order of magnitude cheaper to solve. I do interleave the
occasional full pass over all nodes as a check, so I never let a node sneak above the peak unseen, but the bulk of
the work is the cheap top-K solves. This top-K focusing is precisely what lets the method grind for thousands of
rounds in the time budget.

I still want a warm start, because sequential LP is a local method and the basin it lands in depends on where it
begins. The cleanest warm start is the softmax-Adam refinement itself: even though it plateaus, it gets me from
the flat ceiling down to around `1.52` cheaply and lands in a sensible asymmetric basin, and from there the SLP
takes over and keeps descending where the gradient stalled. So the rung is a two-stage pipeline: a short
`β`-annealed Adam pass to break symmetry and reach `~1.52`, then a long run of trust-region top-K sequential LP
to grind the peak down, checkpointing the best vector throughout so nothing is lost.

What do I expect? The Adam warm start should land near `1.52`, as before. The sequential LP is the new engine,
and I expect it to keep finding small, real improvements round after round — each LP solve lowers the whole
near-tight plateau a little — pushing from `1.52` down into the `1.51` range and, with enough rounds, toward the
`1.5053` band of the `600`-piece AlphaEvolve construction. That is exactly the resolution and the method that
AlphaEvolve used to reach `1.5053`, so `N = 600` with a long SLP run is the right target for this rung. I am
honest that I will *not* reach the `1.50286` record here: that record is a `30000`-piece construction found by an
agentic search spending tens of hours, and the gain from `1.5053` to `1.50286` lives in the fourth decimal place
and demands both far finer resolution and far more compute than a single `600`-piece run. So I expect this rung to
prove the SLP principle — that attacking the whole near-tight plateau with a trust-region LP buys the descent the
gradient could not, into the AlphaEvolve `600`-piece band — and then to taper, limited by resolution and by how
many LP rounds I can afford. The taper is the opening for the endpoint: if more pieces and a longer SLP grind keep
paying, the move is to lift to several thousand pieces and spend a long SLP run there, pushing toward the record.
