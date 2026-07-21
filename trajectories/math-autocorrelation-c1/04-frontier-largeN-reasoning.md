The sequential-LP refinement did what the minimax diagnosis promised. The `β`-annealed Adam warm start reached
`1.526568` at `N = 600` — the plateau where the single-node gradient hits the balloon wall — and the
trust-region SLP ground past it to `1.517237`, buying `0.0093` the gradient could not by pressing the whole
near-tight cluster down together. But the feedback is equally clear about where it stops: `1.517237` sits
`~0.012` above the `600`-piece AlphaEvolve construction at `1.5053`, and after the descent the run only crawled.
The single most important thing in that table is that AlphaEvolve reached `1.5053` at the *very same* `N = 600`.
The gap is therefore not resolution — `600` pieces demonstrably can express a shape that scores `1.5053`, my
constructor just did not find it. It is the honest signature of a single local constructor converging into *a*
good basin, not *the* good basin. So the endpoint question is sharp: with the same SLP engine but more search
breadth and, where it helps, a finer grid, how far down can I genuinely push?

The shape of the diminishing returns is itself a reading. The ladder went `2.000 → 1.537 → 1.527 → 1.517`, each
step roughly a tenth the size of the last, the curve flattening hard as it approaches the low `1.51`s. That is
the profile of a search running out of easy structure, and it sets an honest expectation before I run: not
another `0.01` but at most a few thousandths, because I am adding breadth and polish to an engine that has
already found most of what it can at this resolution. If this rung bought another `0.01`, that would tell me the
rung-3 basin was surprisingly far from the best `600`-piece basin; reading the size of the gain is how I will
know whether "one warm start, one basin" was the whole story.

Two levers are available. The first is basin diversity. My rung-3 run was one warm start into one basin; an
agentic search of the kind that found `1.5053` effectively tries many structurally different shapes and keeps
the best. I can imitate that cheaply: launch the SLP from several structurally different initializations and
keep the global best. Which shapes to seed, the previous rung answers — the returned profile was asymmetric with
a tall boundary spike (`~9×` the mean at index `0`), mass toward the ends, the middle third thinned to `~0.70×`
the mean, and roughly `215` of `600` heights near zero. It had already drifted into the boundary-spike family,
which is where the good basins live: concentrating mass into a tall spike at a boundary over a thinned interior
is what suppresses the central self-overlap the flat function maximizes. So I seed from a spread of explicit
boundary-spike profiles, varying which end carries the heavier spike since the optimum is asymmetric and I do
not know a priori which way it leans. Concretely each init starts from a flat vector, ramps an added spike of
height `s_L` linearly into the first `n/12` pieces and `s_R` into the last `n/12`, adds the full spike at the
two endpoints, jitters with small Gaussian noise, and normalizes. I sweep the pairs `(s_L, s_R) ∈
{(10,4),(4,10),(8,8),(14,6),(6,14)}` — left-heavy, right-heavy, balanced, and two more lopsided. The `n/12` ramp
and noise give the SLP a smooth feasible shape to linearize from; a bare delta at one node would hand the LP a
degenerate near-singular Jacobian.

The seed sweep has a built-in consistency check, because reversing a height vector reverses its autoconvolution
and leaves the peak and mass untouched, so `R(a) = R(a[::-1])` exactly (`[1,2]` and `[2,1]` both score `16/9`).
The optimum comes in a mirror pair, and the seed pairs `(10,4)/(4,10)` and `(14,6)/(6,14)` are mirror images:
they should descend to *equal* scores. If they do, orientation genuinely does not matter; if two mirror seeds
reached materially different scores, that would flag a bug or an under-converged run. The one non-mirror seed,
the balanced `(8,8)`, tests the competing hypothesis that a symmetric two-spike profile could beat the lopsided
single-dominant-spike shape — a genuine question about the optimum's structure the multi-start answers
empirically.

The second lever is resolution, and here I distrust the obvious move. Repeat-lifting a shape to a finer grid is
*free* in principle: replacing each height by `m` copies produces the same step function, so the same
autoconvolution and the same `R`. I verify rather than assume — `a = [1,2]` scores `16/9`, and `[1,1,2,2]`,
`[1,1,1,2,2,2]`, `[1,1,1,1,1,2,2,2,2,2]` all score `1.7778` — so a repeat-lift costs nothing and hands the SLP
more coordinates to coordinate. The catch is entirely the cost of *using* them. The LP scales with pieces —
`n+1` variables and up to `2n−1` constraints — so doubling `n` roughly quadruples the per-round cost, and the
interior-point factorization grows faster still, making a realistic multiplier at `N = 1200` more like five- or
sixfold. Worse, twice as many coordinates is more structure to coordinate, so the finer grid wants *more* rounds
to converge. Putting those together, a single round at `N = 1200` is seconds rather than the tens of
milliseconds it is at `600`, and for the same budget I can afford only about a sixth as many rounds while
needing more of them.

I resolve the trade by trying the lift and reporting what it does. A repeat-lift to `N = 1200` followed by SLP
polishing does not pay within budget: the per-round cost rises with the grid and the finer grid does not recover
below the lift value before the rounds run out. That is not "resolution is useless" — the record uses `50×` more
pieces — but "at *my* budget the extra coordinates cannot be exploited fast enough to beat the coarser grid
where the SLP iterates freely." So the returned frontier stays at `N = 600`, and I spend the budget on breadth
of starts plus a long polish. For the same reason I do not abandon the SLP for a genetic or population-based
search over the `600`-dimensional vector: the SLP's strength is local precision — driving a whole plateau down
together to resolve the fourth decimal, exactly the regime near `1.517` — while a GA would need an enormous
number of evaluations to match that precision and within budget would land well above where the SLP already
sits. The right use of a broad search is to *seed* the SLP, which is what the multi-start does.

Reading where `1.517` sits says how little room is left. At `N = 600` the prefactor is negligible, so `ratio ≈
peak/mean`: the rung-3 profile has its tallest node at `~1.516×` its average, against the record's `~1.5016×`.
The residual `~0.014` is a demand to flatten the top by a further one part in a hundred, and the profile is
already remarkably flat on top — a few hundred of its `1199` nodes packed within `10^{−3}` of the peak, roughly
a third of all nodes at the plateau. That is close to the expressivity limit of `600` pieces: I cannot flatten
more nodes than exist. It is another way of seeing why the last fraction of a percent will not come from this
grid.

The SLP mechanics are unchanged from the previous rung — epigraph `z`, node constraints linearized as
`b_k(a+d) ≈ b_k(a) + Σ_j 2 a_{k−j} d_j`, mass fixed, trust region, accept only if true `R` drops — and only the
schedule changes. Most of the budget goes to the multi-start at `N = 600`: `six` starts of about `200` SLP
rounds each plus a final polish of about `250`, roughly `1450` LP solves. I run these as full-constraint passes
rather than top-K now, because near the floor I cannot afford to miss a node quietly rising into the peak, and a
full-constraint `linprog` over all `1199` constraints is on the order of a second per solve; `1450` solves is
about `20` minutes, the budget I have — which is exactly why the finer grid is off the table, since the same
time at `N = 1200` with per-solve cost up fivefold buys too few rounds to run six diverse starts to convergence.
After the search identifies the best basin, I give it a long dedicated polish at a tightened trust (`~6·10^{−5}`,
grown more slowly at `×1.04`), checkpointing the global best continuously. The tighter trust follows the same
error arithmetic: near the floor the improving steps are tiny and the dropped quadratic term `~n·tr^2` has to
stay well under the sub-thousandth improvements I am chasing.

So I expect `pieces_N = 600` and a `ratio` a hair under `1.517`, edging toward AlphaEvolve's `1.5053`, on a
profile that sharpens the boundary-spike structure: a spike roughly `9–10×` the mean at one end, a couple of
hundred heights near zero, the middle third thinned to `~0.70×` the mean, and a large set of autoconvolution
nodes — some hundreds — packed within a hair of the peak, the flat top the LP drives down together. Each of
these is a falsifiable consequence of the boundary-spike-plus-plateau picture; if the returned vector were
smoothly humped with no sparse interior and few near-tight nodes, my account of why this problem descends would
be wrong. A basin dramatically below `1.517` would also surprise me and mean the rung-3 basin was far from the
best; I expect instead a small, honest improvement confirming the rung-3 basin was already good, and that this
is near the floor a single SLP constructor reaches.

That is what this rung really pins down: the frontier of a single diversified SLP constructor, with the record
`1.5028628969` standing above it as a *different-method* distance, not a harder-tuning one. The remaining
`~0.014` is not a tuning gap — more restarts, a longer polish, a different trust schedule change neither the
kind of solution the engine expresses nor the valley it can reach. Closing it needs something structural: far
more pieces, so the profile can hold the finely irregular record shape a `600`-node plateau cannot, and a search
ranging over structurally different constructions rather than linearizing around one. The record is a
`30000`-piece construction found by a large-scale agentic search over tens of hours — two orders of magnitude
more pieces and vastly more compute — and the gain from `1.5053` to `1.50286` lives entirely in the fourth
decimal, bought with that scale, not a cleverer local step. The honest number I report is what the evaluator
returns on the returned vector, and the gap from there down to `1.50286`, and the further gap to the provable
floor `1.28`, is the part of the first autocorrelation inequality this constructor leaves open.
