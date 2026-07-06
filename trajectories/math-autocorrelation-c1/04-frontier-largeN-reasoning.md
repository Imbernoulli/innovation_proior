The sequential-LP refinement did what the minimax diagnosis promised. The `β`-annealed Adam warm start reached
`1.526568` at `N = 600` — the plateau where the single-node gradient hits the balloon wall — and the
trust-region SLP then ground past it to `1.517237`, buying `0.0093` that the gradient could not, by pressing the
whole near-tight cluster of nodes down together. So the engine is right. But the feedback is equally clear about
where it stops: `1.517237` sits `~0.012` above the `600`-piece AlphaEvolve construction at `1.5053` and `~0.014`
above the record `1.5028628969`, and after the descent the run only crawled — restart kicks bought a few
ten-thousandths and no more. The single most important thing in that table is that AlphaEvolve reached `1.5053`
at the *very same* `N = 600`. The gap is therefore not resolution: `600` pieces demonstrably can express a shape
that scores `1.5053`, my constructor just did not find it. It is the honest signature of a single local
constructor — it converges into *a* good basin, not *the* good basin. So the endpoint question is sharp: with
the same SLP engine but more search breadth and, where it helps, a finer grid, how far down can I genuinely push,
and how close can I get to the record?

It is worth laying the whole descent out in one column, because the *shape* of the diminishing returns is itself
a reading. The ceiling was `2.000000`; annealing on `50` heights reached `1.537084`, a drop of `0.463`; lifting
to `600` and running the softmax-Adam warm start reached `1.526568`, another `0.010`; the trust-region SLP then
reached `1.517237`, another `0.0093`. Each rung buys about an order of magnitude less than the one before, and
the curve is flattening hard as it approaches the low `1.51`s. That is exactly the profile of a search running
out of the easy structure and pressing against something harder — and it lets me set an honest expectation for
this rung before I run it: not another `0.01`, but at most a few thousandths, because I am adding search breadth
and polish to an engine that has already found most of what it can find at this resolution. If this rung bought
another `0.01`, that would tell me the rung-3 basin was surprisingly far from the best `600`-piece basin; I do
not expect that, and reading the size of the gain is the way I will know whether the diagnosis "one warm start,
one basin" was the whole story or only part of it.

Two levers are available, and I should be clear-eyed about what each can and cannot buy before I spend budget on
it. The first is basin diversity. My rung-3 run was one warm start into one basin; an agentic search of the kind
that found `1.5053` effectively tries many structurally different shapes and keeps the best. I can imitate that
cheaply and honestly: launch the SLP from several structurally different initializations and keep the global
best. The question is which shapes to seed, and the previous rung answers it. The returned `600`-piece profile
was asymmetric with a tall spike at the boundary (about `9×` the mean at index `0`), mass heavier toward the
ends, the middle third thinned to `~0.70×` the mean, and roughly `215` of `600` heights driven near zero — it
had already drifted into the boundary-spike family. That family is where the good basins live for this problem:
concentrating mass into a tall spike at a boundary over a thinned, sparse interior is what suppresses the central
self-overlap that the flat function maximizes. So I seed not only from the rung-3 shape but from a spread of
explicit boundary-spike profiles, varying which end carries the heavier spike, since the optimum is asymmetric
and I do not know a priori which way it leans. Each start runs a slice of the budget, restart-kicking out of
trust collapse, and I keep whichever lands lowest.

Concretely the boundary-spike init starts from a flat vector of ones, ramps an added spike of height `s_L`
linearly into the first `n/12` pieces and one of height `s_R` into the last `n/12`, adds the full spike value at
the two endpoints, jitters the whole thing with small Gaussian noise, and normalizes. I sweep the pairs
`(s_L, s_R) ∈ {(10,4), (4,10), (8,8), (14,6), (6,14)}` — left-heavy, right-heavy, balanced, and two more
lopsided — so that across the five starts I have covered both orientations and a range of spike magnitudes. The
noise and the `n/12` ramp width are there so the SLP has a smooth, feasible shape to linearize from rather than a
discontinuous spike; a bare delta at one node would give the LP a degenerate near-singular Jacobian to work with.
This is the cheap, honest version of what an agentic search does — try several promising shapes, keep the
lowest, then grind the winner — and it is exactly targeted at the failure mode the feedback exposed, that one
warm start finds one basin.

The seed sweep has a structure I can exploit as a built-in consistency check, because the objective has a
reflection symmetry I should not ignore. Reversing a height vector reverses its autoconvolution, which leaves the
peak and the mass untouched, so `R(a) = R(a[::-1])` exactly — I can confirm it on the trivial case, `[1,2]` and
`[2,1]` both score `16/9`. That means the optimum comes in a mirror pair, and the seed pairs `(10,4)`/`(4,10)`
and `(14,6)`/`(6,14)` are mirror images of each other: they should descend to *equal* scores, one landing on a
right-heavy profile and the other on its left-heavy reflection. If they do, that is reassurance that the search
is behaving and the orientation genuinely does not matter; if two mirror seeds reached materially different
scores, that would flag a bug or a badly under-converged run. So covering both orientations is not wasted work —
it is redundancy I am using as a check, and it is why I do not need to guess in advance which end carries the
spike. The one seed that is *not* part of a mirror pair, the balanced `(8,8)`, is doing something different: it
tests the competing hypothesis that a symmetric two-spike profile — matching spikes at both ends — could beat the
lopsided single-dominant-spike shape. Which basin wins among the lopsided seeds and the balanced one is a genuine
question about the optimum's structure that I cannot settle from the rung-3 shape alone, and the multi-start
answers it empirically rather than by assumption.

The second lever is resolution, and here I have to be careful and quantitative, because a finer grid is
seductive and I have a specific reason to distrust it. Repeat-lifting the best shape to a finer grid is *free* in
principle: replacing each height by `m` copies produces the *same* step function, only subdivided, so it has the
same autoconvolution and the same `R`. I should verify that rather than assume it. Take `a = [1, 2]`, which
scores `R = 16/9 ≈ 1.7778`; lifting by copies to `[1,1,2,2]`, `[1,1,1,2,2,2]`, `[1,1,1,1,1,2,2,2,2,2]` gives
`1.7778` every time — the value is exactly invariant under the lift, as it must be since the underlying function
is unchanged. So a repeat-lift costs nothing and risks nothing; it just hands the SLP more coordinates to
coordinate, room to carve structure a coarse vector cannot hold. The catch is entirely in the cost of *using*
those coordinates. The LP at the heart of each SLP round scales with the number of pieces — `n + 1` variables
and up to `2n − 1` constraints — and even under the optimistic assumption that solve time grows only like
variables-times-constraints, doubling `n` doubles both factors and so quadruples the per-round cost; the
interior-point method's factorization step in practice grows faster still, so the realistic multiplier at
`N = 1200` is more like five-to-sixfold per round. Worse, the round *count* needed also rises: twice as many
coordinates is twice as much structure to coordinate, so the finer grid wants *more* rounds to converge, not the
same number. Putting those together, a single round at `N = 1200` is a handful of seconds rather than the tens of
milliseconds it is at `600`, and for the same wall-clock budget I can afford only roughly a sixth as many rounds
while needing more of them. So a finer grid buys representational room and spends the round
budget fast, and the trade is delicate: the LP solve, not the FFT, is the bottleneck, and it is the SLP's
iteration count that actually finds the lower basin.

I resolve that trade by trying the lift and reporting honestly what it does. A repeat-lift to `N = 1200`,
followed by SLP polishing there, does not pay within budget: the per-round cost rises with the grid and the finer
grid does not recover below the lift value before the rounds run out. That is not a failure of the idea that more
resolution *could* help — the record uses `50×` more pieces than this — it is a statement that at *my* budget the
extra coordinates cannot be exploited fast enough to beat the coarser grid where the SLP can iterate freely. So
the returned frontier stays at `N = 600`, where the SLP is fast enough to do the many restarts that actually
find a lower basin, and I spend the budget on the multi-start search plus a long polish rather than on a grid I
cannot afford to work. That is the disciplined reading of the negative result: not "resolution is useless" but
"this constructor's bottleneck is the LP solve, and at this budget breadth beats depth-of-grid."

I did consider abandoning the SLP for a fundamentally broader search at this rung — a genetic algorithm or a
large population-based evolutionary method over the `600`-dimensional height vector, which is closer in spirit to
what the record-holding agentic search does. I rejected it for a concrete budget reason rather than a taste one.
The SLP's strength is local precision: it drives a whole plateau of near-tight nodes down together, resolving the
fourth decimal, which is exactly the regime I am now in near `1.517`. A GA over `600` dimensions spends its
budget exploring coarsely and would need an enormous number of evaluations to match that local precision; within
a `~20`-minute budget it would almost certainly land well above where the SLP already sits, because it has no
mechanism as sharp as the minimax LP for flattening a plateau. The right use of a broad search here is not to
replace the SLP but to *seed* it — which is precisely what the multi-start does. So I keep the precise local
engine and buy breadth by giving it several starting basins, rather than trading the precision away for breadth
I cannot afford to exploit.

It also helps to read where `1.517` sits in the peak/mean and plateau terms, because it says how little room is
left at this resolution. At `N = 600` the prefactor `2N/(2N−1) = 1.00083` is negligible, so `ratio ≈ peak/mean`:
the rung-3 profile has its tallest autoconvolution node at about `1.516×` its average node, against the record's
`~1.5016×` and the flat ceiling's `2×`. The residual `~0.014` to the record is thus a demand to flatten the top
of the node profile by a further one part in a hundred. And the rung-3 profile is already remarkably flat on top
— it reportedly has a few hundred of its `1199` autoconvolution nodes packed within `10^{−3}` of the peak, so
roughly a third of all nodes are already at the plateau the LP built. That is close to the expressivity limit of
`600` pieces: I cannot flatten more nodes than exist, and a large fraction are already flat. It is another way of
seeing why the last fraction of a percent will not come from this grid — the plateau is nearly as wide as the
node set can hold — and why the honest move, past this endpoint, would be a different construction entirely
rather than more of the same engine.

The mechanics of the SLP are unchanged from the previous rung — epigraph variable `z` for the peak, node
constraints `b_k = (a*a)_k ≤ z` linearized as `b_k(a + d) ≈ b_k(a) + Σ_j 2 a_{k−j} d_j`, mass held fixed, trust
region bounding `|d_j|`, accept only if the true `R` drops, grow on success and shrink-then-restart-kick on
rejection, top-K focus on the near-tight nodes with occasional full passes — and only the *schedule* around it
changes. Most of the budget goes to the multi-start search at `N = 600`, because that is where I expect the real
gain: finding a basin lower than the single rung-3 one. The arithmetic of that schedule is `six` starts of about
`200` SLP rounds each plus a final polish of about `250` rounds — roughly `1450` LP solves in total. I run these
as full-constraint passes now rather than top-K, because near the floor I cannot afford to miss a node quietly
rising into the peak, and at `N = 600` a full-constraint `linprog` over all `1199` node constraints is on the
order of a second per solve; `1450` such solves is on the order of `20` minutes, which is the budget I have. That
is precisely why the finer grid is off the table: the same `20` minutes at `N = 1200`, with per-solve cost up
five- or six-fold, would buy only a couple of hundred rounds total — not enough to run six diverse starts to
convergence and polish the winner. So the schedule is set by the budget: breadth of starts and depth of polish at
the resolution where a round is cheap. After the search identifies the best basin, I give that
best a long dedicated full-constraint SLP polish at a tightened trust (`~6·10^{−5}`, slower growth) with restart
kicks, checkpointing the global best continuously so a long run never loses ground. The tighter trust for the
polish is deliberate and follows the same error arithmetic as before: near the floor the genuinely improving
steps are tiny, and the dropped quadratic term `~n·tr^2` has to stay well under the sub-thousandth improvements I
am now chasing, so a trust of `~6·10^{−5}` rather than `10^{−4}`, grown more slowly (`×1.04` rather than
`×1.05`), keeps each step inside the region where the linear model is trustworthy to the precision that matters.
A looser trust here would repeatedly overshoot onto an unmodelled node, get rejected, and waste rounds; a tighter
one late is how the polish keeps making real, if small, progress.

What do I expect the harness to report? The multi-start search should find a basin a little below `1.517` — the
boundary-spike seeds are in the right family, and several of them exploring should at least confirm which basin
is lowest and shave a little off the single warm start — landing somewhere in the low `1.51`s, edging toward the
AlphaEvolve `600`-piece `1.5053`, with the long polish inching it a bit lower still. On the metric that is
`pieces_N = 600` and a `ratio` a hair under `1.517`, on a profile that should sharpen the peak-suppressing
boundary-spike structure: a spike roughly `9–10×` the mean at one end, a couple of hundred of the `600` heights
near zero, the middle third thinned to `~0.70×` the mean, and — the tell that the minimax is doing its job — a
large set of autoconvolution nodes, some hundreds of them, packed within a hair of the peak, the flat-top plateau
the LP drives down together. If the multi-start found a basin dramatically below `1.517`, that would surprise me
and mean the rung-3 basin was far from the best available; I expect instead a small, honest improvement that
confirms the rung-3 basin was already good and this is near the floor a single SLP constructor reaches.

I can make those predictions concrete enough to check against the returned vector rather than leave them as
adjectives. At `N = 600` with the prefactor negligible, a `ratio` of `~1.517` means `peak/mean ≈ 1.516`, so the
returned profile's tallest autoconvolution node should sit almost exactly `1.516×` its average node — a number I
can read off directly. The plateau should be wide: I expect a few hundred of the `1199` nodes within `10^{−3}`
of the peak, because that flat top is precisely what the minimax LP builds, and if instead only a handful of
nodes were near-tight the LP would have had nothing to press on and the descent would have stalled far higher.
And the heights should be end-loaded and sparse — a spike near `9–10×` the mean at one boundary, a couple of
hundred heights near zero, the middle third thinned to roughly `0.70×` the mean. Each of these is a falsifiable
consequence of the boundary-spike-plus-plateau picture; if the returned vector were, say, smoothly humped with no
sparse interior and few near-tight nodes, my whole account of why this problem descends would be wrong. So the
returned profile is not just an output to report but a set of predictions to confront.

That framing — "the floor a single SLP constructor reaches" — is what this rung is really pinning down, and the
diminishing-returns column is the evidence it is close. The ladder went `2.000 → 1.537 → 1.527 → 1.517`, each
step roughly a tenth the size of the last, and this rung's expected few-thousandths would continue that geometric
tail. A search converging geometrically toward a value is telling me that value is an asymptote, not a way
station — the point past which *this* constructor, however I schedule its restarts and polish, does not go.
Recognizing that is not giving up; it is correctly identifying that the remaining distance is not a tuning
distance. More restarts of the same LP, a longer polish, a slightly different trust schedule — none of these
changes the kind of solution the engine can express or the kind of valley it can reach. Whatever closes the gap
to `1.50286` has to change something structural: far more pieces, so the profile can hold the finely irregular
record shape the `600`-node plateau cannot, and a search that ranges over structurally different constructions
rather than linearizing around one. That is beyond what I set out to do here, and naming it precisely is the
honest endpoint of this rung — the single-constructor frontier, with the record standing above it as a
different-method distance, not a harder-tuning one.

I am honest, though, that I do *not* expect to reach `1.5028628969`, and I want to say why precisely rather than
wave at it. That record is a `30000`-piece deliberately irregular construction found by a large-scale agentic
search running for tens of hours — two orders of magnitude more pieces and vastly more compute than a single
bounded SLP run on a `600`-piece grid commands. The gain from `1.5053` to `1.50286` lives entirely in the fourth
decimal place, below the third, and it was bought with that scale, not with a cleverer local step. My constructor
is a single trust-region engine linearizing around one point at a time; even diversified over several basins and
polished long, it converges into a good basin but not the global one, and at `600` pieces it cannot even express
the finely irregular record shape. So the endpoint of this ladder is the frontier a single diversified SLP
constructor can reach, with the record `1.5028628969` standing above as the still-open distance — exactly the way
the `30000`-piece record stands above any modest-grid local construction. The honest number I report is the one
the evaluator returns on the returned vector, and the remaining gap from there down to `1.50286`, and the further
gap down to the provable floor `1.28`, is the part of the first autocorrelation inequality this constructor leaves
open.
